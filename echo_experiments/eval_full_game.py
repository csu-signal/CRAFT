"""
Full-game evaluation of a local builder -- a trained LoRA checkpoint or an
untrained base model of any size -- through complete multi-turn episodes
against frozen directors on held-out structures. The protocol (structures,
episodes per structure, seeds, starting boards, oracle setting) lives in
eval_harness.py and is shared with baseline_sanity_check.py's API builders.

Generation mirrors trainer.py's generate_fn (same chat template and
oracle/base system prompt selection) so a checkpoint is evaluated the way it
was trained.

Usage (run from echo_experiments/, with HF_HOME=/data/huggingface_cache):
    # zero-shot base models
    python eval_full_game.py --label base_7b  --episodes_per_structure 5
    python eval_full_game.py --label cot_7b   --episodes_per_structure 5 --prompt_style cot
    python eval_full_game.py --label base_14b --episodes_per_structure 5 --base_model Qwen/Qwen2.5-14B-Instruct
    python eval_full_game.py --label base_72b --episodes_per_structure 5 --base_model Qwen/Qwen2.5-72B-Instruct --quantize 4bit

    # trained checkpoints
    python eval_full_game.py --label echo --episodes_per_structure 5 --checkpoint /data/craft_echo_runs/<run>/checkpoint-350

    # no-oracle ablation: same flags + --no_oracle --out_dir eval_results_no_oracle
"""
import argparse
import os
import sys
from pathlib import Path


def _pre_parse_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--director_mode", type=str, default="api")
    p.add_argument("--director_gpu", type=int, default=1)
    known, _ = p.parse_known_args()
    gpus = known.gpus
    if known.director_mode == "local":
        gpus += f",{known.director_gpu}"
    return gpus, len(known.gpus.split(","))


# Builder runs on --gpus (default physical GPU 0; pass e.g. 0,1 to shard a
# large model). A local director is appended as the last visible device.
_VISIBLE, _N_BUILDER_GPUS = _pre_parse_gpus()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _VISIBLE)

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from local_model_utils import load_local_director_pipeline
from eval_harness import DEFAULT_MAX_TOKENS, add_protocol_args, builder_system_prompt, run_eval

load_dotenv()

TRAIN_MAX_PROMPT_LENGTH = 3584  # train.py's max_prompt_length

LOCAL_DIRECTOR_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def load_builder(base_model, checkpoint=None, quantize=None, n_gpus=1):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint or base_model)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": torch.bfloat16}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantize == "4bit", load_in_8bit=quantize == "8bit",
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    if quantize or n_gpus > 1:
        # only the builder's GPUs -- a local director sits on the last visible device
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {i: torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if "device_map" not in kwargs:
        model = model.to("cuda:0")
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tokenizer


def make_checkpoint_generate_fn(model, tokenizer, max_prompt_length, max_new_tokens, temperature, prompt_style):
    """Mirrors trainer.py's generate_fn (chat template, oracle/base system
    prompt, top_p=0.9) plus the optional CoT instruction. temperature=0 -> greedy."""
    device = next(model.parameters()).device
    stats = {"prompt_tokens_max": 0, "prompts_over_train_limit": 0, "prompts_truncated": 0}

    def generate_fn(prompt_text, oracle_moves):
        chat_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": builder_system_prompt(bool(oracle_moves), prompt_style)},
                {"role": "user", "content": prompt_text},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)
        stats["prompt_tokens_max"] = max(stats["prompt_tokens_max"], input_ids.shape[1])
        stats["prompts_over_train_limit"] += int(input_ids.shape[1] > TRAIN_MAX_PROMPT_LENGTH)
        if max_prompt_length and input_ids.shape[1] > max_prompt_length:
            # left truncation (training's behaviour) drops the system prompt and task instructions
            stats["prompts_truncated"] += 1
            input_ids = input_ids[:, -max_prompt_length:]
        input_ids = input_ids.to(device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature > 0 else dict(do_sample=False)
        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, **sampling,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return input_ids[0].cpu(), new_tokens.cpu(), decoded

    return generate_fn, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_protocol_args(parser)
    g = parser.add_argument_group("local builder")
    g.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint dir; omit for the zero-shot base model")
    g.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    g.add_argument("--quantize", type=str, default=None, choices=["4bit", "8bit"])
    g.add_argument("--gpus", type=str, default="0", help="builder GPU(s), e.g. 0 or 0,1")
    g.add_argument("--temperature", type=float, default=1.0, help="0 = greedy")
    g.add_argument("--max_new_tokens", type=int, default=None,
                   help=f"default: {DEFAULT_MAX_TOKENS['default']} (default style) / {DEFAULT_MAX_TOKENS['cot']} (cot)")
    g.add_argument("--max_prompt_length", type=int, default=0,
                   help="0 = never truncate (default). Training left-truncated at 3584, which cuts off the "
                        "system prompt; prompts over that length are counted in the results either way")
    g = parser.add_argument_group("local director (default: API directors)")
    g.add_argument("--director_mode", type=str, default="api", choices=["api", "local"])
    g.add_argument("--director_gpu", type=int, default=1)
    g.add_argument("--director_max_new_tokens", type=int, default=448)
    g.add_argument("--director_quantize", type=str, default=None, choices=[None, "4bit", "8bit"])
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_TOKENS[args.prompt_style]
    print(f"[eval-full-game] label={args.label!r} checkpoint={args.checkpoint!r} base_model={args.base_model!r} "
          f"quantize={args.quantize} gpus={args.gpus} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    model, tokenizer = load_builder(args.base_model, args.checkpoint, args.quantize, _N_BUILDER_GPUS)
    generate_fn, gen_stats = make_checkpoint_generate_fn(
        model, tokenizer, args.max_prompt_length, max_new_tokens, args.temperature, args.prompt_style)

    director_setup = {"director_mode": args.director_mode}
    if args.director_mode == "local":
        path = LOCAL_DIRECTOR_MODELS.get(args.director_model, args.director_model)
        local_idx = _N_BUILDER_GPUS  # appended after the builder's GPUs in CUDA_VISIBLE_DEVICES
        print(f"loading shared local director model {path} on physical GPU {args.director_gpu}...")
        pipe, tok = load_local_director_pipeline(path, quantize=args.director_quantize, gpus=[local_idx],
                                                 max_new_tokens=args.director_max_new_tokens)
        director_setup.update(director_model_path=path, director_pipe=pipe, director_tok=tok)

    config = {
        "builder_kind": "local", "base_model": args.base_model, "checkpoint": args.checkpoint,
        "quantize": args.quantize, "temperature": args.temperature, "top_p": 0.9 if args.temperature > 0 else None,
        "max_new_tokens": max_new_tokens, "max_prompt_length": args.max_prompt_length,
        "generation_stats": gen_stats,  # updated during the run; saved with the results
    }
    run_eval(args, generate_fn, config, checkpoint=args.checkpoint, director_setup=director_setup)
    print(f"[eval-full-game] longest prompt {gen_stats['prompt_tokens_max']} tokens; "
          f"{gen_stats['prompts_over_train_limit']} over training's {TRAIN_MAX_PROMPT_LENGTH}; "
          f"{gen_stats['prompts_truncated']} truncated")


if __name__ == "__main__":
    main()
