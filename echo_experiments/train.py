"""
CLI entrypoint for CRAFT builder training -- the three GRPO-family
conditions from the within-turn experiment design (ECHO, episode_return,
rloo_per_turn). 

    python train.py --mode echo --steps 300
    python train.py --mode episode_return --steps 300
    python train.py --mode rloo_per_turn --steps 300

Run `python data_split.py` first to generate the training-only structure
pool this reads from. Each --mode trains an otherwise identical checkpoint
(same base model, LoRA config, turn budget, oracle_n) differing only in how
advantages.compute_multiturn_advantages assigns credit -- that's the one
independent variable the within-turn eval (eval_within_turn.py) is set up
to compare.
"""
import argparse
import datetime
import os
from pathlib import Path


def _pre_parse_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--director_gpu", type=int, default=1)
    known, _ = p.parse_known_args()
    if known.gpu == known.director_gpu:
        return known.gpu, known.director_gpu, str(known.gpu)
    return known.gpu, known.director_gpu, f"{known.gpu},{known.director_gpu}"


_GPU, _DIRECTOR_GPU, _CUDA_VISIBLE_DEVICES = _pre_parse_gpus()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _CUDA_VISIBLE_DEVICES)
# Local index of the director model within CUDA_VISIBLE_DEVICES: same
# device as the builder (local cuda:0) when --gpu == --director_gpu, else
# the second entry we just added to CUDA_VISIBLE_DEVICES (local cuda:1).
_LOCAL_DIRECTOR_GPU = 0 if _GPU == _DIRECTOR_GPU else 1

import torch
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from data_split import load_training_pool
from reward import BuilderRewardFunction
from trainer import CRAFTEchoTrainer, Fixed_GRPOConfig

load_dotenv()

# Local director model keys -- subset of run_craft.py's LOCAL_MODELS relevant
# to the within-turn experiments so far.
LOCAL_DIRECTOR_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def build_structure_dataset(structures):
    rows = [{"prompt": f"CRAFT structure {i}", "structure_idx": i} for i in range(len(structures))]
    return HFDataset.from_list(rows)


def build_grpo_config(
    run_name, output_dir, per_device_train_batch_size, num_generations,
    max_prompt_length, max_completion_length, max_steps, learning_rate,
    kl_coef, logging_steps, save_steps, report_to, seed, temperature,
):
    return Fixed_GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        generation_batch_size=per_device_train_batch_size * num_generations,
        steps_per_generation=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        learning_rate=learning_rate,
        beta=kl_coef,
        logging_steps=logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        report_to=report_to,
        seed=seed,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        max_grad_norm=0.5,
        bf16=True,
        remove_unused_columns=False,
    )


def train(
    mode="echo",
    steps=300,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    train_pool_path=None,
    oracle_n=5,
    max_turns=8,
    num_generations=3,
    per_device_train_batch_size=2,
    max_prompt_length=3584,  # if oracle moves set to 20, needs longer prompt budget
    max_completion_length=220,
    learning_rate=5e-6,
    temperature=1.0,
    director_model_name="gpt-4.1-mini",
    director_mode="api",
    director_gpu=0,
    director_max_new_tokens=448,
    director_quantize=None,
    log_dir="craft_echo_runs",
    save_every=25,
    seed=42,
    report_to="none",
    logprob_batch_size=16,
    train_micro_batch_size=4,
    resume_from_checkpoint=None,
):
    assert mode in ("echo", "episode_return", "rloo_per_turn"), (
        f"unknown mode {mode!r} -- choose echo, episode_return, or rloo_per_turn "
        "(PPO is a separate pipeline, see trainer.py's module docstring)"
    )
    set_seed(seed)

    structures = load_training_pool(train_pool_path) if train_pool_path else load_training_pool()
    dataset = build_structure_dataset(structures)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"craft_{mode}_{model_name.split('/')[-1]}_seed{seed}_{now}"
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Real builder prompts (spatial reference + few-shot examples + oracle
    # candidates + live discussion) run ~2700-2800 tokens -- the previous
    # max_prompt_length=2048 default truncated every one of them. Default
    # truncation_side is "right", which cuts off the END of the prompt --
    # exactly where the live board state, director discussion, and oracle
    # candidates live -- leaving the model to autocomplete a fragment of the
    # static preamble instead of answering. Truncate from the left instead
    # (drops older static reference material first) as a safety net on top
    # of the larger max_prompt_length below.
    tokenizer.truncation_side = "left"

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda:0")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    shared_director_model, shared_director_tokenizer = None, None
    if director_mode == "local":
        from local_model_utils import load_local_director_pipeline
        director_model_path = LOCAL_DIRECTOR_MODELS.get(director_model_name, director_model_name)
        print(f"loading shared local director model {director_model_path} on cuda:{director_gpu}...")
        shared_director_model, shared_director_tokenizer = load_local_director_pipeline(
            director_model_path, quantize=director_quantize, gpus=[director_gpu],
            max_new_tokens=director_max_new_tokens,
        )

    reward_fn = BuilderRewardFunction()
    grpo_config = build_grpo_config(
        run_name=run_name,
        output_dir=str(run_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=steps,
        learning_rate=learning_rate,
        kl_coef=0.0,
        logging_steps=1,
        save_steps=save_every,
        report_to=report_to,
        seed=seed,
        temperature=temperature,
    )
    # With director_mode="local" both physical GPUs are visible to this
    # process (builder on cuda:0, director pipeline on cuda:1), so
    # TrainingArguments.n_gpu auto-detects 2 -- which makes
    # transformers.Trainer._wrap_model silently wrap the *builder* model in
    # nn.DataParallel across both cards (fighting the director pipeline
    # already resident on cuda:1) the moment .train() starts. That's a
    # read-only property backed by ._n_gpu; overriding it here forces
    # single-GPU training regardless of how many devices are visible, without
    # having to hide the second GPU (which the local director needs).
    grpo_config._n_gpu = 1

    trainer = CRAFTEchoTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        structures=structures,
        advantage_mode=mode,
        oracle_n=oracle_n,
        max_turns=max_turns,
        director_model_name=director_model_name,
        director_mode=director_mode,
        shared_director_model=shared_director_model,
        shared_director_tokenizer=shared_director_tokenizer,
        run_id=seed,
        logprob_batch_size=logprob_batch_size,
        train_micro_batch_size=train_micro_batch_size,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(run_dir / "final_model"))
    tokenizer.save_pretrained(str(run_dir / "final_model"))
    print(f"done -> {run_dir / 'final_model'}")
    return str(run_dir / "final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CRAFT builder with ECHO / episode_return / rloo_per_turn")
    parser.add_argument("--gpu", type=int, default=0,
                        help="physical GPU index for the builder model -- consumed before torch is imported "
                             "(_pre_parse_gpus above), so it takes effect via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mode", choices=["echo", "episode_return", "rloo_per_turn"], default="echo")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_pool", type=str, default=None, help="path from data_split.py; default data/train_structures.json")
    parser.add_argument("--oracle_n", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=8,
                        help="episode turn cap -- each turn costs 3 frozen director generations + 1 builder "
                             "generation, so this is the single biggest lever on rollout wall-clock")
    parser.add_argument("--num_generations", type=int, default=3,
                        help="G -- parallel rollouts per structure per step. Episodes per generation phase = "
                             "batch_size * num_generations^2 (confirmed against trainer.py's "
                             "_generate_and_score_completions), so this scales rollout cost quadratically -- "
                             "at ~89 sec/episode (observed), G=3/batch_size=2 runs ~26.6 min/step vs ~5.9 "
                             "min/step at the old G=2/batch_size=1 defaults")
    parser.add_argument("--batch_size", type=int, default=2, help="structures per step -- scales rollout cost linearly")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--director_model", type=str, default="gpt-4.1-mini",
                        help="api: model name (e.g. gpt-4.1-mini); local: key from LOCAL_DIRECTOR_MODELS "
                             f"({list(LOCAL_DIRECTOR_MODELS)}) or a full HF path")
    parser.add_argument("--director_mode", type=str, default="api", choices=["api", "local"],
                        help="api calls OpenAI for each director (default -- matches run_craft.py's own "
                             "builder convention and this session's baseline-check finding that local "
                             "Mistral directors weren't a clear win over API ones). local reuses one shared "
                             "open-weight model across all three directors instead -- see "
                             "local_model_utils.load_local_director_pipeline")
    parser.add_argument("--director_gpu", type=int, default=1,
                        help="physical GPU index for the shared local director model -- consumed before torch "
                             "is imported, same as --gpu. Defaults to a different physical GPU than --gpu (1 "
                             "vs 0), so builder and director share the two cards instead of both loading onto "
                             "one; pass the same value as --gpu to force them back onto a single device")
    parser.add_argument("--director_max_new_tokens", type=int, default=448,
                        help="cap on the local director's generation length. Measured against the actual "
                             "Mistral-7B pipeline with no cap: natural <think>+<message> completions ran "
                             "120-290 tokens (p90=275) on fresh turns, with longer conversation history "
                             "pushing some responses higher -- 256 was cutting off ~60%% of turns before "
                             "the closing </message> tag (confirmed via missing-closing-tag rate in a real "
                             "run's logs), silently truncating live director instructions. 448 leaves real "
                             "margin without going back to the original flat 512.")
    parser.add_argument("--director_quantize", type=str, default=None, choices=[None, "4bit", "8bit"],
                        help="quantization for the local director model, if --director_mode local")
    parser.add_argument("--log_dir", type=str, default="craft_echo_runs")
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--logprob_batch_size", type=int, default=16,
                        help="inference-only (no-grad) chunk size for the post-rollout old/ref logprob "
                             "pass -- independent of --train_micro_batch_size (the actual gradient-computing "
                             "forward+backward chunk size); no-grad passes need far less memory per sequence")
    parser.add_argument("--train_micro_batch_size", type=int, default=4,
                        help="max turn-sequences forward+backward'd through the model at once during the "
                             "actual gradient step. _split_by_episodes bundles steps_per_generation's worth "
                             "of full episodes (batch_size x num_generations turns) into one training_step "
                             "call for GRPO group-size reasons unrelated to GPU memory -- at the 7B model's "
                             "prompt/completion lengths that can OOM (confirmed: 6 episodes x up to 20 turns "
                             "= up to 120 sequences, one MLP alloc needing 13.53 GiB). This re-chunks that "
                             "slice into micro-batches with the standard gradient-accumulation loss scaling, "
                             "so only one chunk's activations are resident at a time -- lower if still OOMing, "
                             "raise for speed if memory allows")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="path to a checkpoint-N directory (e.g. craft_echo_runs/<run_name>/checkpoint-175) "
                             "to continue from -- preserves that run's already-trained LoRA weights, optimizer "
                             "state, and global_step rather than starting over. Writes into a fresh run_dir/wandb "
                             "run rather than the original one, so the before/after reward-formulation change is "
                             "visible as two adjacent curves instead of overwriting history")
    args = parser.parse_args()
    if _GPU == _DIRECTOR_GPU:
        print(f"physical GPU: {_GPU} (builder + local director share it, visible as cuda:0)")
    else:
        print(f"physical GPUs: builder={_GPU} (local cuda:0), director={_DIRECTOR_GPU} (local cuda:1)")

    train(
        mode=args.mode,
        steps=args.steps,
        model_name=args.model,
        train_pool_path=args.train_pool,
        oracle_n=args.oracle_n,
        max_turns=args.max_turns,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        director_model_name=args.director_model,
        director_mode=args.director_mode,
        director_gpu=_LOCAL_DIRECTOR_GPU,
        director_max_new_tokens=args.director_max_new_tokens,
        director_quantize=args.director_quantize,
        log_dir=args.log_dir,
        save_every=args.save_every,
        seed=args.seed,
        report_to=args.report_to,
        logprob_batch_size=args.logprob_batch_size,
        train_micro_batch_size=args.train_micro_batch_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
