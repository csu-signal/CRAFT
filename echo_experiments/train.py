"""
Train a CRAFT builder with GRPO.

    python train.py --mode echo --steps 300
    python train.py --mode episode_return --steps 300
    python train.py --mode rloo_per_turn --steps 300

Requires the training pool from `python data_split.py`.
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
    max_prompt_length=3584,
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
    assert mode in ("echo", "episode_return", "rloo_per_turn"), f"unknown mode {mode!r}"
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
    # live board state and discussion are at the end of the prompt
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
    # prevent Trainer from wrapping the builder in DataParallel when the director's GPU is visible
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
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index for the builder model")
    parser.add_argument("--mode", choices=["echo", "episode_return", "rloo_per_turn"], default="echo")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_pool", type=str, default=None, help="path from data_split.py; default data/train_structures.json")
    parser.add_argument("--oracle_n", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=8, help="episode turn cap")
    parser.add_argument("--num_generations", type=int, default=3,
                        help="rollouts per structure; episodes per generation phase = batch_size * num_generations^2")
    parser.add_argument("--batch_size", type=int, default=2, help="structures per step")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--director_model", type=str, default="gpt-4.1-mini",
                        help="api: model name (e.g. gpt-4.1-mini); local: key from LOCAL_DIRECTOR_MODELS "
                             f"({list(LOCAL_DIRECTOR_MODELS)}) or a full HF path")
    parser.add_argument("--director_mode", type=str, default="api", choices=["api", "local"],
                        help="api: OpenAI directors; local: one shared open-weight model for all three")
    parser.add_argument("--director_gpu", type=int, default=1,
                        help="physical GPU index for the local director model; same as --gpu to share one device")
    parser.add_argument("--director_max_new_tokens", type=int, default=448,
                        help="local director generation cap")
    parser.add_argument("--director_quantize", type=str, default=None, choices=[None, "4bit", "8bit"],
                        help="quantization for the local director model, if --director_mode local")
    parser.add_argument("--log_dir", type=str, default="craft_echo_runs")
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--logprob_batch_size", type=int, default=16,
                        help="chunk size for the no-grad old/ref logprob pass")
    parser.add_argument("--train_micro_batch_size", type=int, default=4,
                        help="sequences per forward+backward chunk; lower if OOM")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="checkpoint-N directory to resume from (writes to a new run dir)")
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
