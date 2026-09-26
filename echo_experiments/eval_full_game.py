"""
Full-game evaluation: run a trained checkpoint (or the untrained base model)
as the builder through complete multi-turn episodes against frozen
directors, on the held-out benchmark structures (data/structures_dataset_20.json
-- never used for training, see data_split.py) rather than the training pool.

This replaces the original within-turn eval (a single frozen decision point)
now that training shows real final_progress signal worth measuring end to
end: does the trained policy actually get closer to progress=1 over a
complete episode, not just pick well at one frozen snapshot.

Reuses the exact same rollout.run_builder_episode used by both training and
the earlier baseline_sanity_check.py/baseline_tools_check.py comparisons --
the only new piece is a generate_fn that loads a LoRA checkpoint locally and
generates with it, matching trainer.py's own generate_fn (same chat
template, same system prompt selection) so the eval faithfully reflects what
was actually trained.

Usage:
    # a trained checkpoint
    python eval_full_game.py --checkpoint craft_echo_runs/craft_echo_.../checkpoint-200 --label echo_step200 --report_to wandb

    # the untrained base model, for a zero-shot reference point
    python eval_full_game.py --label base --report_to wandb

    # against local Mistral directors instead of the API default, if that's what training used
    python eval_full_game.py --checkpoint ... --label echo_step200 --director_mode local --director_model mistral-7b --director_gpu 1
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from agents.builder_agent import BUILDER_SYSTEM_PROMPT_ORACLE, BUILDER_SYSTEM_PROMPT_BASE
from data_split import load_benchmark_structures
from local_model_utils import load_local_director_pipeline
from rollout import run_builder_episode

load_dotenv()

LOCAL_DIRECTOR_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

BUILDER_LOG_KEYS = [
    "progress_delta", "matched_oracle", "parse_failure", "move_invalid", "completed",
    "off_list_penalty", "invalid_move_penalty", "clarify_penalty",
    "completion_bonus", "efficiency_bonus", "action", "training_reward", "final_progress",
]


def _safe_mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_rate(bools):
    vals = [1.0 if bool(v) else 0.0 for v in bools if v is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(reward_infos, group_sizes):
    L = {k: [info[k] for info in reward_infos if k in info] for k in BUILDER_LOG_KEYS}
    rewards = [info.get("training_reward", 0.0) for info in reward_infos]
    return {
        "builder/final_progress_mean": _safe_mean(L["final_progress"]),
        "builder/completed_rate": _safe_rate(L["completed"]),
        "builder/oracle_match_rate": _safe_rate([v for v in L["matched_oracle"] if v is not None]),
        "builder/clarify_rate": _safe_rate([a == "clarify" for a in L["action"]]),
        "builder/parse_failure_rate": _safe_rate(L["parse_failure"]),
        "builder/invalid_move_rate": _safe_rate(L["move_invalid"]),
        "builder/progress_delta_mean": _safe_mean(L["progress_delta"]),
        "builder/mean_episode_length": float(sum(group_sizes) / len(group_sizes)) if group_sizes else float("nan"),
        "reward": _safe_mean(rewards),
        "reward_std": (torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0),
    }


def make_checkpoint_generate_fn(model, tokenizer, device, max_prompt_length, max_completion_length, temperature):
    """Mirrors trainer.py's _run_single_episode.generate_fn exactly (same
    chat template, same oracle/base system prompt selection, same sampling
    params) so this eval reflects what the checkpoint was actually trained
    to do -- just without unwrap_model_for_generation/accelerator, since
    there's no distributed training context here, just a loaded model."""
    def generate_fn(prompt_text, oracle_moves):
        system_prompt = BUILDER_SYSTEM_PROMPT_ORACLE if oracle_moves else BUILDER_SYSTEM_PROMPT_BASE
        chat_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(
            chat_text, return_tensors="pt",
            truncation=True, max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return input_ids[0].cpu(), new_tokens.cpu(), decoded

    return generate_fn


def run(
    checkpoint=None,
    base_model="Qwen/Qwen2.5-7B-Instruct",
    label="base",
    director_model_name="gpt-4.1-mini",
    director_mode="api",
    director_gpu=1,
    director_max_new_tokens=448,
    director_quantize=None,
    oracle_n=20,
    max_turns=20,
    n_structures=20,   # the benchmark set only has 20 structures total
    episodes_per_structure=1,
    seed=42,
    temperature=1.0,
    max_prompt_length=3584,
    max_completion_length=220,
    log_every_episodes=5,
    report_to="wandb",
    run_name=None,
):
    print(f"[eval-full-game] label={label!r} checkpoint={checkpoint!r} base_model={base_model!r}")
    print(f"[eval-full-game] director_mode={director_mode!r} director_model={director_model_name!r}")

    structures = load_benchmark_structures()
    import random
    rng = random.Random(seed)
    structure_indices = rng.sample(range(len(structures)), min(n_structures, len(structures)))

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint or base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint).to(device)
    model.eval()

    generate_fn = make_checkpoint_generate_fn(
        model, tokenizer, device, max_prompt_length, max_completion_length, temperature,
    )

    director_pipe, director_tok, director_model_path = None, None, director_model_name
    if director_mode == "local":
        director_model_path = LOCAL_DIRECTOR_MODELS.get(director_model_name, director_model_name)
        print(f"loading shared local director model {director_model_path} on cuda:{director_gpu}...")
        director_pipe, director_tok = load_local_director_pipeline(
            director_model_path, quantize=director_quantize, gpus=[director_gpu],
            max_new_tokens=director_max_new_tokens,
        )
    else:
        print(f"using OpenAI API directors: {director_model_path} (no local model loaded)")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = run_name or f"fullgame_{label}_seed{seed}_{now}"

    use_wandb = report_to == "wandb"
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "craft_echo"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            group="eval_full_game",
            config={
                "label": label, "checkpoint": checkpoint, "base_model": base_model,
                "director_mode": director_mode, "director_model": director_model_name,
                "oracle_n": oracle_n, "max_turns": max_turns,
                "n_structures": n_structures, "episodes_per_structure": episodes_per_structure,
            },
        )

    all_infos_window, group_sizes_window = [], []
    all_infos_total, group_sizes_total = [], []
    episode_num = 0
    total_episodes = len(structure_indices) * episodes_per_structure

    for structure_idx in structure_indices:
        structure_data = structures[structure_idx]
        for _ in range(episodes_per_structure):
            episode_num += 1
            print(f"[eval-full-game] episode {episode_num}/{total_episodes} (structure_idx={structure_idx})")
            episode = run_builder_episode(
                structure_data=structure_data,
                generate_fn=generate_fn,
                structure_index=structure_idx,
                run_id=seed,
                oracle_n=oracle_n,
                max_turns=max_turns,
                director_model_name=director_model_path,
                director_mode=director_mode,
                director_api_key=os.getenv("OPENAI_API_KEY") if director_mode == "api" else None,
                shared_director_model=director_pipe,
                shared_director_tokenizer=director_tok,
                seed=seed * 7919 + episode_num,
            )
            all_infos_window.extend(episode["reward_infos"])
            group_sizes_window.append(len(episode["rewards"]))
            all_infos_total.extend(episode["reward_infos"])
            group_sizes_total.append(len(episode["rewards"]))

            if episode_num % log_every_episodes == 0 or episode_num == total_episodes:
                metrics = summarize(all_infos_window, group_sizes_window)
                print(f"  [{episode_num}/{total_episodes}] " + " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in metrics.items()))
                if use_wandb:
                    wandb.log(metrics, step=episode_num)
                all_infos_window, group_sizes_window = [], []

    final_metrics = summarize(all_infos_total, group_sizes_total)
    print("\n" + "=" * 60)
    print(f"FINAL over {total_episodes} episodes ({run_name})")
    for k, v in final_metrics.items():
        print(f"  {k:35s} = {v:.4f}")
    print("=" * 60)

    if use_wandb:
        wandb.summary.update({f"final_{k}": v for k, v in final_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-game eval: run a checkpoint (or base model) through complete episodes on the held-out benchmark set")
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint dir; omit for zero-shot base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--label", type=str, required=True, help="e.g. echo_step200, episode_return_step200, base")
    parser.add_argument("--director_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--director_mode", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--director_gpu", type=int, default=1)
    parser.add_argument("--director_max_new_tokens", type=int, default=448)
    parser.add_argument("--director_quantize", type=str, default=None, choices=[None, "4bit", "8bit"])
    parser.add_argument("--oracle_n", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--n_structures", type=int, default=20, help="the benchmark set has 20 structures total")
    parser.add_argument("--episodes_per_structure", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=3584)
    parser.add_argument("--max_completion_length", type=int, default=220)
    parser.add_argument("--log_every_episodes", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        label=args.label,
        director_model_name=args.director_model,
        director_mode=args.director_mode,
        director_gpu=args.director_gpu,
        director_max_new_tokens=args.director_max_new_tokens,
        director_quantize=args.director_quantize,
        oracle_n=args.oracle_n,
        max_turns=args.max_turns,
        n_structures=args.n_structures,
        episodes_per_structure=args.episodes_per_structure,
        seed=args.seed,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        log_every_episodes=args.log_every_episodes,
        report_to=args.report_to,
        run_name=args.run_name,
    )
