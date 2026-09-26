"""
Baseline sanity check: does the echo_experiments harness itself let ANY
builder make progress against the frozen local directors, or does even a
strong API builder stall the same way the untrained Qwen2.5-1.5B policy
does?

Runs the exact same rollout.run_builder_episode used by training --
same 3 shared local directors, same create_builder_prompt, same
_extract_command_line/parse_builder_response parsing, same
compute_builder_reward -- swapping only the builder's generate_fn from the
trainable local policy to an OpenAI API call (gpt-4o-mini by default). No
gradients, no LoRA, no TRL/GRPOTrainer -- pure rollout + reward
aggregation, logged to the same wandb project/metric names as train.py so
the two conditions plot side by side.

If the API builder ALSO shows near-zero progress_delta / high clarify_rate
under this harness, that points at a bug in echo_experiments' shared code
(rollout.py, reward.py, or the game engine) rather than Qwen's capability.
If it looks like the vanilla run_craft.py baseline (steady progress, near-
zero clarify), the harness is fine and Qwen is just weak/undertrained.

Usage:
    python baseline_sanity_check.py --builder_model gpt-4o-mini --report_to wandb
"""
import argparse
import datetime
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI

from data_split import load_training_pool, load_benchmark_structures
from local_model_utils import load_local_director_pipeline
from rollout import run_builder_episode
from agents.builder_agent import BUILDER_SYSTEM_PROMPT_ORACLE, BUILDER_SYSTEM_PROMPT_BASE

load_dotenv()

# Mirrors train.py's LOCAL_DIRECTOR_MODELS -- kept as a separate copy rather
# than importing train.py, since train.py's module-level _pre_parse_gpus()
# reads sys.argv at import time and sets CUDA_VISIBLE_DEVICES as a side
# effect. Not needed here: this script never touches transformers.Trainer,
# so the nn.DataParallel auto-wrap issue that forced that dance in train.py
# doesn't apply -- there's no trainable model, just the director pipeline
# on one explicit GPU and an OpenAI API call for the builder.
LOCAL_DIRECTOR_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

BUILDER_LOG_KEYS = [
    "progress_delta", "matched_oracle", "parse_failure", "move_invalid", "completed",
    "off_list_penalty", "invalid_move_penalty", "clarify_penalty",
    "completion_bonus", "efficiency_bonus", "action", "training_reward",
]


def _safe_mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_rate(bools):
    vals = [1.0 if bool(v) else 0.0 for v in bools if v is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(reward_infos, group_sizes):
    """Mirrors CRAFTEchoTrainer._compute_and_log_builder_metrics's formulas
    exactly, so the two conditions are comparable metric-for-metric."""
    L = {k: [info[k] for info in reward_infos if k in info] for k in BUILDER_LOG_KEYS}
    rewards = [info.get("training_reward", 0.0) for info in reward_infos]
    return {
        "builder/progress_delta_mean": _safe_mean(L["progress_delta"]),
        "builder/completed_rate": _safe_rate(L["completed"]),
        "builder/oracle_match_rate": _safe_rate([v for v in L["matched_oracle"] if v is not None]),
        "builder/clarify_rate": _safe_rate([a == "clarify" for a in L["action"]]),
        "builder/parse_failure_rate": _safe_rate(L["parse_failure"]),
        "builder/invalid_move_rate": _safe_rate(L["move_invalid"]),
        "builder/mean_episode_length": float(sum(group_sizes) / len(group_sizes)) if group_sizes else float("nan"),
        "reward": _safe_mean(rewards),
        "reward_std": (torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0),
    }


def make_api_generate_fn(client, model_name, temperature, max_tokens):
    """Same (input_ids, new_tokens, decoded_text) shape run_builder_episode
    expects from the trainable policy's generate_fn -- input_ids/new_tokens
    are never used for anything here (no gradients computed on this data),
    so dummy placeholder tensors are fine.

    System prompt selection and default temperature/max_tokens deliberately
    mirror agents.builder_agent.BuilderAgent.generate_move exactly -- the
    method run_craft.py's own API builder path actually calls -- rather than
    inventing a different convention for this script. An earlier version of
    this file used trainer.py's format-only BUILDER_SYSTEM_PROMPT at
    temperature=0.7, which is NOT what run_craft.py does and made the two
    conditions not actually comparable."""
    dummy = torch.zeros(1, dtype=torch.long)

    def generate_fn(prompt_text, oracle_moves):
        system_prompt = BUILDER_SYSTEM_PROMPT_ORACLE if oracle_moves else BUILDER_SYSTEM_PROMPT_BASE
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        decoded_text = completion.choices[0].message.content or ""
        return dummy, dummy, decoded_text

    return generate_fn


def run(
    builder_model="gpt-4o-mini",
    director_model_name="mistral-7b",
    director_mode="local",
    director_gpu=1,
    director_max_new_tokens=448,
    director_quantize=None,
    oracle_n=20,
    max_turns=20,
    n_structures=30,
    episodes_per_structure=1,
    seed=42,
    temperature=0.1,  # matches BuilderAgent.generate_move exactly
    builder_max_tokens=250,  # matches BuilderAgent.generate_move exactly
    dataset="train_pool",  # "train_pool" (original diagnostic use) or "benchmark" (held-out set, comparable to eval_full_game.py)
    train_pool_path=None,
    log_every_episodes=5,
    report_to="wandb",
    run_name=None,
):
    print(f"[baseline] builder_model={builder_model!r} (temperature={temperature}, max_tokens={builder_max_tokens})")
    print(f"[baseline] director_mode={director_mode!r} director_model={director_model_name!r}")
    print(f"[baseline] dataset={dataset!r}")

    if dataset == "benchmark":
        # The same held-out 20-structure set eval_full_game.py evaluates
        # trained checkpoints on -- use this to get a directly comparable
        # API-builder number, rather than the training pool this script
        # originally used for the harness-vs-Qwen diagnostic.
        structures = load_benchmark_structures()
    else:
        structures = load_training_pool(train_pool_path) if train_pool_path else load_training_pool()

    import random
    rng = random.Random(seed)
    structure_indices = rng.sample(range(len(structures)), min(n_structures, len(structures)))

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

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generate_fn = make_api_generate_fn(client, builder_model, temperature, builder_max_tokens)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = run_name or f"baseline_{builder_model}_{director_model_name}_seed{seed}_{now}"

    use_wandb = report_to == "wandb"
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "craft_echo"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            group="baseline_sanity_check",
            config={
                "builder_model": builder_model, "director_model": director_model_name,
                "oracle_n": oracle_n, "max_turns": max_turns, "temperature": temperature,
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
            print(f"[baseline] episode {episode_num}/{total_episodes} (structure_idx={structure_idx})")
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
    parser = argparse.ArgumentParser(description="Baseline sanity check: API builder vs local directors, same harness as training")
    parser.add_argument("--builder_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--director_model", type=str, default="mistral-7b",
                        help="local: key from LOCAL_DIRECTOR_MODELS or a full HF path; api: an OpenAI model name (e.g. gpt-4.1-mini)")
    parser.add_argument("--director_mode", type=str, default="local", choices=["local", "api"],
                        help="local reuses one shared open-weight model (--director_gpu) across all three "
                             "directors, same as training's default; api calls OpenAI for each director "
                             "instead, same convention as build_frozen_directors(director_mode='api')")
    parser.add_argument("--director_gpu", type=int, default=1)
    parser.add_argument("--director_max_new_tokens", type=int, default=448)
    parser.add_argument("--director_quantize", type=str, default=None, choices=[None, "4bit", "8bit"])
    parser.add_argument("--oracle_n", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--n_structures", type=int, default=30, help="distinct structures sampled from the training pool")
    parser.add_argument("--episodes_per_structure", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="matches BuilderAgent.generate_move exactly -- run_craft.py's real API builder "
                             "path, not an independently chosen value")
    parser.add_argument("--builder_max_tokens", type=int, default=250,
                        help="matches BuilderAgent.generate_move exactly. An earlier version of this script "
                             "used max_tokens=220 with no system prompt guiding format, which caught "
                             "gpt-4o-mini getting cut off mid-reasoning before its PLACE:/REMOVE:/CLARIFY: "
                             "line -- generate_move's oracle system prompt pushes the model to answer "
                             "directly instead of reasoning at length first, so 250 matches production "
                             "rather than needing the larger margin that fix required")
    parser.add_argument("--train_pool", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="train_pool", choices=["train_pool", "benchmark"],
                        help="'train_pool' is the original harness-vs-Qwen diagnostic set; 'benchmark' is "
                             "the same held-out 20-structure set eval_full_game.py evaluates checkpoints "
                             "on, for a directly comparable API-builder baseline number")
    parser.add_argument("--log_every_episodes", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run(
        builder_model=args.builder_model,
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
        builder_max_tokens=args.builder_max_tokens,
        train_pool_path=args.train_pool,
        dataset=args.dataset,
        log_every_episodes=args.log_every_episodes,
        report_to=args.report_to,
        run_name=args.run_name,
    )
