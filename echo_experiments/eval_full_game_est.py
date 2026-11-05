"""
Estimate director API tokens for an eval_full_game.py run by answering the directors with a local Qwen2.5-14B.

Takes the same flags as eval_full_game.py; run a small sample (--n_structures / --episodes_per_structure)
and the report extrapolates to the full protocol. wandb is always off.

    python eval_full_game_est.py --gpus 0 --est_gpu 1 --label echo --no_oracle \
        --checkpoint /data/craft_echo_runs/<run>/checkpoint-350 \
        --n_structures 2 --episodes_per_structure 1 --out_dir eval_results_est
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpus", type=str, default="0")
_pre.add_argument("--est_gpu", type=int, default=1)
_pre.add_argument("--full_structures", type=int, default=20)
_pre.add_argument("--full_episodes", type=int, default=3)
_pre.add_argument("--correction", type=float, default=1.0,
                  help="API tokens per Qwen token; ~1.0 for gpt-4.1-mini, 1.5 for Claude")
EST_ARGS, _rest = _pre.parse_known_args()
# builder GPU(s) first so eval_full_game's cuda:0 is the builder; the estimator is the last visible device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{EST_ARGS.gpus},{EST_ARGS.est_gpu}"
sys.argv = [sys.argv[0]] + _rest + ["--gpus", EST_ARGS.gpus, "--report_to", "none"]

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from huggingface_hub import snapshot_download

import eval_full_game
import eval_harness
import rollout
import token_estimator
from token_estimator import EstimatorClient, TokenEstimator


def _local_snapshot(repo):
    """Cached snapshot dir for a hub repo id; transformers skips its hub lookups when given a local path."""
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return repo


token_estimator.MODEL = _local_snapshot(token_estimator.MODEL)
_load_builder = eval_full_game.load_builder
eval_full_game.load_builder = lambda base_model, *a, **kw: _load_builder(_local_snapshot(base_model), *a, **kw)

EST = TokenEstimator(max_tokens=500, device=f"cuda:{len(EST_ARGS.gpus.split(','))}")
EPISODES = []

_build_directors = rollout.build_frozen_directors


def _build_est_directors(*args, **kwargs):
    agents = _build_directors(*args, **kwargs)
    for agent in agents.values():
        agent.client = EstimatorClient(EST)
    return agents


_run_episode = eval_harness.run_builder_episode


def _run_est_episode(*args, **kwargs):
    start, t0 = len(EST.calls), time.time()
    episode = _run_episode(*args, **kwargs)
    calls = EST.calls[start:]
    EPISODES.append({
        "structure_idx": kwargs["structure_index"], "seed": kwargs["seed"],
        "turns": len(calls) // 3, "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "truncated_calls": sum(c["truncated"] for c in calls),
        "max_input_tokens": max((c["input_tokens"] for c in calls), default=0),
        "seconds": round(time.time() - t0),
    })
    print(f"[est] episode {EPISODES[-1]}")
    return episode


rollout.build_frozen_directors = _build_est_directors
eval_harness.run_builder_episode = _run_est_episode


def summarize(label, max_turns, out_dir):
    c = EST_ARGS.correction
    n = len(EPISODES)
    turns = sum(e["turns"] for e in EPISODES)
    tin = sum(e["input_tokens"] for e in EPISODES) * c
    tout = sum(e["output_tokens"] for e in EPISODES) * c
    full_eps = EST_ARGS.full_structures * EST_ARGS.full_episodes
    # per-turn cost grows with conversation history, so the max-turns bound uses the longest sampled episode's rate
    longest = max(EPISODES, key=lambda e: e["turns"])
    report = {
        "label": label, "correction": c, "sample_episodes": n, "sample_turns": turns,
        "sample": {"calls": len(EST.calls), "truncated_calls": sum(e["truncated_calls"] for e in EPISODES),
                   "input_tokens": round(tin), "output_tokens": round(tout),
                   "max_prompt_tokens": round(max(e["max_input_tokens"] for e in EPISODES) * c)},
        "per_episode_mean": {"turns": turns / n, "input_tokens": round(tin / n), "output_tokens": round(tout / n)},
        "full_run": {
            "episodes": full_eps,
            "at_sample_mean": {"input_tokens": round(tin / n * full_eps), "output_tokens": round(tout / n * full_eps)},
            "all_episodes_max_turns": {
                "input_tokens": round(longest["input_tokens"] * c / longest["turns"] * max_turns * full_eps),
                "output_tokens": round(longest["output_tokens"] * c / longest["turns"] * max_turns * full_eps),
            },
        },
        "episodes": EPISODES,
    }
    print("\n" + "=" * 60 + f"\nTOKEN ESTIMATE ({label})\n" + json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
    path = Path(out_dir) / f"token_estimate_{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"[est] wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--label", required=True)
    p.add_argument("--max_turns", type=int, default=20)
    p.add_argument("--out_dir", default="eval_results")
    known, _ = p.parse_known_args()
    try:
        eval_full_game.main()
    finally:
        if EPISODES:
            summarize(known.label, known.max_turns, known.out_dir)
