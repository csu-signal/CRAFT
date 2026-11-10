"""
Estimate API tokens for a full-game eval by answering every API call with a local Qwen2.5-14B.

--builder local wraps eval_full_game.py (only the directors call an API); --builder api wraps
baseline_sanity_check.py (the builder and the directors do). All other flags pass through to that script.
Run a small sample (--n_structures / --episodes_per_structure); the report extrapolates to the full
protocol, per role. wandb is always off.

    python eval_full_game_est.py --builder local --gpus 0 --est_gpu 1 --label echo --no_oracle \
        --checkpoint /data/craft_echo_runs/<run>/checkpoint-350 --n_structures 2 --out_dir eval_results_est
    python eval_full_game_est.py --builder api --est_gpu 1 --label api_claude-sonnet-4-6 --no_oracle \
        --builder_model claude-sonnet-4-6 --n_structures 2 --out_dir eval_results_est
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--builder", choices=["local", "api"], default="local")
_pre.add_argument("--gpus", type=str, default="0")
_pre.add_argument("--est_gpu", type=int, default=1)
_pre.add_argument("--full_structures", type=int, default=20)
_pre.add_argument("--full_episodes", type=int, default=3)
_pre.add_argument("--director_correction", type=float, default=1.0,
                  help="API tokens per Qwen token for the directors; ~1.0 for gpt-4.1-mini")
_pre.add_argument("--builder_correction", type=float, default=1.5,
                  help="API tokens per Qwen token for an API builder; ~1.5 for recent Claude tokenizers")
EST_ARGS, _rest = _pre.parse_known_args()
if EST_ARGS.builder == "local":
    # builder GPU(s) first so eval_full_game's cuda:0 is the builder; the estimator is the last visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{EST_ARGS.gpus},{EST_ARGS.est_gpu}"
    _rest += ["--gpus", EST_ARGS.gpus]
    EST_DEVICE = f"cuda:{len(EST_ARGS.gpus.split(','))}"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(EST_ARGS.est_gpu)
    EST_DEVICE = "cuda:0"
sys.argv = [sys.argv[0]] + _rest + ["--report_to", "none"]

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from huggingface_hub import snapshot_download

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


class _Tagged:
    """Records which role made each estimator call."""

    def __init__(self, est, role):
        self.est, self.role = est, role

    def _chat(self, messages, max_tokens):
        text, call = self.est._chat(messages, max_tokens)
        call["role"] = self.role
        return text, call


token_estimator.MODEL = _local_snapshot(token_estimator.MODEL)
EST = TokenEstimator(max_tokens=500, device=EST_DEVICE)
EPISODES = []

if EST_ARGS.builder == "local":
    import eval_full_game as target
    _load_builder = target.load_builder
    target.load_builder = lambda base_model, *a, **kw: _load_builder(_local_snapshot(base_model), *a, **kw)
else:
    import anthropic
    import openai
    import baseline_sanity_check as target
    # make_api_generate_fn builds its client from these at call time; the directors bound OpenAI at import
    # and get their own clients replaced below
    anthropic.Anthropic = lambda *a, **kw: EstimatorClient(_Tagged(EST, "builder"))
    openai.OpenAI = lambda *a, **kw: EstimatorClient(_Tagged(EST, "builder"))

_build_directors = rollout.build_frozen_directors


def _build_est_directors(*args, **kwargs):
    agents = _build_directors(*args, **kwargs)
    for agent in agents.values():
        agent.client = EstimatorClient(_Tagged(EST, "director"))
    return agents


_run_episode = eval_harness.run_builder_episode


def _role_totals(calls, role):
    calls = [c for c in calls if c.get("role") == role]
    return {"calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "truncated_calls": sum(c["truncated"] for c in calls),
            "max_input_tokens": max((c["input_tokens"] for c in calls), default=0)}


def _run_est_episode(*args, **kwargs):
    start, t0 = len(EST.calls), time.time()
    episode = _run_episode(*args, **kwargs)
    calls = EST.calls[start:]
    EPISODES.append({"structure_idx": kwargs["structure_index"], "seed": kwargs["seed"],
                     "turns": len(episode["reward_infos"]), "seconds": round(time.time() - t0),
                     **{role: _role_totals(calls, role) for role in ("director", "builder")}})
    print(f"[est] episode {EPISODES[-1]}")
    return episode


rollout.build_frozen_directors = _build_est_directors
eval_harness.run_builder_episode = _run_est_episode


def _role_report(role, correction, max_turns, full_eps):
    per_ep = [e[role] for e in EPISODES]
    n, turns = len(EPISODES), sum(e["turns"] for e in EPISODES)
    tin = sum(r["input_tokens"] for r in per_ep) * correction
    tout = sum(r["output_tokens"] for r in per_ep) * correction
    calls = sum(r["calls"] for r in per_ep)
    if not calls:
        return None
    # per-turn cost grows with conversation history, so the max-turns bound uses the longest sampled episode's rate
    longest = max(EPISODES, key=lambda e: e["turns"])
    lt = longest["turns"] or 1
    return {
        "correction": correction,
        "sample": {"calls": calls, "truncated_calls": sum(r["truncated_calls"] for r in per_ep),
                   "input_tokens": round(tin), "output_tokens": round(tout),
                   "input_per_call": round(tin / calls), "output_per_call": round(tout / calls),
                   "max_prompt_tokens": round(max(r["max_input_tokens"] for r in per_ep) * correction)},
        "per_episode_mean": {"turns": turns / n, "input_tokens": round(tin / n), "output_tokens": round(tout / n)},
        "full_run": {
            "episodes": full_eps,
            "at_sample_mean": {"input_tokens": round(tin / n * full_eps), "output_tokens": round(tout / n * full_eps)},
            "all_episodes_max_turns": {
                "input_tokens": round(longest[role]["input_tokens"] * correction / lt * max_turns * full_eps),
                "output_tokens": round(longest[role]["output_tokens"] * correction / lt * max_turns * full_eps),
            },
        },
    }


def summarize(label, max_turns, out_dir):
    full_eps = EST_ARGS.full_structures * EST_ARGS.full_episodes
    report = {"label": label, "builder": EST_ARGS.builder, "sample_episodes": len(EPISODES),
              "sample_turns": sum(e["turns"] for e in EPISODES),
              "director": _role_report("director", EST_ARGS.director_correction, max_turns, full_eps),
              "builder_calls": _role_report("builder", EST_ARGS.builder_correction, max_turns, full_eps),
              "episodes": EPISODES}
    print("\n" + "=" * 60 + f"\nTOKEN ESTIMATE ({label})\n"
          + json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
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
        target.main()
    finally:
        if EPISODES:
            summarize(known.label, known.max_turns, known.out_dir)
