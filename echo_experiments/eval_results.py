"""
Per-episode metrics, aggregation, and result files for full-game evals.

Episodes are appended to <out_dir>/<run_name>.partial.jsonl as they finish; use --resume to continue.
"""
import csv
import json
import math
from pathlib import Path


EPISODE_METRICS = [
    "final_progress", "progress_gain", "completed",
    "oracle_match_rate", "correct_move_rate",
    "clarify_rate", "parse_failure_rate", "invalid_move_rate",
    "episode_length", "reward_mean",
    "director_failure_rate", "completion_tokens_mean",
]


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(vals) / len(vals) if vals else float("nan")


def _rate(bools):
    vals = [1.0 if v else 0.0 for v in bools if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def episode_metrics(episode):
    """Per-episode metrics from a run_builder_episode() result.

    oracle_match_rate: strict match against the candidates shown (NaN if none shown).
    correct_move_rate: strict match against every valid target-advancing move."""
    infos = episode["reward_infos"]
    n_turns = len(infos)
    return {
        "part_type": episode.get("part_type"),
        "initial_progress": episode.get("initial_progress", float("nan")),
        "final_progress": episode["final_progress"],
        "progress_gain": episode["final_progress"] - episode.get("initial_progress", float("nan")),
        "completed": float(any(info.get("completed") for info in infos)),
        "oracle_match_rate": _rate([info.get("matched_oracle_strict") for info in infos]),
        "oracle_match_lenient_rate": _rate([info.get("matched_oracle") for info in infos]),
        "correct_move_rate": _rate([info.get("correct_move") for info in infos]),
        "clarify_rate": _rate([info.get("action") == "clarify" for info in infos if "action" in info]),
        "parse_failure_rate": _rate([info.get("parse_failure") for info in infos]),
        "invalid_move_rate": _rate([info.get("move_invalid") for info in infos]),
        "episode_length": float(n_turns),
        "reward_mean": _mean([info.get("training_reward", 0.0) for info in infos]),
        "director_failure_rate": (sum(info.get("director_failures", 0) for info in infos) / (3 * n_turns)
                                  if n_turns else float("nan")),
        "director_retries": int(sum(info.get("director_retries", 0) for info in infos)),
        "completion_tokens_mean": _mean([info.get("completion_tokens") for info in infos]),
        "progress_curve": [episode.get("initial_progress")] + [info.get("progress") for info in infos],
    }


def aggregate_episodes(per_episode):
    """mean / std (ddof=1) / sem across episodes for each metric, skipping NaNs."""
    out = {}
    for k in EPISODE_METRICS:
        vals = [ep[k] for ep in per_episode
                if isinstance(ep.get(k), (int, float)) and not math.isnan(ep[k])]
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        out[k] = {"mean": mean, "std": std, "sem": std / n ** 0.5 if n > 1 else float("nan"), "n": n}
    return out


class EpisodeLog:
    """Append-only per-episode log, used for crash recovery / --resume."""

    def __init__(self, out_dir, run_name, resume=False):
        self.path = Path(out_dir) / f"{run_name}.partial.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes = []
        if self.path.exists():
            if not resume:
                raise SystemExit(f"{self.path} already exists -- pass --resume to continue that run, "
                                 f"or a different --run_name")
            self.episodes = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
            print(f"[resume] {len(self.episodes)} episodes already done in {self.path}")

    def done(self, structure_idx, rep):
        return any(e["structure_idx"] == structure_idx and e["rep"] == rep for e in self.episodes)

    def append(self, record):
        self.episodes.append(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


RESULT_FIELDS = ["label", "checkpoint", "run_name", "n_episodes"] + [
    f"{k}_{stat}" for k in EPISODE_METRICS for stat in ("mean", "std")
]


def save_eval_results(out_dir, out_csv, label, checkpoint, run_name, config, per_episode):
    """Write <out_dir>/<run_name>.json and append a mean/std row to out_csv."""
    agg = aggregate_episodes(per_episode)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "label": label, "checkpoint": checkpoint, "run_name": run_name,
            "config": config, "n_episodes": len(per_episode),
            "metrics": agg, "per_episode": per_episode,
        }, f, indent=2)

    out_csv = Path(out_csv)
    if out_csv.exists():
        with open(out_csv) as f:
            header = f.readline().strip().split(",")
        if header != RESULT_FIELDS:
            legacy = out_csv.with_suffix(".legacy.csv")
            out_csv.rename(legacy)
            print(f"[results] {out_csv} had an older column layout; moved it to {legacy}")
    row = {"label": label, "checkpoint": checkpoint or "", "run_name": run_name, "n_episodes": len(per_episode)}
    for k, v in agg.items():
        row[f"{k}_mean"], row[f"{k}_std"] = v["mean"], v["std"]
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"wrote {json_path} and appended row to {out_csv}")
    return agg
