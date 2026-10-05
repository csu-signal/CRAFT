"""
Loads eval_results/<run_name>.json files
into one tidy frame: a row per episode, a column per metric.
"""
import json
from pathlib import Path

import pandas as pd

from .registry import parse_label, short_model_name

# settings that should match across conditions (warn if not)
COMPARABILITY_KEYS = ["director_mode", "director_model", "oracle_n", "max_turns", "n_structures",
                      "episodes_per_structure", "part_type"]

MUST_MATCH_KEYS = ["structures_sha", "oracle_in_prompt"]


def load_results(results_dir="eval_results", labels=None):
    rows, configs = [], {}
    for path in sorted(Path(results_dir).glob("*.json")):
        run = json.loads(path.read_text())
        if labels and run["label"] not in labels:
            continue
        configs[run["run_name"]] = (run["label"], run.get("config", {}))
        for i, ep in enumerate(run["per_episode"]):
            rows.append({
                "run_name": run["run_name"], "label": run["label"],
                "checkpoint": run.get("checkpoint"), "episode": i, **ep,
            })
    if not rows:
        raise SystemExit(f"no eval results found in {results_dir}/ (expected *.json from eval_full_game.py)")
    df = pd.DataFrame(rows)
    _warn_incomparable(configs)
    labels_seen = {}
    for label, cfg in configs.values():
        model = cfg.get("builder_model") or cfg.get("base_model")
        labels_seen.setdefault(label, set()).add((model, cfg.get("quantize")))
    clash = {l: m for l, m in labels_seen.items() if len(m) > 1}
    if clash:
        raise SystemExit(f"[analysis] the same label was used for different models: {clash} -- labels must be "
                         f"unique per condition (e.g. base_7b, base_14b)")
    models = {l: short_model_name(*next(iter(m))) for l, m in labels_seen.items()}
    conditions = sorted({parse_label(l, models.get(l)) for l in df["label"].unique()}, key=lambda c: c.sort_key)
    _warn_duplicate_names(conditions)
    return df, conditions


def _warn_duplicate_names(conditions):
    seen = {}
    for c in conditions:
        seen.setdefault(c.display, []).append(c.label)
    for name, labels in seen.items():
        if len(labels) > 1:
            print(f"[analysis] warning: {labels} would all be shown as {name!r} -- pass --labels to pick one "
                  f"(e.g. only the final checkpoint of each method)")


def _warn_incomparable(configs):
    for key in MUST_MATCH_KEYS:
        values = {str(cfg.get(key)) for _, cfg in configs.values()}
        if len(values) > 1:
            raise SystemExit(f"[analysis] runs differ in {key!r} ({sorted(values)}) -- analyze each setting "
                             f"separately (point --results at one directory per setting)")
    for key in COMPARABILITY_KEYS:
        values = {}
        for run_name, (label, cfg) in configs.items():
            values.setdefault(str(cfg.get(key)), []).append(label)
        if len(values) > 1:
            detail = "; ".join(f"{v}: {', '.join(sorted(set(ls)))}" for v, ls in values.items())
            print(f"[analysis] warning: runs differ in {key!r} ({detail}) -- comparisons may not be fair")


def structure_units(df, label, metric):
    """Per-structure mean of `metric` for one condition; the unit for paired tests."""
    sub = df[df["label"] == label]
    return sub.groupby("structure_idx")[metric].mean().dropna()


def pass_units(df, label, metric):
    """Per-pass mean of `metric`: pass k is episode k of every structure."""
    sub = df[df["label"] == label]
    return sub.groupby("rep")[metric].mean().dropna()


SEM_UNITS = {"passes": pass_units, "structures": structure_units}
