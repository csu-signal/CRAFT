"""
progress_plotting.py
--------------------
Produces two figures:
  Figure 1 — Raw metrics anchored at t=0 (synthetic zero baseline)
  Figure 2 — Delta metrics: value[t] - value[turn_1], showing agent-contributed gain

Both figures split by partialCompletionCategory with SEM bands.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict
#%matplotlib inline
SIM_DIR = Path("craft_gricean_simulations").parent
RUN_FILTER = 1
N_TURNS    = 20

METRIC_KEYS = [
    "overall_progress",
    "completion_percentage",
    "iou_score",
    "position_accuracy",
    "distance_score",
]
BINARY_KEYS = [
    "move_executed",
    "failed_move",
    "correct_structure_placement",
    "correct_side_placement",
]
ALL_KEYS   = METRIC_KEYS + BINARY_KEYS
DELTA_KEYS = [f"{k}_delta" for k in METRIC_KEYS]

# ── Load files ────────────────────────────────────────────────────────────────

files  = sorted(SIM_DIR.glob("*.json"))
groups = defaultdict(dict)
for f in files:
    m = re.match(r"craft_structure_(\d+)_(\d+)\.json", f.name)
    if m:
        struct_id, run = m.group(1), int(m.group(2))
        groups[struct_id][run] = f

run1_files = {sid: groups[sid][RUN_FILTER]
              for sid in groups if RUN_FILTER in groups[sid]}
print(f"Loaded {len(run1_files)} structures (run {RUN_FILTER} only)")

condition_labels = set()
for sid, fpath in run1_files.items():
    with open(fpath) as f:
        d = json.load(f)
    cat = d["games"][0].get("partialCompletionCategory", "unknown")
    condition_labels.add(cat)

condition_labels = sorted(condition_labels)
print(f"Conditions found: {condition_labels}")

# ── Extract metric from a turn ────────────────────────────────────────────────

def get_metric_at_turn(turn, key):
    if key in METRIC_KEYS:
        pd = turn.get("progress_data", {})
        if isinstance(pd, dict) and "metrics" in pd:
            return pd["metrics"].get(key, None)
    elif key in BINARY_KEYS:
        v = turn.get(key, None)
        return float(v) if v is not None else None
    return None

# ── Build time series ─────────────────────────────────────────────────────────
# Index layout: 0 = synthetic t=0, 1..N_TURNS = actual turns

TOTAL_LEN = N_TURNS + 1

data = {cond: {k: [] for k in ALL_KEYS + DELTA_KEYS} for cond in condition_labels}
data["ALL"] = {k: [] for k in ALL_KEYS + DELTA_KEYS}

for sid, fpath in run1_files.items():
    with open(fpath) as f:
        d = json.load(f)
    game  = d["games"][0]
    turns = game["turns"]
    cond  = game.get("partialCompletionCategory", "unknown")

    # Initialise with synthetic t=0 = 0.0 for continuous, None for binary
    struct_vals = {}
    for k in METRIC_KEYS:
        struct_vals[k] = [0.0] + [None] * N_TURNS
    for k in BINARY_KEYS:
        struct_vals[k] = [None] + [None] * N_TURNS

    for t in turns:
        tn = t.get("turn_number", None)
        if tn is None or tn < 1 or tn > N_TURNS:
            continue
        for k in ALL_KEYS:
            v = get_metric_at_turn(t, k)
            if v is not None:
                struct_vals[k][tn] = v

    # Delta: value[t] - value[turn_1], anchored so turn 1 = 0

    

    for k in METRIC_KEYS:
        dk = f"{k}_delta"
        baseline = struct_vals[k][1]
         
        if baseline is not None:
            # headroom = 1.0 - baseline
            struct_vals[dk] = [None] + [
                (struct_vals[k][t] - baseline) if struct_vals[k][t] is not None else None
                for t in range(1, TOTAL_LEN)
            ]

            # struct_vals[dk] = [None] + [
            #     ((struct_vals[k][t] - baseline) / headroom) if (struct_vals[k][t] is not None and headroom > 1e-9) else None
            #     for t in range(1, TOTAL_LEN)
            # ]
            struct_vals[dk][1] = 0.0   # explicitly anchor turn 1 = 0
        else:
            struct_vals[dk] = [None] * TOTAL_LEN

    for k in ALL_KEYS + DELTA_KEYS:
        data[cond][k].append(struct_vals[k])
        data["ALL"][k].append(struct_vals[k])

# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(series_list, total_len=TOTAL_LEN):
    means = np.full(total_len, np.nan)
    sems  = np.full(total_len, np.nan)
    for t in range(total_len):
        vals = [s[t] for s in series_list if s[t] is not None]
        if len(vals) >= 2:
            arr      = np.array(vals, dtype=float)
            means[t] = np.mean(arr)
            sems[t]  = np.std(arr, ddof=1) / np.sqrt(len(arr))
        elif len(vals) == 1:
            means[t] = vals[0]; sems[t] = 0.0
    return means, sems

# ── Print tables ──────────────────────────────────────────────────────────────

turns_axis = np.arange(0, N_TURNS + 1)

print(f"\n{'='*80}\nRAW METRICS — turn-by-turn means (ALL structures)\n{'='*80}")
for k in ALL_KEYS:
    means, sems = compute_stats(data["ALL"][k])
    print(f"\n  {k}")
    print(f"  {'turn':>5}  {'mean':>8}  {'sem':>8}  {'n':>5}")
    print(f"  {'-'*35}")
    for t in range(TOTAL_LEN):
        n_obs = sum(1 for s in data["ALL"][k] if s[t] is not None)
        m, se = means[t], sems[t]
        ms = f"{m:.4f}" if not np.isnan(m) else "   n/a"
        ss = f"{se:.4f}" if not np.isnan(se) else "   n/a"
        print(f"  {t:>5}  {ms:>8}  {ss:>8}  {n_obs:>5}")

print(f"\n{'='*80}\nDELTA METRICS — gain from turn 1 baseline\n{'='*80}")
for dk in DELTA_KEYS:
    means, sems = compute_stats(data["ALL"][dk])
    print(f"\n  {dk}")
    print(f"  {'turn':>5}  {'mean':>8}  {'sem':>8}  {'n':>5}")
    print(f"  {'-'*35}")
    for t in range(1, TOTAL_LEN):
        n_obs = sum(1 for s in data["ALL"][dk] if s[t] is not None)
        m, se = means[t], sems[t]
        ms = f"{m:.4f}" if not np.isnan(m) else "   n/a"
        ss = f"{se:.4f}" if not np.isnan(se) else "   n/a"
        print(f"  {t:>5}  {ms:>8}  {ss:>8}  {n_obs:>5}")

# ── Colors ────────────────────────────────────────────────────────────────────

if len(condition_labels) > 1:
    cmap   = cm.get_cmap("tab10", len(condition_labels))
    colors = {cond: cmap(i) for i, cond in enumerate(condition_labels)}
else:
    colors = {condition_labels[0]: "#2563EB"}

# ── Plot helper ───────────────────────────────────────────────────────────────

def plot_metric_grid(keys, title, fname, x_start=0, hline_val=0.0, ylabel="Value"):
    n_cols = 3
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax_idx, key in enumerate(keys):
        ax = axes_flat[ax_idx]

        for cond in condition_labels:
            n_structs = len(data[cond][key])
            if n_structs == 0:
                continue
            means, sems = compute_stats(data[cond][key])
            color = colors[cond]
            label = f"{cond} (n={n_structs})"

            mask = (turns_axis >= x_start) & ~np.isnan(means)
            x = turns_axis[mask]; y = means[mask]; e = sems[mask]

            ax.plot(x, y, color=color, linewidth=2, label=label,
                    marker='o', markersize=3, zorder=3)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, zorder=2)

        display = key.replace("_delta", "").replace("_", " ").title()
        if key.endswith("_delta"):
            display += " (Δ from turn 1)"
        ax.set_title(display, fontsize=11, fontweight='bold')
        ax.set_xlabel("Turn", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(x_start - 0.3, N_TURNS + 0.3)
        ax.set_xticks(range(x_start, N_TURNS + 1, 2))
        ax.axhline(hline_val, color='gray', linewidth=0.8,
                   linestyle='--', zorder=1, alpha=0.6)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(condition_labels) > 1:
            ax.legend(fontsize=7, framealpha=0.8)

    for ax_idx in range(len(keys), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()       # ← add this line before close
    plt.close(fig)
 

# ── Figure 1: Raw (all metrics + binary), t=0 anchored ───────────────────────

plot_metric_grid(
    keys      = ALL_KEYS,
    title     = (f"CRAFT Raw Metrics — Turn by Turn  "
                 f"(Run {RUN_FILTER}, t=0 anchored, SEM over structures)"),
    fname     = "craft_metrics_raw.png",
    x_start   = 0,
    hline_val = 0.0,
    ylabel    = "Value",
)

# ── Figure 2: Delta metrics (continuous only), turn 1 anchored ───────────────

plot_metric_grid(
    keys      = DELTA_KEYS,
    title     = (f"CRAFT Metric Deltas — Gain from Turn 1 Baseline  "
                 f"(Run {RUN_FILTER}, turn 1 = 0, SEM over structures)"),
    fname     = "craft_metrics_delta.png",
    x_start   = 1,
    hline_val = 0.0,
    ylabel    = "Δ Value (from turn 1)",
)

print("\nDone.")