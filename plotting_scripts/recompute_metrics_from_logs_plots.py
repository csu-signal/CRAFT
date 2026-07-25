"""
plot_craft_metrics_by_model.py
-----------------------------
Plots CRAFT metrics across separate model-combination directories.

Expected directory structure:
ROOT_DIR/
    qwen-72b_gpt-4o-mini,,1773456132000/
        craft_structure_001_2.json
        craft_structure_002_2.json
        ...
    qwen-32b_gpt-4o-mini,,1773401844000/
        craft_structure_001_2.json
        ...
    ...

This script produces:

  Figure 1 — Raw metrics anchored at t=0
  Figure 2 — Delta metrics: value[t] - value[turn_1]

Curves are grouped by MODEL COMBINATION DIRECTORY, not by partialCompletionCategory.
Aggregation is over structures within each model directory, with SEM bands.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

class TaskProgressTracker:
    """
    Tracks progress toward target structure using distance-based metrics
    """
    
    def __init__(self, target_structure):
        self.target_structure = target_structure
        self.progress_history = []
        self.move_history = []
    
    def calculate_progress(self, current_structure):
        """
        Calculate progress using multiple metrics
        
        Returns:
            dict: Progress metrics including IoU, distance, and completion percentage
        """
        
        # Normalize structures for comparison
        current_norm = self._normalize_structure(current_structure)
        target_norm = self._normalize_structure(self.target_structure)
        
        # Calculate different metrics
        iou_score = self._calculate_iou(current_norm, target_norm)
        distance_score = self._calculate_distance(current_norm, target_norm)
        completion_percentage = self._calculate_completion_percentage(current_norm, target_norm)
        position_accuracy = self._calculate_position_accuracy(current_norm, target_norm)
        
        progress_data = {
            'iou_score': iou_score,
            'distance_score': distance_score,
            'completion_percentage': completion_percentage,
            'position_accuracy': position_accuracy,
            'overall_progress': (iou_score + completion_percentage + position_accuracy) / 3,
            'blocks_placed_correctly': self._count_correct_blocks(current_norm, target_norm),
            'blocks_total_target': self._count_total_blocks(target_norm),
            'blocks_total_current': self._count_total_blocks(current_norm)
        }
        
        return progress_data
    
    def track_move(self, move_data, current_structure, turn_number):
        """
        Track a move and calculate progress delta
        
        Args:
            move_data: The move that was executed
            current_structure: Structure after the move
            turn_number: Current turn number
            
        Returns:
            dict: Progress metrics and delta from previous turn
        """
        
        current_progress = self.calculate_progress(current_structure)
        print(f"DEBUG: Progress calculation - target blocks: {self._count_total_blocks(self._normalize_structure(self.target_structure))}")
        print(f"DEBUG: Progress calculation - current blocks: {self._count_total_blocks(self._normalize_structure(current_structure))}")
        
        # Calculate delta from previous turn
        progress_delta = 0
        if self.progress_history:
            previous_progress = self.progress_history[-1]['metrics']['overall_progress']
            progress_delta = current_progress['overall_progress'] - previous_progress
        
        # Store progress record
        progress_record = {
            'turn_number': turn_number,
            'move': move_data,
            'metrics': current_progress,
            'progress_delta': progress_delta,
            'structure_snapshot': copy.deepcopy(current_structure)
        }
        
        self.progress_history.append(progress_record)
        self.move_history.append(move_data)
        
        return progress_record

    def _normalize_structure(self, structure):
        """
        Normalize structure format for comparison
        Converts coordinate keys to tuples and handles missing positions
        """
        normalized = {}
        
        for i in range(3):
            for j in range(3):
                # Try both formats: with and without spaces
                coord_key_spaces = f"({i}, {j})"
                coord_key_no_spaces = f"({i},{j})"
                coord_tuple = (i, j)
                
                if coord_key_no_spaces in structure:
                    normalized[coord_tuple] = structure[coord_key_no_spaces]
                elif coord_key_spaces in structure:
                    normalized[coord_tuple] = structure[coord_key_spaces]
                else:
                    normalized[coord_tuple] = []
        
        return normalized
    
    # def _normalize_structure(self, structure):
    #     """
    #     Normalize structure format for comparison
    #     Converts coordinate keys to tuples and handles missing positions
    #     """
    #     normalized = {}
        
    #     for i in range(3):
    #         for j in range(3):
    #             coord_key = f"({i}, {j})"
    #             coord_tuple = (i, j)
                
    #             if coord_key in structure:
    #                 normalized[coord_tuple] = structure[coord_key]
    #             else:
    #                 normalized[coord_tuple] = []
        
    #     return normalized
    
    def _calculate_iou(self, current, target):
        """
        Calculate Intersection over Union (IoU) for block positions
        """
        intersection = 0
        union = 0
        
        for coord in current.keys():
            current_blocks = set(current[coord])
            target_blocks = set(target[coord])
            
            intersection += len(current_blocks.intersection(target_blocks))
            union += len(current_blocks.union(target_blocks))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, current, target):
        """
        Calculate normalized distance between current and target states
        Lower distance = better progress (closer to target)
        """
        total_distance = 0
        total_possible_distance = 0
        
        for coord in current.keys():
            current_blocks = current[coord]
            target_blocks = target[coord]
            
            # Calculate edit distance (insertions + deletions needed)
            current_set = set(current_blocks)
            target_set = set(target_blocks)
            
            # Distance = blocks to remove + blocks to add
            distance = len(current_set - target_set) + len(target_set - current_set)
            total_distance += distance
            
            # Maximum possible distance for this position
            max_distance = len(current_set) + len(target_set)
            total_possible_distance += max_distance
        
        if total_possible_distance == 0:
            return 1.0  # Perfect if both empty
        
        # Return 1 - normalized_distance (so higher = better)
        normalized_distance = total_distance / total_possible_distance
        return 1.0 - normalized_distance
    
    def _calculate_completion_percentage(self, current, target):
        """
        Calculate percentage of target blocks that are correctly placed
        """
        correct_blocks = 0
        total_target_blocks = 0
        
        for coord in target.keys():
            target_blocks = target[coord]
            current_blocks = current[coord]
            
            total_target_blocks += len(target_blocks)
            
            # Count blocks that are in correct position and layer
            for i, target_block in enumerate(target_blocks):
                if i < len(current_blocks) and current_blocks[i] == target_block:
                    correct_blocks += 1
        
        return correct_blocks / total_target_blocks if total_target_blocks > 0 else 0.0
    
    def _calculate_position_accuracy(self, current, target):
        """
        Calculate accuracy based on correct block placement regardless of layer order
        """
        correct_positions = 0
        total_positions = 9  # 3x3 grid
        
        for coord in target.keys():
            target_blocks = set(target[coord])
            current_blocks = set(current[coord])
            
            # Position is correct if it has exactly the right blocks (regardless of order)
            if target_blocks == current_blocks:
                correct_positions += 1
        
        return correct_positions / total_positions
    
    def _count_correct_blocks(self, current, target):
        """Count total number of blocks in correct positions"""
        correct_count = 0
        
        for coord in target.keys():
            target_blocks = target[coord]
            current_blocks = current[coord]
            
            # Count blocks that match position and layer
            for i, target_block in enumerate(target_blocks):
                if i < len(current_blocks) and current_blocks[i] == target_block:
                    correct_count += 1
        
        return correct_count
    
    def _count_total_blocks(self, structure):
        """Count total blocks in structure"""
        total = 0
        for coord in structure.keys():
            total += len(structure[coord])
        return total
    
    def get_progress_summary(self):
        """
        Get summary of progress over time
        """
        if not self.progress_history:
            return {"message": "No progress tracked yet"}
        
        latest = self.progress_history[-1]
        
        summary = {
            'current_turn': latest['turn_number'],
            'overall_progress': latest['metrics']['overall_progress'],
            'completion_percentage': latest['metrics']['completion_percentage'],
            'blocks_correct': latest['metrics']['blocks_placed_correctly'],
            'blocks_total_needed': latest['metrics']['blocks_total_target'],
            'recent_trend': self._calculate_recent_trend(),
            'is_improving': self._is_improving(),
            'estimated_turns_remaining': self._estimate_remaining_turns()
        }
        
        return summary
    
    def _calculate_recent_trend(self, window_size=3):
        """Calculate trend over recent moves"""
        if len(self.progress_history) < 2:
            return 0.0
        
        recent_deltas = [record['progress_delta'] for record in self.progress_history[-window_size:]]
        return sum(recent_deltas) / len(recent_deltas)
    
    def _is_improving(self, window_size=3):
        """Check if progress is generally improving"""
        if len(self.progress_history) < 2:
            return True
        
        recent_trend = self._calculate_recent_trend(window_size)
        return recent_trend > -0.05  # Allow for small fluctuations
    
    def _estimate_remaining_turns(self):
        """Rough estimate of turns needed to complete"""
        if not self.progress_history:
            return float('inf')
        
        current_progress = self.progress_history[-1]['metrics']['overall_progress']
        
        if current_progress >= 0.95:
            return 0
        
        if len(self.progress_history) < 3:
            return float('inf')
        
        # Calculate average progress per turn
        total_progress = current_progress
        turns_taken = len(self.progress_history)
        avg_progress_per_turn = total_progress / turns_taken if turns_taken > 0 else 0
        
        if avg_progress_per_turn <= 0:
            return float('inf')
        
        remaining_progress = 1.0 - current_progress
        estimated_turns = remaining_progress / avg_progress_per_turn
        
        return max(1, int(estimated_turns))

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(
    "/Users/hannahvanderhoeven/Documents/GitHub/LLM_Pragmatic_Analysis/data/craft_data/experiment1"
)
# "craft_gricean_simulations_open_weight_testing_20test_notools"  


#RUN_FILTER = 1
N_TURNS = 20
SAVE_PREFIX = "craft_metrics_by_model"

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

ALL_KEYS = METRIC_KEYS + BINARY_KEYS
DELTA_KEYS = [f"{k}_delta" for k in METRIC_KEYS]
TOTAL_LEN = N_TURNS + 1
TURNS_AXIS = np.arange(0, N_TURNS + 1)

def recompute_metrics_from_logs(game: dict) -> dict:
    """
    Recompute all progress metrics from structure_snapshots in turn logs.
    Returns dict: metric_name -> list of length N_TURNS+1 (index 0 = turn 0 baseline)
    """
    target_structure = game["target_structure"]
    tracker = TaskProgressTracker(target_structure)
    turns = game["turns"]
    n_turns = game["turns_taken"]

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

    # initialize with turn 0 baseline — board state before any moves
    # use structure_before from turn 1 as t=0 state
    first_turn = next((t for t in turns if t.get("turn_number") == 1), None)

    if first_turn is not None:
        t0_structure = first_turn.get("structure_before", {})
    else:
        # fallback: empty board
        t0_structure = {f"({i},{j})": [] for i in range(3) for j in range(3)}

    t0_metrics = tracker.calculate_progress(t0_structure)

    result = {k: [None] * (n_turns + 1) for k in METRIC_KEYS + BINARY_KEYS}
    delta_result = {f"{k}_delta": [None] * (n_turns + 1) for k in METRIC_KEYS}

    # set t=0
    for k in METRIC_KEYS:
        result[k][0] = t0_metrics[k]
    for k in BINARY_KEYS:
        result[k][0] = None  # no move at t=0

    prev_metrics = t0_metrics

    for turn in turns:
        tn = turn.get("turn_number")
        if tn is None or tn < 1 or tn > n_turns:
            continue

        # get board state after this turn
        snapshot = turn.get("progress_data", {}).get("structure_snapshot")

        if snapshot is None:
            # failed move — board unchanged, use structure_before
            snapshot = turn.get("structure_before", t0_structure)

        # recompute metrics from snapshot
        metrics = tracker.calculate_progress(snapshot)

        for k in METRIC_KEYS:
            result[k][tn] = metrics[k]
            delta_result[f"{k}_delta"][tn] = metrics[k] - t0_metrics[k]

        # binary keys — read directly from turn log, these are reliable
        result["move_executed"][tn]              = float(turn.get("move_executed", 0) or 0)
        result["failed_move"][tn]               = float(turn.get("failed_move", 0) or 0)
        result["correct_structure_placement"][tn] = float(turn.get("correct_structure_placement", 0) or 0)
        result["correct_side_placement"][tn]      = float(turn.get("correct_side_placement", 0) or 0)

        prev_metrics = metrics

    # merge
    result.update(delta_result)
    return result

def recompute_all_flat(root_dir: Path):
    """
    Same recomputation as recompute_all but returns flat structure:
        data[model_label][metric] = list of series   (no category dimension)
        data["ALL"][metric] = list of series          (pooled across models)
    """
    from collections import defaultdict
    import re

    def clean_model_label(dirname):
        name = dirname.split(",,")[0]
        parts = name.split("_")
        model_map = {
            'qwen-7b':'Qwen-7B','qwen-14b':'Qwen-14B','qwen-32b':'Qwen-32B','qwen-72b':'Qwen-72B',
            'mistral-7b':'Mistral-7B','llama-8b':'Llama-8B','gemma-9b':'Gemma-9B',
            'deepseek-v2-lite':'DeepSeek-Lite',
            'gpt-4o':'GPT-4o','gpt-4o-mini':'GPT-4o-Mini','gpt-4.1-mini':'GPT-4.1-Mini',
            'claude-sonnet-4-6':'Claude-Sonnet-4.6','gemini-2.5-flash':'Gemini-2.5-Flash',
            'gemini-3-flash-preview':'Gemini-3-Flash',
            'gemini-3.1-flash-lite-preview':'Gemini-3.1-Flash-lite',
        }
        builder_map = {"gpt-4o-mini": "4o-mini"}
        builder_map = {"gpt-5.4-mini": "5.4-mini"}
        model   = model_map.get(parts[0], parts[0])
        builder = builder_map.get(parts[1], parts[1]) if len(parts) > 1 else ""
        return f"{model} + {builder}" if builder else model

    METRIC_KEYS = [
        "overall_progress", "completion_percentage",
        "iou_score", "position_accuracy", "distance_score",
    ]
    BINARY_KEYS = [
        "move_executed", "failed_move",
        "correct_structure_placement", "correct_side_placement",
    ]
    ALL_KEYS   = METRIC_KEYS + BINARY_KEYS
    DELTA_KEYS = [f"{k}_delta" for k in METRIC_KEYS]

    data = defaultdict(lambda: defaultdict(list))
    data["ALL"]  # touch it so it exists
    model_labels = []

    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_label = clean_model_label(model_dir.name)
        found = False

        for fpath in sorted(model_dir.glob("*.json")):
            m = re.match(r"craft_structure_(\d+)_(\d+)\.json", fpath.name)
            # if not m or int(m.group(2)) != run_filter:
            #     continue

            with open(fpath) as f:
                d = json.load(f)

            if "games" not in d or not d["games"]:
                print(f"Warning: skipping {fpath.name} — no games")
                continue

            game     = d["games"][0]
            # recompute from snapshots — no dependency on logged metric values
            struct_vals = recompute_metrics_from_logs(game)

            for k in ALL_KEYS + DELTA_KEYS:
                data[model_label][k].append(struct_vals[k])
                data["ALL"][k].append(struct_vals[k])  # pool across models

            found = True

        if found and model_label not in model_labels:
            model_labels.append(model_label)
            print(f"Recomputed: {model_label:<28} ({len(data[model_label]['overall_progress'])} structures)")
             
    return dict(data), model_labels

                  
# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_model_label(dirname: str) -> str:
    name = dirname.split(",,")[0]

    parts = name.split("_")

    model = parts[0]
    builder = parts[1] if len(parts) > 1 else ""

    # shorten names
    model = model.replace("qwen-", "Qwen").replace("b", "B")
    model = model.replace("mistral-", "Mistral-")
    model = model.replace("gemma-", "Gemma-")
    model = model.replace("llama-", "Llama-")
    model = model.replace("deepseek-v2-lite", "DeepSeek-Lite")

    builder = builder.replace("gpt-", "").replace("mini", "-mini")

    return f"{model} + {builder}"


def get_metric_at_turn(turn, key):
    if key in METRIC_KEYS:
        pd = turn.get("progress_data", {})
        if isinstance(pd, dict) and "metrics" in pd:
            return pd["metrics"].get(key, None)
    elif key in BINARY_KEYS:
        v = turn.get(key, None)
        return float(v) if v is not None else None
    return None


def compute_stats(series_list, total_len=TOTAL_LEN):
    means = np.full(total_len, np.nan)
    sems = np.full(total_len, np.nan)

    for t in range(total_len):
        try:
            vals = [s[t] for s in series_list if t < len(s) and s[t] is not None]
            if len(vals) >= 2:
                arr = np.array(vals, dtype=float)
                means[t] = np.mean(arr)
                sems[t] = np.std(arr, ddof=1) / np.sqrt(len(arr))
            elif len(vals) == 1:
                means[t] = vals[0]
                sems[t] = 0.0
        except Exception as e:
                    print(f"Error {e}")

    return means, sems


def build_struct_series(game):
    """
    Build one structure's metric time series.
    Index layout:
        0 = synthetic t=0
        1..N_TURNS = actual turns
    """
    turns = game["turns"]

    struct_vals = {}

    # Continuous metrics: synthetic t=0 baseline = 0.0
    for k in METRIC_KEYS:
        struct_vals[k] = [0.0] + [None] * N_TURNS

    # Binary metrics: no synthetic value at t=0
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

    # Delta metrics relative to turn 1
    for k in METRIC_KEYS:
        dk = f"{k}_delta"
        baseline = struct_vals[k][1]

        if baseline is not None:
            struct_vals[dk] = [None] + [
                (struct_vals[k][t] - baseline) if struct_vals[k][t] is not None else None
                for t in range(1, TOTAL_LEN)
            ]
            struct_vals[dk][1] = 0.0
        else:
            struct_vals[dk] = [None] * TOTAL_LEN

    return struct_vals


def discover_run_files_by_model(root_dir: Path):
    """
    Returns:
        run_files_by_model: dict[model_label][struct_id] = json_path
        model_labels: list[str]
    """
    model_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    run_files_by_model = {}
    model_labels = []

    print(f"Scanning root dir: {root_dir}")
    print(f"Found {len(model_dirs)} model directories\n")

    for model_dir in model_dirs:
        model_label = clean_model_label(model_dir.name)
        files = sorted(model_dir.glob("*.json"))

        groups = defaultdict(dict)
        for f in files:
            m = re.match(r"craft_structure_(\d+)_(\d+)\.json", f.name)
            if m:
                struct_id, run = m.group(1), int(m.group(2))
                groups[struct_id][run] = f

        run_files = {
            sid: groups[sid]
            for sid in groups
        }

        if len(run_files) == 0:
            print(f"Skipping {model_label:<28}")
            continue

        run_files_by_model[model_label] = run_files
        model_labels.append(model_label)
        print(f"Loaded  {model_label:<28} : {len(run_files)} structures")

    print()
    return run_files_by_model, model_labels


def print_summary_tables(data, model_labels):
    print(f"\n{'=' * 90}")
    print("RAW METRICS — turn-by-turn means (ALL structures per model)\n")
    print(f"{'=' * 90}")
    for modelKey in data.keys():
        print(f"\nModel Data: {modelKey}")
        for k in ALL_KEYS:
            means, sems = compute_stats(data[modelKey][k])
            print(f"\n{k}")
            print(f"{'turn':>5}  {'mean':>8}  {'sem':>8}  {'n':>5}")
            print("-" * 35)
            for t in range(TOTAL_LEN):
                try:
                    n_obs = sum(1 for s in data[modelKey][k] if t < len(s) and s[t] is not None)
                    m, se = means[t], sems[t]
                    ms = f"{m:.4f}" if not np.isnan(m) else "   n/a"
                    ss = f"{se:.4f}" if not np.isnan(se) else "   n/a"
                    print(f"{t:>5}  {ms:>8}  {ss:>8}  {n_obs:>5}")
                except Exception as e:
                    print(f"Error {e}")

    # print(f"\n{'=' * 90}")
    # print("DELTA METRICS — gain from turn 1 baseline (ALL structures pooled across models)")
    # print(f"{'=' * 90}")
    # for dk in DELTA_KEYS:
    #     means, sems = compute_stats(data["ALL"][dk])
    #     print(f"\n{dk}")
    #     print(f"{'turn':>5}  {'mean':>8}  {'sem':>8}  {'n':>5}")
    #     print("-" * 35)
    #     for t in range(1, TOTAL_LEN):
    #         n_obs = sum(1 for s in data["ALL"][dk] if s[t] is not None)
    #         m, se = means[t], sems[t]
    #         ms = f"{m:.4f}" if not np.isnan(m) else "   n/a"
    #         ss = f"{se:.4f}" if not np.isnan(se) else "   n/a"
    #         print(f"{t:>5}  {ms:>8}  {ss:>8}  {n_obs:>5}")

    print(f"\n{'=' * 90}")
    print("PER-MODEL COUNTS")
    print(f"{'=' * 90}")
    for model in model_labels:
        n_structs = len(data[model]["overall_progress"])
        print(f"{model:<28} : {n_structs} structures")

def plot_metric_grid(
    data,
    model_labels,
    colors,
    keys,
    title,
    fname,
    x_start=0,
    hline_val=0.0,
    ylabel="Value",
):
    n_cols = 2 if len(keys) > 1 else 1
    n_rows = int(np.ceil(len(keys) / n_cols))

    model_labels = sorted(
        model_labels,
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 7 * n_rows),
        constrained_layout=False,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # linestyles = ['-',  '--', '-.',  ':',  '-',  '--', '-.',  ':']
     
    linestyles = [
        (0, ()),           # solid
        (0, (5, 2)),       # dashed
        (0, (3, 1, 1, 1)), # dashdot tight
        (0, (1, 1)),       # dotted tight
        (0, (5, 1)),       # dashed tight
        (0, (3, 2, 1, 2)), # dashdot loose
        (0, (1, 2)),       # dotted loose
        (0, (5, 5)),       # dashed loose
    ]
    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    markersize = 5

    for ax_idx, key in enumerate(keys):
        ax = axes_flat[ax_idx]

        for idx, model in enumerate(model_labels):   # ← idx tracked here
            series_list = data[model].get(key, [])   # ← series_list defined here
            if len(series_list) == 0:
                continue

            means, sems = compute_stats(series_list)
            mask = (TURNS_AXIS >= x_start) & ~np.isnan(means)
            x = TURNS_AXIS[mask]
            y = means[mask]
            e = sems[mask]

            ax.plot(
                x, y,
                label=f"{model} (n={len(series_list)})",
                color=colors[model],
                linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)],
                linewidth=2,
                markersize=3,
                zorder=3,
            )
            # ax.fill_between(
            #     x,
            #     y - e,
            #     y + e,
            #     color=colors[model],
            #     alpha=0.15,
            #     zorder=2,
            # )

            #error bars at specific turns
            ax.errorbar(x[::2], y[::2], yerr=e[::2], fmt='none',
            color=colors[model], capsize=2, linewidth=0.5, alpha=0.2)

            #95 percent CI 
            # final_idx = np.where(mask)[0][-1]
            # ax.errorbar(x[-1], y[-1], yerr=1.96*e[-1],
            # fmt='none', color=colors[model], capsize=3, linewidth=1.2)

        display = key.replace("_delta", "").replace("_", " ").title()
        if key.endswith("_delta"):
            display += " (Δ from turn 1)"

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(x_start - 0.3, N_TURNS + 0.3)
        ax.set_xticks(range(x_start, N_TURNS + 1, 2))
        ax.axhline(
            hline_val,
            color="gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            zorder=1,
        )
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax_idx in range(len(keys), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(bottom=0.22, top=0.88, hspace=0.30, wspace=0.20)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=10,
        )

    #plt.show()
    plt.savefig(f"{display}.png")
    plt.close(fig)
# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_files_by_model, model_labels = discover_run_files_by_model(ROOT_DIR)

    if not model_labels:
        raise ValueError(
            f"No model directories with files found under {ROOT_DIR}"
        )
    data, model_labels = recompute_all_flat(ROOT_DIR)  # in some turns the dynamic progress tracker fails; but we can recompute the progress metrics entirely from the structure snapshot vs target

    print_summary_tables(data, model_labels)

    if len(model_labels) > 1:
        cmap = cm.get_cmap("tab10", len(model_labels))
        colors = {model: cmap(i) for i, model in enumerate(model_labels)}
    else:
        colors = {model_labels[0]: "#2563EB"}

    plot_metric_grid(
        data=data,
        model_labels=model_labels,
        colors=colors,
        keys=ALL_KEYS,
        title=(
            f"CRAFT Raw Metrics by Model Combination — Turn by Turn "
            f"(All Runs, t=0 anchored, SEM over structures)"
        ),
        fname=f"{SAVE_PREFIX}_raw.png",
        x_start=0,
        hline_val=0.0,
        ylabel="Value",
    )

    plot_metric_grid(
        data=data,
        model_labels=model_labels,
        colors=colors,
        keys=DELTA_KEYS,
        title=(
            f"CRAFT Metric Deltas by Model Combination — Gain from Turn 1 Baseline "
            f"(All Runs, turn 1 = 0, SEM over structures)"
        ),
        fname=f"{SAVE_PREFIX}_delta.png",
        x_start=1,
        hline_val=0.0,
        ylabel="Δ Value (from turn 1)",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()