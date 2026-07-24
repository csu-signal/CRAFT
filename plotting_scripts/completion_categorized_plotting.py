"""
plot_craft_metrics_by_category_and_model.py
------------------------------------------
Plots selected CRAFT metrics, split by partialCompletionCategory.

For each partialCompletionCategory:
    - creates one figure
    - each subplot is one selected metric
    - each line is one selected model combination
    - SEM bands are computed over structures

Expected directory structure:
ROOT_DIR/
    qwen-72b_gpt-4o-mini,,1773456132000/
        craft_structure_001_2.json
        craft_structure_002_2.json
        ...
    qwen-32b_gpt-4o-mini,,1773401844000/
        ...

Notes:
- Grouping is by folder/model combination + partialCompletionCategory from JSON
- Supports both raw metrics and delta metrics
- You can select models, metrics, and categories from the config block below
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import copy
from pathlib import Path

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

ROOT_DIR = Path("/home/hannah/CRAFT/CRAFT/craft_results/api/experiment1/gemini-3-flash-preview_gpt-5.4-mini").parent

RUN_FILTER = 3
N_TURNS = 20
SAVE_DIR = Path("craft_plots_by_category")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Set to None to include all discovered models/categories.
SELECTED_MODELS = None
# Example:
# SELECTED_MODELS = [
#     "Qwen72B + 4o-mini",
#     "Llama-8B + 4o-mini",
#     "Mistral-7B + 4o-mini",
# ]

SELECTED_CATEGORIES = None
# Example:
# SELECTED_CATEGORIES = ["partial_25", "partial_50"]

SELECTED_METRICS = [
    "distance_score_delta",
    "position_accuracy_delta",
    "overall_progress_delta",
    "completion_percentage_delta",
]

# Raw metrics available:
#   overall_progress
#   completion_percentage
#   iou_score
#   position_accuracy
#   distance_score
#   move_executed
#   failed_move
#   correct_structure_placement
#   correct_side_placement
#
# Delta metrics available:
#   overall_progress_delta
#   completion_percentage_delta
#   iou_score_delta
#   position_accuracy_delta
#   distance_score_delta


# ── Metric definitions ────────────────────────────────────────────────────────

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

SELECTED_MODELS = [
    "DeepSeek-Lite + 4o-mini",
    "Qwen7B + 4o-mini",
    "Mistral-7B + 4o-mini",
    "Llama-8B + 4o-mini",
    "Gemma-9B + 4o-mini",
    "Qwen14B + 4o-mini",
    "Qwen32B + 4o-mini",
    "Qwen72B + 4o-mini",
]
 

SELECTED_CATEGORIES = None

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


def recompute_all(root_dir: Path, run_filter: int):
    """
    Walk all model dirs and recompute metrics for every game file.
    Returns same data structure as the plotting code expects:
        data[category][model_label][metric] = list of per-structure series
    """
    from collections import defaultdict
    import re

    def clean_model_label(dirname):
        name = dirname.split(",,")[0]
        parts = name.split("_")
        model_map = {
            "qwen-72b": "Qwen72B", "qwen-32b": "Qwen32B",
            "qwen-14b": "Qwen14B", "qwen-7b": "Qwen7B",
            "mistral-7b": "Mistral-7B", "llama-8b": "Llama-8B",
            "gemma-9b": "Gemma-9B", "deepseek-v2-lite": "DeepSeek-Lite",
        }
        builder_map = {"gpt-4o-mini": "4o-mini"}
        model  = model_map.get(parts[0], parts[0])
        builder = builder_map.get(parts[1], parts[1]) if len(parts) > 1 else ""
        return f"{model} + {builder}" if builder else model

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model_labels = []

    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_label = clean_model_label(model_dir.name)
        found = False

        for fpath in sorted(model_dir.glob("*.json")):
            m = re.match(r"craft_structure_(\d+)_(\d+)\.json", fpath.name)
            if not m or int(m.group(2)) != run_filter:
                continue

            with open(fpath) as f:
                d = json.load(f)

            if "games" not in d or not d["games"]:
                continue

            game = d["games"][0]
            category = game.get("partialCompletionCategory", "unknown")

            # recompute from snapshots
            struct_vals = recompute_metrics_from_logs(game)

            for k, series in struct_vals.items():
                data[category][model_label][k].append(series)

            found = True

        if found and model_label not in model_labels:
            model_labels.append(model_label)
            print(f"Recomputed: {model_label}")

    return data, model_labels
    
# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_model_label(dirname: str) -> str:
    """
    Converts directory name into a shorter legend label.
    Example:
        qwen-72b_gpt-4o-mini,,1773456132000 -> Qwen72B + 4o-mini
    """
    name = dirname.split(",,")[0]
    parts = name.split("_")

    model = parts[0] if len(parts) > 0 else name
    builder = parts[1] if len(parts) > 1 else ""

    model_map = {
        "qwen-72b": "Qwen72B",
        "qwen-32b": "Qwen32B",
        "qwen-14b": "Qwen14B",
        "qwen-7b": "Qwen7B",
        "mistral-7b": "Mistral-7B",
        "llama-8b": "Llama-8B",
        "gemma-9b": "Gemma-9B",
        "deepseek-v2-lite": "DeepSeek-Lite",
    }
    builder_map = {
        "gpt-4o-mini": "4o-mini",
    }

    model = model_map.get(model, model)
    builder = builder_map.get(builder, builder)

    return f"{model} + {builder}" if builder else model


def safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip())


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
        vals = [s[t] for s in series_list if s[t] is not None]
        if len(vals) >= 2:
            arr = np.array(vals, dtype=float)
            means[t] = np.mean(arr)
            sems[t] = np.std(arr, ddof=1) / np.sqrt(len(arr))
        elif len(vals) == 1:
            means[t] = vals[0]
            sems[t] = 0.0

    return means, sems


def build_struct_series(game):
    """
    Build one structure's time series.
    Index layout:
        0 = synthetic t=0
        1..N_TURNS = actual turns
    """
    turns = game["turns"]
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


    # for k in METRIC_KEYS:
    #     dk = f"{k}_delta"
    
    #     baseline_idx = next(
    #         (t for t in range(1, TOTAL_LEN) if struct_vals[k][t] is not None),
    #         None
    #     )
    
    #     if baseline_idx is not None:
    #         baseline = struct_vals[k][baseline_idx]
    
    #         struct_vals[dk] = [None] + [
    #             (struct_vals[k][t] - baseline) if struct_vals[k][t] is not None else None
    #             for t in range(1, TOTAL_LEN)
    #         ]
    
    #         struct_vals[dk][baseline_idx] = 0.0
    #     else:
    #         struct_vals[dk] = [None] * TOTAL_LEN
        
    # if baseline_idx is not None:
    #     baseline = struct_vals[k][baseline_idx]

    #     struct_vals[dk] = [None] + [
    #         (struct_vals[k][t] - baseline) if struct_vals[k][t] is not None else None
    #         for t in range(1, TOTAL_LEN)
    #     ]

    #     struct_vals[dk][baseline_idx] = 0.0
    # else:
    #     struct_vals[dk] = [None] * TOTAL_LEN
        
    for k in METRIC_KEYS:
        dk = f"{k}_delta"
    
        baseline_idx = next(
            (t for t in range(1, TOTAL_LEN) if struct_vals[k][t] is not None),
            None
        )
    
        if baseline_idx is not None:
            baseline = struct_vals[k][baseline_idx]
    
            struct_vals[dk] = [None] + [
                (struct_vals[k][t] - baseline) if struct_vals[k][t] is not None else None
                for t in range(1, TOTAL_LEN)
            ]
    
            struct_vals[dk][baseline_idx] = 0.0
        else:
            struct_vals[dk] = [None] * TOTAL_LEN

    return struct_vals


def discover_run_files_by_model(root_dir: Path, run_filter: int):
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
            sid: groups[sid][run_filter]
            for sid in groups
            if run_filter in groups[sid]
        }

        if not run_files:
            print(f"Skipping {model_label:<24} : no run {run_filter} files found")
            continue

        run_files_by_model[model_label] = run_files
        model_labels.append(model_label)
        print(f"Loaded  {model_label:<24} : {len(run_files)} structures")

    print()
    return run_files_by_model, model_labels


def resolve_selected_items(all_items, selected_items, item_name="items"):
    if selected_items is None:
        return list(all_items)

    resolved = [x for x in selected_items if x in all_items]
    missing = [x for x in selected_items if x not in all_items]

    if missing:
        print(f"Warning: requested {item_name} not found: {missing}")

    return resolved


def print_data_summary(data, categories_to_plot, models_to_plot, metrics_to_plot):
    print(f"\n{'=' * 100}")
    print("DATA SUMMARY")
    print(f"{'=' * 100}")

    for category in categories_to_plot:
        print(f"\nCategory: {category}")
        for model in models_to_plot:
            counts = {
                metric: len(data[category][model][metric])
                for metric in metrics_to_plot
            }
            max_n = max(counts.values()) if counts else 0
            print(f"  {model:<24} n={max_n}")


def plot_category_metric_grid(
    data,
    category,
    models_to_plot,
    metric_keys,
    colors,
    fname,
):
    n_cols = 2 if len(metric_keys) > 1 else 1
    n_rows = int(np.ceil(len(metric_keys) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.5 * n_rows),
        constrained_layout=False,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    all_delta = all(k.endswith("_delta") for k in metric_keys)
    x_start = 1 if all_delta else 0
    ylabel = "Δ Value (from turn 1)" if all_delta else "Value"
    # Option 1 — vary both linestyle and marker, easier to distinguish in grayscale/print
    linestyles = ['-',  '--', '-.',  ':',  '-',  '--', '-.',  ':' ]
    markers    = ['o',  's',  '^',  'D',  'v',  'P',  'X',  '*' ]
    
   
    
     
     
    for ax_idx, key in enumerate(metric_keys):
        ax = axes_flat[ax_idx]

        for idx, model in enumerate(models_to_plot):
            series_list = data[category][model].get(key, [])
            # series_list = data["firstLayer"][model]["overall_progress_delta"]
            # means, _ = compute_stats(series_list)
            # print(model, "valid points:", np.sum(~np.isnan(means)))
            if len(series_list) == 0:
                continue

            means, sems = compute_stats(series_list)

            mask = (TURNS_AXIS >= (1 if key.endswith("_delta") else 0)) & ~np.isnan(means)
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
                markersize=4,
            )
                    # ax.fill_between(
            #     x,
            #     y - e,
            #     y + e,
            #     color=colors[model],
            #     alpha=0.15,
            #     zorder=2,
            # )

        display = key.replace("_delta", "").replace("_", " ").title()
        if key.endswith("_delta"):
            display += " (Δ from turn 1)"

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim((1 if key.endswith("_delta") else 0) - 0.3, N_TURNS + 0.3)
        ax.set_xticks(range((1 if key.endswith("_delta") else 0), N_TURNS + 1, 2))
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax_idx in range(len(metric_keys), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(
        f"CRAFT Metrics by Model — Partial Completion: {category}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.subplots_adjust(bottom=0.22, top=0.88, hspace=0.30, wspace=0.20)

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

    # fig.savefig(fname, dpi=160, bbox_inches="tight")
    # print(f"Saved {fname}")
    plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_files_by_model, discovered_models = discover_run_files_by_model(ROOT_DIR, RUN_FILTER)

    
    if not discovered_models:
        raise ValueError(
            f"No model directories with run {RUN_FILTER} files found under {ROOT_DIR}"
        )

    metrics_to_plot = []
    for metric in SELECTED_METRICS:
        if metric in ALL_KEYS or metric in DELTA_KEYS:
            metrics_to_plot.append(metric)
        else:
            print(f"Warning: unknown metric skipped: {metric}")

    if not metrics_to_plot:
        raise ValueError("No valid metrics selected.")

    # data[category][model][metric] = list of structure series
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    data, discovered_models = recompute_all(ROOT_DIR, RUN_FILTER) # in some turns the dynamic progress tracker fails; but we can recompute the progress metrics entirely from the structure snapshot vs target
    discovered_categories = sorted(data.keys())
    categories_to_plot = resolve_selected_items(
        discovered_categories,
        SELECTED_CATEGORIES,
        item_name="categories",
    )
    models_to_plot = resolve_selected_items(
        discovered_models,
        SELECTED_MODELS,
        item_name="models",
    )

    if not categories_to_plot:
        raise ValueError("No matching categories selected.")
    if not models_to_plot:
        raise ValueError("No matching models selected.")

    print("\nModels to plot:")
    for m in models_to_plot:
        print(" ", m)

    print("\nCategories to plot:")
    for c in categories_to_plot:
        print(" ", c)

    print("\nMetrics to plot:")
    for k in metrics_to_plot:
        print(" ", k)

    print_data_summary(data, categories_to_plot, models_to_plot, metrics_to_plot)

    if len(models_to_plot) > 1:
        cmap = cm.get_cmap("tab10", len(models_to_plot))
        colors = {model: cmap(i) for i, model in enumerate(models_to_plot)}
    else:
        colors = {models_to_plot[0]: "#2563EB"}

    for category in categories_to_plot:
        out_name = SAVE_DIR / f"craft_{safe_filename(category)}.png"
        plot_category_metric_grid(
            data=data,
            category=category,
            models_to_plot=models_to_plot,
            metric_keys=metrics_to_plot,
            colors=colors,
            fname=out_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()