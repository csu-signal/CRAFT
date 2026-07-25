import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

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

N_TURNS, TOTAL_LEN = 20, 21
METRIC_KEYS = ['overall_progress','completion_percentage','iou_score',
               'position_accuracy','distance_score']
BINARY_KEYS = ['move_executed','failed_move',
               'correct_structure_placement','correct_side_placement']

LABELS = {
    'qwen-7b':'Qwen-7B','qwen-14b':'Qwen-14B','qwen-32b':'Qwen-32B','qwen-72b':'Qwen-72B',
    'mistral-7b':'Mistral-7B','llama-8b':'Llama-8B','gemma-9b':'Gemma-9B',
    'deepseek-v2-lite':'DeepSeek-Lite',
    'gpt-4o':'GPT-4o','gpt-4o-mini':'GPT-4o-Mini','gpt-4.1-mini':'GPT-4.1-Mini',
    'claude-sonnet-4-6':'Claude-Sonnet-4.6','gemini-2.5-flash':'Gemini-2.5-Flash',
    'gemini-3-flash-preview':'Gemini-3-Flash',
    'gemini-3.1-flash-lite-preview':'Gemini-3.1-Flash-lite',
}
label = lambda d: LABELS.get(d.split(",,")[0].split("_")[0], d.split(",,")[0].split("_")[0])

def recompute(game):
    tracker = TaskProgressTracker(game["target_structure"])
    turns, n = game["turns"], game["turns_taken"]
    t0 = next((t for t in turns if t.get("turn_number") == 1), None)
    t0_struct = t0.get("structure_before", {}) if t0 else \
                {f"({i},{j})": [] for i in range(3) for j in range(3)}
    out = {k: [None]*(n+1) for k in METRIC_KEYS + BINARY_KEYS}
    for k, v in tracker.calculate_progress(t0_struct).items():
        if k in METRIC_KEYS: out[k][0] = v

    for turn in turns:
        tn = turn.get("turn_number")
        if tn is None or not 1 <= tn <= n: continue
        snap = turn.get("progress_data", {}).get("structure_snapshot") \
               or turn.get("structure_before", t0_struct)
        m = tracker.calculate_progress(snap)
        for k in METRIC_KEYS: out[k][tn] = m[k]
        for k in BINARY_KEYS: out[k][tn] = float(turn.get(k, 0) or 0)

    for k, s in out.items():                       # pad to TOTAL_LEN
        if len(s) < TOTAL_LEN:
            last = next((v for v in reversed(s) if v is not None), None)
            out[k] = s + [last if k in METRIC_KEYS else None]*(TOTAL_LEN-len(s))
    return out

def load(root, run_filter=3):
    data = defaultdict(lambda: defaultdict(list))
    for md in sorted(Path(root).iterdir()):
        if not md.is_dir(): continue
        lab = label(md.name)
        for f in sorted(md.glob("*.json")):
            m = re.match(r"craft_structure_(\d+)_(\d+)\.json", f.name)
            #if not m or int(m.group(2)) != run_filter: continue
            d = json.load(open(f))
            if not d.get("games"): continue
            s = recompute(d["games"][0])
            for k in METRIC_KEYS + BINARY_KEYS: data[lab][k].append(s[k])
    return data

def table(data, turn=20):
    rows = []
    for model, md in data.items():
        r = {'model': model}
        for k in ['overall_progress','completion_percentage',
                  'position_accuracy','iou_score','failed_move', 'distance_score']:
            v = [s[turn] for s in md[k] if len(s) > turn and s[turn] is not None]
            r[k], r[k+'_sem'], r[k+'_n'] = (np.mean(v),
                np.std(v, ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0, len(v))
        rows.append(r)
    return pd.DataFrame(rows).sort_values('overall_progress', ascending=False)

BASE = "/Users/hannahvanderhoeven/Documents/GitHub/LLM_Pragmatic_Analysis/data/craft_data/experiment1/"
FRON = "dpip_oracle_upper_bound_no_tools_gpt-4o-mini_closedmodel_geminifix"
raw = pd.concat([table(load(BASE))], ignore_index=True)
raw.to_csv("craftStatsExperiment1.csv")
print(raw.round(3).to_string(index=False))