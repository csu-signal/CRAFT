import copy
import json
import random
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
from datetime import datetime
from task_progress_tracker import TaskProgressTracker
from structure_generator_v2 import (
    generate_dataset,
    get_director_views as get_director_views_fn,
    validate_placement_action,
    apply_placement_action,
    ALL_COORDS, OPTIONAL, REQUIRED_FULL,
       get_block_encoding_reference,       
    get_coordinate_system_reference     
)
# from agents.oracle import enumerate_correct_actions
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
from agents.builder_tools import simulate_move
        
def char_to_color(c):
    return {
        "b":"blue","o":"orange","g":"green","y":"yellow","r":"red","n":"none"
    }.get(c, "unknown") 


_ALLOWED_ACTIONS = {"place", "remove", "none"}
_ALLOWED_BLOCKS = {"gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol", "none"}
_ALLOWED_LAYERS = {"0", "1", "2", "none"}

# Keep this in sync with your prompt list
_ALLOWED_REL_SLOTS = {
    "my_bottom_left",
    "my_bottom_middle",
    "my_bottom_right",
    "my_middle_left",
    "my_middle_middle",
    "my_middle_right",
    "my_top_left",
    "my_top_middle",
    "my_top_right",
    "my_bottom_left_to_middle",
    "my_bottom_middle_to_right",
    "my_middle_left_to_middle",
    "my_middle_middle_to_right",
    "my_top_left_to_middle",
    "my_top_middle_to_right",
    "none",
}


#    <intent>
# action: place|remove|none
# block: gs|gl|bs|bl|rs|rl|ys|yl|os|ol|none
# target_layer: 0|1|2|none
# relative_slot: one of
#   - my_bottom_left
#   - my_bottom_middle
#   - my_bottom_right
#   - my_middle_left
#   - my_middle_middle
#   - my_middle_right
#   - my_top_left
#   - my_top_middle
#   - my_top_right
#   - my_bottom_left_to_middle
#   - my_bottom_middle_to_right
#   - my_middle_left_to_middle
#   - my_middle_middle_to_right
#   - my_top_left_to_middle
#   - my_top_middle_to_right
# note: optional short note (e.g., "do after D1 orange corner")
# </intent>



def parse_intent(raw_response: str) -> Dict[str, Any]:
    """
    Parse a Director's <intent> ... </intent> block.

    Expected format (order flexible, keys case-insensitive):
      <intent>
      action: place|remove|none
      block: gs|gl|...|none
      target_layer: 0|1|2|none
      relative_slot: my_bottom_left|...|none
      note: optional text...
      </intent>

    Returns a normalized dict with:
      {
        "ok": bool,
        "action": str,
        "block": str,
        "target_layer": Optional[int],
        "relative_slot": str,
        "note": str,
        "errors": [..],
        "raw_intent": str
      }
    """
    result: Dict[str, Any] = {
        "ok": False,
        "action": "none",
        "block": "none",
        "target_layer": None,
        "relative_slot": "none",
        "note": "",
        "errors": [],
        "raw_intent": "",
    }

    if not raw_response or not isinstance(raw_response, str):
        result["errors"].append("raw_response missing or not a string")
        return result

    m = re.search(r"<intent>\s*(.*?)\s*</intent>", raw_response, re.DOTALL | re.IGNORECASE)
    if not m:
        result["errors"].append("no <intent> block found")
        return result

    intent_text = m.group(1).strip()
    result["raw_intent"] = intent_text

    # Normalize lines: allow either "key: value" per line OR "key=value"
    lines = []
    for ln in intent_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # ignore bullets if model outputs "- key: value"
        if ln.startswith("-"):
            ln = ln.lstrip("-").strip()
        lines.append(ln)

    fields: Dict[str, str] = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
        elif "=" in ln:
            k, v = ln.split("=", 1)
        else:
            # allow "note ..." freeform line
            if "note" not in fields:
                fields["note"] = ln
            else:
                fields["note"] += " " + ln
            continue

        k = k.strip().lower()
        v = v.strip()

        # Canonicalize key aliases
        if k in {"layer", "targetlayer", "target_layer", "target-layer"}:
            k = "target_layer"
        elif k in {"relativeslot", "relative_slot", "relative-slot", "slot"}:
            k = "relative_slot"

        # Store (last one wins, but accumulate note)
        if k == "note":
            fields["note"] = (fields.get("note", "") + " " + v).strip()
        else:
            fields[k] = v

    # ---- action ----
    action = fields.get("action", "none").strip().lower()
    if action not in _ALLOWED_ACTIONS:
        result["errors"].append(f"invalid action: {action}")
        action = "none"
    result["action"] = action

    # ---- block ----
    block = fields.get("block", "none").strip().lower()
    # Sometimes models output "blue large" etc. Try to map minimal.
    # If it's not a valid code, leave as none.
    if block not in _ALLOWED_BLOCKS:
        result["errors"].append(f"invalid block: {block}")
        block = "none"
    result["block"] = block

    # ---- target_layer ----
    tl = fields.get("target_layer", "none").strip().lower()
    if tl not in _ALLOWED_LAYERS:
        # try to extract a digit 0/1/2
        mtl = re.search(r"\b([0-2])\b", tl)
        tl = mtl.group(1) if mtl else "none"
    if tl not in _ALLOWED_LAYERS:
        result["errors"].append(f"invalid target_layer: {fields.get('target_layer')}")
        result["target_layer"] = None
    else:
        result["target_layer"] = None if tl == "none" else int(tl)

    # ---- relative_slot ----
    rs = fields.get("relative_slot", "none").strip()
    rs_norm = rs.lower().replace(" ", "_")
    # Some models may output MY_BOTTOM_LEFT; normalize
    rs_norm = rs_norm.replace("__", "_")
    if rs_norm not in _ALLOWED_REL_SLOTS:
        result["errors"].append(f"invalid relative_slot: {rs}")
        rs_norm = "none"
    result["relative_slot"] = rs_norm

    # ---- note ----
    result["note"] = fields.get("note", "").strip()

    # Basic completeness check
    # ok if action is place/remove and has a valid block & slot & target_layer
    if result["action"] in {"place", "remove"}:
        missing = []
        if result["block"] == "none":
            missing.append("block")
        if result["relative_slot"] == "none":
            missing.append("relative_slot")
        if result["target_layer"] is None:
            missing.append("target_layer")
        if missing:
            result["errors"].append("missing required fields: " + ", ".join(missing))
        else:
            result["ok"] = True
    else:
        # action none: still valid parse, but not usable as a proposal
        result["ok"] = False

    return result


# ═════════════════════════════════════════════════════════════════════════════
# STEP Null — Orcale comes in 
# ═════════════════════════════════════════════════════════════════════════════
# Sentinel strings used in the "flag" field of returned dicts.
FLAG_OK                  = "ok"
FLAG_MISSING_SPAN        = "missing_span_info"
FLAG_BLOCKED_REMOVE      = "blocked_correct_remove"
FLAG_SIM_FAILED          = "sim_failed"
FLAG_LARGE_SKIP_ENDPOINT = "large_block_secondary_endpoint_skipped"

def get_oracle_moves(state, n=5, rng=None):
    """
    Returns up to N correct moves in clean format for builder prompt.
    Only includes flag=ok entries. Samples randomly if more than N available.
    """
 
    from agents.oracle import enumerate_correct_actions 
    all_correct = enumerate_correct_actions(state)
    ok_moves = [
        entry["move"] 
        for entry in all_correct 
        if entry["flag"] == FLAG_OK
    ]
    
    if rng and len(ok_moves) > n:
        ok_moves = rng.sample(ok_moves, n)
    elif len(ok_moves) > n:
        ok_moves = ok_moves[:n]
    
    return ok_moves


class EnhancedGameState:
    """Enhanced GameState class that integrates with all agents and progress tracking"""
    PARTIAL_OPTIONS = ["empty", "firstLayer", "firstTwoLayers", "D1Wall", "D2Wall","D3Wall"]

    def __init__(self, target_structure, target_spans=None, available_blocks=None, strict_target=False, invisible_cells=None, partComplete = False, partType= None, current_structure= None):
        # self.current_structure = {f"({i}, {j})": [] for i in range(3) for j in range(3)} #remove space
        self.current_structure = {f"({i},{j})": [] for i in range(3) for j in range(3)} if current_structure != None else current_structure
        self.target_structure = target_structure
        self.partType = EnhancedGameState.PARTIAL_OPTIONS[random.randint(0, 5)] if partType == None else partType
        if(partComplete):
            match self.partType:
                case "firstLayer":
                    for c in self.target_structure:
                        # start part complete, copy the bottom layer from the target into the current
                        if(len(self.target_structure[c]) > 0):
                            self.current_structure[c].append(self.target_structure[c][0])
                case "firstTwoLayers":
                    for c in self.target_structure:
                        # start part complete, copy the bottom 2 layers from the target into the current
                        if(len(self.target_structure[c]) > 0):
                            self.current_structure[c].append(self.target_structure[c][0])
                        if(len(self.target_structure[c]) > 1):
                            self.current_structure[c].append(self.target_structure[c][1])
                case "D1Wall":
                     self.current_structure["(0,0)"] = self.target_structure["(0,0)"]
                     self.current_structure["(1,0)"] = self.target_structure["(1,0)"]
                     self.current_structure["(2,0)"] = self.target_structure["(2,0)"]
                case "D2Wall":
                     self.current_structure["(0,0)"] = self.target_structure["(0,0)"]
                     self.current_structure["(0,1)"] = self.target_structure["(0,1)"]
                     self.current_structure["(0,2)"] = self.target_structure["(0,2)"]
                case "D3Wall":
                     self.current_structure["(0,2)"] = self.target_structure["(0,2)"]
                     self.current_structure["(1,2)"] = self.target_structure["(1,2)"]
                     self.current_structure["(2,2)"] = self.target_structure["(2,2)"]

        # structure = {f"({i}, {j})": [] for i in range(3) for j in range(3)}  # With spaces
        self.target_spans = target_spans or {}
        self.strict_target = strict_target
        self.current_spans = {}  # tracks spans of current placed blocks
        self.turn_count = 0
        self.conversation_history = []
        self.move_history = []
        GAME_INVISIBLE_CELLS = {"(1,1)", "(2,1)"}
        self.invisible_cells = set(invisible_cells) if invisible_cells is not None else GAME_INVISIBLE_CELLS    
        
        # Initialize progress tracker
        self.progress_tracker = TaskProgressTracker(target_structure)
        
        # Available blocks for the game
        if available_blocks is None:
            self.available_blocks = [
               "gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol"
            ]
        else:
            self.available_blocks = available_blocks


    def get_director_views(self, spans=None):
        if spans is None:
            spans = self.current_spans
        return get_director_views_fn(self.current_structure, spans=spans)

    def get_target_director_views(self, spans=None):
        if spans is None:
            spans = self.target_spans
        return get_director_views_fn(self.target_structure, spans=spans)
    
    def getTargetDirectorViews(self):
        d1 = {key:  self.target_structure[key] for key in ["(0,0)", "(1,0)", "(2,0)"]}
        d2 = {key:  self.target_structure[key] for key in ["(0,2)", "(0,1)", "(0,0)"]}
        d3 = {key:  self.target_structure[key] for key in ["(2,2)", "(1,2)", "(0,2)"]}
        return d1, d2, d3

 

    def execute_move(self, move_json) -> Tuple[bool, Dict[str, Any], bool, bool, bool]:
        """
        Execute a move and update game state

        Args:
            move_json: Parsed move from builder

        Returns:
            Tuple of (success: bool, progress_data: dict, structurePlacement: bool,
                    sidePlacement: bool, overallState: bool)
        """
        try:
            action  = move_json.get("action")
            position = move_json.get("position")
            block   = move_json.get("block")  # often None for REMOVE given your parser
            layer   = move_json.get("layer", 0)
            span_to = move_json.get("span_to", None)

            # Add this safety normalization:
            if action == "remove" and block is None:
                # Infer block from current stack for any code that needs it
                stack = self.current_structure.get(position, [])
                if stack:
                    move_json['block'] = stack[-1]
                    block = stack[-1]

            # will be used for REMOVE scoring
            removed_block = None

            valid, reason = self._validate_move(move_json)
            if not valid:
                return False, {"error": reason}, False, False, False

            success = False
            structurePlacement = False
            sidePlacement = False

            # -------------------- EXECUTE MOVE + STRUCTURE PLACEMENT --------------------
            if action == "place":
                success = self._place_block(position, block, layer, span_to=span_to)

                # Correct if target at that exact cell/layer matches the placed block
                try:
                    structurePlacement = self.target_structure[position][layer] == block
                except Exception:
                    # If target doesn't have that layer/cell, it's incorrect to place there
                    structurePlacement = False

            elif action == "remove":
                # determine what is actually being removed (top of stack) BEFORE removing
                stack_before = self.current_structure.get(position, [])
                removed_block = stack_before[-1] if len(stack_before) > 0 else None

                success = self._remove_block(position, layer, span_to=span_to)

                if not success or removed_block is None:
                    return False, {"error": "Move execution failed"}, False, False, False

                # Correct if target does NOT want the removed block at that cell/layer
                try:
                    structurePlacement = self.target_structure[position][layer] != removed_block
                except Exception:
                    # If target doesn't even have that layer, removing extra is correct
                    structurePlacement = True

            else:
                return False, {"error": f"Unknown action: {action}"}, False, False, False

            # -------------------- SIDE PLACEMENT (ANY VISIBLE DIRECTOR) --------------------
            # Evaluate against the director-visible target views.
            # For PLACE: correct if any visible side expects `block` at that (pos,layer).
            # For REMOVE: correct if any visible side does NOT expect `removed_block` at that (pos,layer).
            eval_block = None
            if action == "place":
                eval_block = block
            elif action == "remove":
                eval_block = removed_block

            sidePlacement = False
            if eval_block is not None:
                d1_view, d2_view, d3_view = self.getTargetDirectorViews()

                def side_check(view_dict):
                    # None => that side cannot see this position
                    if position not in view_dict:
                        return None
                    try:
                        if action == "place":
                            return view_dict[position][layer ] == eval_block
                        else:  # remove; HINT If we remove a block, the removal is correct only if that block should not be there according to the target.
                            return view_dict[position][layer] != eval_block 
                    except Exception:
                        # If that layer doesn't exist in that side's target:
                        # - placing there is incorrect
                        # - removing extra is correct
                        return False if action == "place" else True

                checks = [side_check(d1_view), side_check(d2_view), side_check(d3_view)]
                checks = [c for c in checks if c is not None]  # keep only sides that can see position
                sidePlacement = any(checks) if checks else False

            # -------------------- PROGRESS + OVERALL STATE --------------------
            if success:
                self.move_history.append(move_json)

                progress_data = self.progress_tracker.track_move(
                    move_json, self.current_structure, self.turn_count
                )

                overallState = True
                for coord in self.current_structure:
                    for idx, placed in enumerate(self.current_structure[coord]):
                        try:
                            if self.target_structure[coord][idx] != placed:
                                overallState = False
                                break
                        except Exception:
                            overallState = False
                            break
                    if not overallState:
                        break

                return True, progress_data, structurePlacement, sidePlacement, overallState

            return False, {"error": "Move execution failed"}, False, False, False

        except Exception as e:
            # IMPORTANT: keep return arity consistent (5 values)
            return False, {"error": f"Move execution error: {str(e)}"}, False, False, False
        
    def _validate_move(self, move_json) -> Tuple[bool, str]:
        """Validate if a move is legal. Returns (is_valid, reason)."""
        action  = move_json.get('action')
        position = move_json.get('position')
        block   = move_json.get('block')
        layer   = move_json.get('layer', 0)
        span_to = move_json.get('span_to', None)

        if not action or not position:
            return False, f"Missing fields: action={action}, position={position}"

        # ── Normalize position ────────────────────────────────────
        if '(' in position:
            try:
                coords = position.strip('()').replace(' ', '').split(',')
                if len(coords) == 2:
                    position = f"({coords[0].strip()},{coords[1].strip()})"
                    move_json['position'] = position
                else:
                    return False, f"Bad coordinate format: {coords}"
            except Exception as e:
                return False, f"Position normalization failed: {e}"

        if not re.match(r'\(\d,\d\)', position):
            return False, f"Invalid position format: {position}"

        try:
            x, y = map(int, re.findall(r'\d', position))
            if not (0 <= x <= 2 and 0 <= y <= 2):
                return False, f"Coordinates out of bounds: ({x},{y})"
        except Exception as e:
            return False, f"Coordinate parsing failed: {e}"

        # ── Place ─────────────────────────────────────────────────
        if action == "place":
            if block not in self.available_blocks:
                return False, f"Block '{block}' not in available blocks"

            current_stack = self.current_structure.get(position, [])

            # Layer must be next available — always enforced regardless of strict_target
            if layer != len(current_stack):
                return False, (
                    f"Wrong layer at {position}: got {layer}, "
                    f"expected {len(current_stack)} (stack has {len(current_stack)} blocks)"
                )

            # Stack height — always enforced
            if len(current_stack) >= 3:
                return False, f"Stack full at {position} (3 blocks already)"

            # ── Large block ───────────────────────────────────────
            if block.endswith('l'):

                # Self-span check — always enforced
                if span_to == position:
                    return False, f"span_to cannot be same as position: {position}"

                # span_to required — always enforced for large blocks
                if not span_to:
                    return False, (
                        f"Large block '{block}' needs span_to — "
                        f"specify which adjacent cell it extends into"
                    )

                # Normalize span_to
                if '(' in span_to:
                    try:
                        coords2 = span_to.strip('()').replace(' ', '').split(',')
                        if len(coords2) == 2:
                            span_to = f"({coords2[0].strip()},{coords2[1].strip()})"
                            move_json['span_to'] = span_to
                    except Exception:
                        return False, f"span_to normalization failed: {span_to}"

                INVISIBLE = {"(1,1)", "(2,1)"}

                # if large block and either endpoint is invisible, forbid but allow user to either use it in both structure + validation or remove both
                if block.endswith("l") and self.invisible_cells:
                    if position in self.invisible_cells or span_to in self.invisible_cells:
                        return False, (
                            f"Illegal span into invisible cell: {position}↔{span_to}. "
                            f"Large blocks cannot include {self.invisible_cells}."
                        )

                if not re.match(r'\(\d,\d\)', span_to):
                    return False, f"Invalid span_to format: {span_to}"

                # Orthogonal neighbor — always enforced (physical constraint)
                from structure_generator_v2 import orthogonal_neighbors
                if span_to not in orthogonal_neighbors(position):
                    return False, (
                        f"span_to {span_to} is not adjacent to {position} — "
                        f"valid neighbors: {orthogonal_neighbors(position)}"
                    )

                # Neighbor at same layer — always enforced (physical constraint)
                neighbor_stack = self.current_structure.get(span_to, [])
                if len(neighbor_stack) != len(current_stack):
                    return False, (
                        f"span_to {span_to} has {len(neighbor_stack)} blocks, "
                        f"{position} has {len(current_stack)} — "
                        f"both must be at same height to place on same layer"
                    )

                # Neighbor not full — always enforced
                if len(neighbor_stack) >= 3:
                    return False, f"Neighbor {span_to} stack is already full"

                # Neighbor not already spanned — always enforced
                layer_spans = self.current_spans.get(layer, [])
                if any(position in (a, b) or span_to in (a, b) for a, b in layer_spans):
                    return False, (
                        f"{position} or {span_to} already occupied by another span at layer {layer}"
                    )

                # ── Target checks for large block (strict_target only) ──
                if self.strict_target:
                    target_stack_pos = self.target_structure.get(position, [])
                    target_stack_nbr = self.target_structure.get(span_to, [])

                    if layer >= len(target_stack_pos):
                        return False, f"{position} layer {layer} does not exist in target"
                    if layer >= len(target_stack_nbr):
                        return False, f"{span_to} layer {layer} does not exist in target"

                    if block != target_stack_pos[layer]:
                        return False, (
                            f"Wrong block at {position} layer {layer}: "
                            f"placed '{block}' but target expects '{target_stack_pos[layer]}'"
                        )
                    if block != target_stack_nbr[layer]:
                        return False, (
                            f"Wrong block at span_to {span_to} layer {layer}: "
                            f"placed '{block}' but target expects '{target_stack_nbr[layer]}'"
                        )

                    target_layer_spans = self.target_spans.get(layer, [])
                    intended = tuple(sorted([position, span_to]))
                    target_span_tuples = [tuple(sorted([a, b])) for a, b in target_layer_spans]
                    if intended not in target_span_tuples:
                        return False, (
                            f"Span {position}↔{span_to} at layer {layer} not in target — "
                            f"valid spans at layer {layer}: {target_layer_spans}"
                        )

            # ── Small block ───────────────────────────────────────
            else:
                if self.strict_target:
                    target_stack = self.target_structure.get(position, [])
                    if layer >= len(target_stack):
                        return False, f"{position} layer {layer} does not exist in target"
                    if block != target_stack[layer]:
                        return False, (
                            f"Wrong block at {position} layer {layer}: "
                            f"placed '{block}' but target expects '{target_stack[layer]}'"
                        )

        # ── Remove ────────────────────────────────────────────────
        elif action == "remove":
            stack = self.current_structure.get(position, [])
            if len(stack) == 0:
                return False, f"Cannot remove from {position} — stack is empty"
            if layer != len(stack) - 1:
                return False, (
                    f"Cannot remove layer {layer} at {position} — "
                    f"must remove top block first (layer {len(stack)-1})"
                )

            # Large block remove requires span_to — always enforced
            top_block = stack[-1]
            if top_block.endswith('l'):
                span_layer = len(stack) - 1
                layer_spans = self.current_spans.get(span_layer, [])
                partner = next(
                    (b if a == position else a
                    for a, b in layer_spans if position in (a, b)),
                    None
                )

                if partner is None:
                    return False, f"Large block at {position} layer {layer} has no recorded span partner"

                if not span_to:
                    return False, (
                        f"Cannot remove large block at {position} layer {layer} alone — "
                        f"must also specify span_to={partner}"
                    )

                # ✅ Normalize and verify span_to matches partner
                coords2 = span_to.strip('()').replace(' ', '').split(',')
                if len(coords2) == 2:
                    span_to_norm = f"({coords2[0].strip()},{coords2[1].strip()})"
                    move_json['span_to'] = span_to_norm
                    span_to = span_to_norm

                if span_to != partner:
                    return False, (
                        f"Incorrect span_to for large block removal at {position} layer {layer}: "
                        f"got {span_to}, expected {partner}"
                    )

        else:
            return False, f"Unknown action: '{action}'"

        return True, "ok"

    def _place_block(self, position, block, layer, span_to=None) -> bool:
        """
        Place a block at position.
        If block is large, places on both position and span_to atomically.
        """
        try:
            block_encoded = self._encode_block(block)

            # ── Large block: place on both cells atomically ────────
            if block_encoded.endswith('l'):
                if not span_to:
                    print(f"DEBUG: Large block '{block_encoded}' requires span_to")
                    return False

                if position not in self.current_structure:
                    self.current_structure[position] = []
                if span_to not in self.current_structure:
                    self.current_structure[span_to] = []
                # optional safety: neighbor must be same height
                if len(self.current_structure[position]) != len(self.current_structure[span_to]):
                    print("DEBUG: large block placement requires equal stack heights")
                    return False

                self.current_structure[position].append(block_encoded)
                self.current_structure[span_to].append(block_encoded)

                # Record span
                self.current_spans.setdefault(layer, []).append((position, span_to))

                print(f"DEBUG: Placed large '{block_encoded}' "
                    f"at {position}↔{span_to} layer {layer}")

            # ── Small block: single cell ───────────────────────────
            else:
                if position not in self.current_structure:
                    self.current_structure[position] = []
                self.current_structure[position].append(block_encoded)
                print(f"DEBUG: Placed small '{block_encoded}' "
                    f"at {position} layer {layer}")

            return True

        except Exception as e:
            print(f"Error placing block: {e}")
            return False
    
    def _remove_block(self, position, layer, span_to=None) -> bool:
        """
        Remove the top block from position.
        If block is large, removes from both position and span partner atomically.
        """
        try:
            stack = self.current_structure.get(position, [])

            if len(stack) == 0:
                print(f"DEBUG: Cannot remove — stack at {position} is empty")
                return False

            if layer != len(stack) - 1:
                print(
                    f"DEBUG: Cannot remove layer {layer} — "
                    f"must remove top (layer {len(stack)-1}) first"
                )
                return False

            top_block = stack[-1]

            # ── Large block: remove both halves atomically ─────────
            if top_block.endswith('l'):
                if not span_to:
                    print(f"DEBUG: Large block removal requires span_to")
                    return False

                neighbor_stack = self.current_structure.get(span_to, [])

                # Neighbor must also have this block at same layer
                if len(neighbor_stack) == 0 or neighbor_stack[-1] != top_block:
                    print(
                        f"DEBUG: Span partner {span_to} does not have matching "
                        f"block '{top_block}' at top"
                    )
                    return False

                if len(neighbor_stack) - 1 != layer:
                    print(
                        f"DEBUG: Span partner {span_to} top layer "
                        f"{len(neighbor_stack)-1} != {layer}"
                    )
                    return False

                # Remove from both cells
                self.current_structure[position].pop()
                self.current_structure[span_to].pop()

                # Remove span record
                layer_spans = self.current_spans.get(layer, [])
                self.current_spans[layer] = [
                    (a, b) for a, b in layer_spans
                    if not (a == position and b == span_to)
                    and not (a == span_to and b == position)
                ]

                print(f"DEBUG: Removed large block '{top_block}' "
                    f"from {position}↔{span_to} layer {layer}")

            # ── Small block: remove single cell ───────────────────
            else:
                self.current_structure[position].pop()
                print(f"DEBUG: Removed small block '{top_block}' "
                    f"from {position} layer {layer}")

            return True

        except Exception as e:
            print(f"Error removing block: {e}")
            return False
    
    def _encode_block(self, block_name) -> str:
        """Convert block name to encoded format"""
        # block_name is already in correct format (e.g., 'os', 'bs', 'ys')
        if block_name in self.available_blocks:
            return block_name
        else:
            return "unknown"

 
    def add_to_conversation(self, speaker, message):
        """Add message to conversation history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append(f"[{timestamp}] {speaker}: {message}")
    
    def get_conversation_string(self, last_n_turns=None) -> str:
        """Get conversation as string with optional truncation"""
        if last_n_turns:
            return "\n".join(self.conversation_history[-last_n_turns:])
        return "\n".join(self.conversation_history)
    
    def get_progress_summary(self) -> Dict:
        """Get current progress summary"""
        return self.progress_tracker.get_progress_summary()
    
    def is_complete(self, threshold=0.95) -> bool:
        """Check if game is complete"""
        if not self.progress_tracker.progress_history:
            return False
        
        latest_progress = self.progress_tracker.progress_history[-1]['metrics']['overall_progress']
        # SAFER
        if not self.progress_tracker.progress_history:
            return False
        latest = self.progress_tracker.progress_history[-1]
        return latest.get('metrics', {}).get('overall_progress', 0) >= threshold
       
    
    def increment_turn(self):
        """Increment turn counter"""
        self.turn_count += 1

