#!/usr/bin/env python3
"""
craft_full_info_env.py
======================

Dependency-free CRAFT game engine for the *no information separation*
condition: one Builder that sees all three wall views directly.

Contents
--------
  Board                   stacks + domino (span) bookkeeping, strict physics
  Board.from_partial / start_board
                          part-complete starting states (first layer(s), a wall)
  project_view            3D board -> one Director's 2D wall view (paper App. B.4)
  compute_metrics         IoU / CP / PA / OP (paper App. B.5), visible-cell
                          variants, and view-consistency
  enumerate_oracle_moves  verified forward-progress moves (paper App. B.1)
  infer_structure_from_views
                          the best structure recoverable from the 3 views alone
  OraclePolicy / ViewsPolicy
                          scripted, LLM-free builders for smoke tests and ceilings

Physics (stricter than run_single_builder._place_block, closer to the CRAFT
engine described in the paper):
  * PLACE layer must equal the current stack height (place on top only).
  * Height cap 3.
  * Large blocks need span_to, an orthogonal neighbour of equal height.
  * REMOVE only the top block; a large removal must name the exact partner
    cell of that domino (tracked, not guessed from matching colours).

Notes on the shipped data (structures_dataset_20.json)
------------------------------------------------------
  * `director_views` stores size=1 for every cell (540/540), although the
    paper says a domino whose two cells are both on a wall shows as size 2.
    Views are therefore recomputed from structure+spans by default.
  * 25/131 dominoes touch an interior cell (1,1)/(2,1). The CRAFT builder
    prompt's rule "a large block visible to any director CANNOT span (1,1)
    or (2,1)" does not hold for this dataset, so it is not repeated here.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
CELLS: List[str] = [f"({r},{c})" for r in range(3) for c in range(3)]
WALLS: Dict[str, List[str]] = {           # left-to-right from each Director's seat
    "D1": ["(0,0)", "(1,0)", "(2,0)"],    # left wall
    "D2": ["(0,0)", "(0,1)", "(0,2)"],    # far wall
    "D3": ["(0,2)", "(1,2)", "(2,2)"],    # right wall
}
DIRECTORS = ("D1", "D2", "D3")
VISIBLE_CELLS: List[str] = [c for c in CELLS if any(c in w for w in WALLS.values())]
INTERIOR_CELLS: List[str] = [c for c in CELLS if c not in VISIBLE_CELLS]   # (1,1), (2,1)

MAX_HEIGHT = 3
N_LAYERS = 3
COLORS = {"g": "green", "b": "blue", "r": "red", "y": "yellow", "o": "orange"}
COLOR_TO_CODE = {v: k for k, v in COLORS.items()}
BLOCKS = [c + s for c in "gbryo" for s in "sl"]   # gs gl bs bl ...

# Part-complete starting states.  Layer modes copy the bottom N layers of every
# stack; wall modes copy whole stacks on one Director's wall.
PARTIAL_LAYER_MODES: Dict[str, int] = {"firstLayer": 1, "firstTwoLayers": 2}
PARTIAL_WALL_MODES: Dict[str, str] = {"D1Wall": "D1", "D2Wall": "D2", "D3Wall": "D3"}
PARTIAL_MODES: List[str] = ["none", *PARTIAL_LAYER_MODES, *PARTIAL_WALL_MODES]

_POS_RE = re.compile(r"^\(?\s*(\d)\s*,\s*(\d)\s*\)?$")


def norm_pos(s) -> Optional[str]:
    """'( 0, 1 )' / '0,1' / (0, 1) -> '(0,1)'; None if not a valid grid cell."""
    if s is None:
        return None
    if isinstance(s, (tuple, list)) and len(s) == 2:
        s = f"({s[0]},{s[1]})"
    m = _POS_RE.match(str(s).strip())
    if not m:
        return None
    r, c = int(m.group(1)), int(m.group(2))
    if r > 2 or c > 2:
        return None
    return f"({r},{c})"


def _rc(pos: str) -> Tuple[int, int]:
    r, c = pos.strip("()").split(",")
    return int(r), int(c)


def adjacent(a: str, b: str) -> bool:
    (r1, c1), (r2, c2) = _rc(a), _rc(b)
    return abs(r1 - r2) + abs(c1 - c2) == 1


def is_large(block: str) -> bool:
    return len(block) == 2 and block[1] == "l"


# --------------------------------------------------------------------------- #
# Board
# --------------------------------------------------------------------------- #
@dataclass
class MoveResult:
    ok: bool
    error: Optional[str] = None
    error_type: Optional[str] = None      # layer | span | height | empty | block | position | other


@dataclass
class Board:
    stacks: Dict[str, List[str]] = field(default_factory=lambda: {c: [] for c in CELLS})
    # layer -> list of (a, b) domino endpoint pairs
    spans: Dict[int, List[Tuple[str, str]]] = field(default_factory=dict)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_structure(cls, entry: Dict) -> "Board":
        """Build a Board from a structures_dataset entry (structure + spans)."""
        stacks = {c: list(entry["structure"].get(c, [])) for c in CELLS}
        spans: Dict[int, List[Tuple[str, str]]] = {}
        for layer, pairs in (entry.get("spans") or {}).items():
            spans[int(layer)] = [(norm_pos(a), norm_pos(b)) for a, b in pairs]
        return cls(stacks=stacks, spans=spans)

    @classmethod
    def from_partial(cls, target: "Board",
                     part_type: Union[None, str, Iterable[str]] = "none") -> "Board":
        """
        A part-complete starting board: a physically valid, prefix-correct
        subset of `target`.

        part_type: "none" | "firstLayer" | "firstTwoLayers" | "D1Wall" |
                   "D2Wall" | "D3Wall", or a list of these (union, e.g.
                   ["D1Wall", "D3Wall"]).

        A domino is copied only if BOTH halves fall inside the selection.
        If a wall cell's domino spans off the wall (e.g. (1,0)-(1,1)), that
        stack is cut just below the domino, since half a domino can't exist
        and nothing can rest on a missing block. Layer modes never split a
        domino, because both halves of a domino share the same layer.
        """
        keep = _close_slots(target, partial_slots(target, part_type))
        stacks = {c: [blk for k, blk in enumerate(target.stacks[c]) if (c, k) in keep]
                  for c in CELLS}                  # fresh lists: no aliasing with target
        spans: Dict[int, List[Tuple[str, str]]] = {}
        for layer, pairs in target.spans.items():
            kept = [(a, b) for a, b in pairs if (a, layer) in keep and (b, layer) in keep]
            if kept:
                spans[layer] = kept
        return cls(stacks=stacks, spans=spans)

    def copy(self) -> "Board":
        return Board(stacks=copy.deepcopy(self.stacks), spans=copy.deepcopy(self.spans))

    def to_json(self) -> Dict:
        return {
            "stacks": self.stacks,
            "spans": {str(k): [list(p) for p in v] for k, v in sorted(self.spans.items()) if v},
        }

    # ---- queries ---------------------------------------------------------
    def height(self, c: str) -> int:
        return len(self.stacks[c])

    def partner(self, c: str, layer: int) -> Optional[str]:
        for a, b in self.spans.get(layer, []):
            if a == c:
                return b
            if b == c:
                return a
        return None

    def n_blocks(self) -> int:
        """Physical block count (a domino counts once)."""
        entries = sum(len(s) for s in self.stacks.values())
        return entries - sum(len(v) for v in self.spans.values())

    # ---- moves -----------------------------------------------------------
    def apply(self, move: Dict) -> MoveResult:
        action = move.get("action")
        if action == "place":
            return self.place(move.get("block"), move.get("position"), move.get("layer"), move.get("span_to"))
        if action == "remove":
            return self.remove(move.get("position"), move.get("layer"), move.get("span_to"))
        return MoveResult(False, f"Unsupported action '{action}'", "other")

    def place(self, block, position, layer, span_to=None) -> MoveResult:
        pos = norm_pos(position)
        if pos is None:
            return MoveResult(False, f"Invalid position '{position}'", "position")
        if block not in BLOCKS:
            return MoveResult(False, f"Unknown block code '{block}'", "block")
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            return MoveResult(False, f"Invalid layer '{layer}'", "layer")

        h = self.height(pos)
        if h >= MAX_HEIGHT:
            return MoveResult(False, f"Stack at {pos} is full ({MAX_HEIGHT} blocks)", "height")
        if layer != h:
            return MoveResult(False, f"Cannot place at layer {layer} on {pos}: stack has {h} "
                                     f"block(s), next free layer is {h}", "layer")

        if not is_large(block):
            self.stacks[pos].append(block)
            return MoveResult(True)

        partner = norm_pos(span_to)
        if partner is None:
            return MoveResult(False, f"Large block '{block}' requires a valid span_to", "span")
        if partner == pos or not adjacent(pos, partner):
            return MoveResult(False, f"span_to {partner} is not an orthogonal neighbour of {pos}", "span")
        ph = self.height(partner)
        if ph != h:
            return MoveResult(False, f"Large block needs equal stack heights: {pos} has {h}, "
                                     f"{partner} has {ph} (span/layer mismatch)", "span")
        self.stacks[pos].append(block)
        self.stacks[partner].append(block)
        self.spans.setdefault(layer, []).append((pos, partner))
        return MoveResult(True)

    def remove(self, position, layer, span_to=None) -> MoveResult:
        pos = norm_pos(position)
        if pos is None:
            return MoveResult(False, f"Invalid position '{position}'", "position")
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            return MoveResult(False, f"Invalid layer '{layer}'", "layer")
        h = self.height(pos)
        if h == 0:
            return MoveResult(False, f"Cannot remove: stack at {pos} is empty", "empty")
        if layer != h - 1:
            return MoveResult(False, f"Cannot remove layer {layer} at {pos}: must remove top block "
                                     f"first (layer {h - 1})", "layer")
        top = self.stacks[pos][-1]
        if not is_large(top):
            self.stacks[pos].pop()
            return MoveResult(True)

        true_partner = self.partner(pos, layer)
        want = norm_pos(span_to)
        if want is None:
            return MoveResult(False, f"Removing large block '{top}' at {pos} requires span_to "
                                     f"(its other half is at {true_partner})", "span")
        if want != true_partner:
            return MoveResult(False, f"span_to {want} is wrong: the '{top}' at {pos} layer {layer} "
                                     f"spans to {true_partner}", "span")
        if self.height(true_partner) - 1 != layer:
            return MoveResult(False, f"Cannot remove domino {pos}-{true_partner}: {true_partner} has "
                                     f"blocks stacked on top of it", "layer")
        self.stacks[pos].pop()
        self.stacks[true_partner].pop()
        self.spans[layer] = [p for p in self.spans[layer] if set(p) != {pos, true_partner}]
        return MoveResult(True)


# --------------------------------------------------------------------------- #
# Part-complete starting states
# --------------------------------------------------------------------------- #
def partial_slots(target: Board, part_type: Union[None, str, Iterable[str]]) -> Set[Tuple[str, int]]:
    """Raw (cell, layer) slots selected by part_type, before validity closure."""
    if part_type is None:
        return set()
    modes = [part_type] if isinstance(part_type, str) else list(part_type)
    slots: Set[Tuple[str, int]] = set()
    for mode in modes:
        if mode in ("", "none"):
            continue
        if mode in PARTIAL_LAYER_MODES:
            n = PARTIAL_LAYER_MODES[mode]
            slots |= {(c, k) for c in CELLS for k in range(min(n, target.height(c)))}
        elif mode in PARTIAL_WALL_MODES:
            for c in WALLS[PARTIAL_WALL_MODES[mode]]:
                slots |= {(c, k) for k in range(target.height(c))}
        else:
            raise ValueError(f"Unknown part_type '{mode}'; expected one of {PARTIAL_MODES}")
    return slots


def _close_slots(target: Board, slots: Set[Tuple[str, int]]) -> Set[Tuple[str, int]]:
    """
    Shrink `slots` until it is a buildable board: every kept block rests on a
    kept block (or the floor), and every kept domino half has its partner kept.
    Iterates to a fixed point, because cutting one stack can orphan a domino
    higher up in a neighbouring stack.
    """
    keep = set(slots)
    changed = True
    while changed:
        changed = False
        for c, k in sorted(keep):
            if (c, k) not in keep:
                continue
            drop = k > 0 and (c, k - 1) not in keep
            if not drop and is_large(target.stacks[c][k]):
                p = target.partner(c, k)
                drop = p is None or (p, k) not in keep
            if drop:
                keep.discard((c, k))
                changed = True
    return keep


def start_board(entry: Dict, part_type: Union[None, str, Iterable[str]] = "none") -> Board:
    """Starting board for a dataset entry ("none" = empty board)."""
    return Board.from_partial(Board.from_structure(entry), part_type)


def partial_report(target: Board, start: Board,
                   part_type: Union[None, str, Iterable[str]]) -> Dict:
    """What a part-complete start contains and which selected slots were dropped."""
    selected = partial_slots(target, part_type)
    kept = {(c, k) for c in CELLS for k in range(start.height(c))}
    return {
        "part_type": part_type,
        "start_blocks": start.n_blocks(),
        "target_blocks": target.n_blocks(),
        "moves_remaining": target.n_blocks() - start.n_blocks(),
        "dropped_slots": sorted(selected - kept),   # cut because a domino left the selection
    }


def check_structure(entry: Dict) -> List[str]:
    """Return a list of consistency problems in a dataset entry (empty = OK)."""
    problems = []
    b = Board.from_structure(entry)
    covered = set()
    for layer, pairs in b.spans.items():
        for a, c in pairs:
            if not adjacent(a, c):
                problems.append(f"layer {layer}: span {a}-{c} not adjacent")
            for x in (a, c):
                if layer >= b.height(x) or not is_large(b.stacks[x][layer]):
                    problems.append(f"layer {layer}: span endpoint {x} has no large block")
                covered.add((x, layer))
            if layer < b.height(a) and layer < b.height(c) and b.stacks[a][layer] != b.stacks[c][layer]:
                problems.append(f"layer {layer}: span {a}-{c} colours differ")
    for c, stack in b.stacks.items():
        if len(stack) > MAX_HEIGHT:
            problems.append(f"{c}: height {len(stack)} > {MAX_HEIGHT}")
        for k, blk in enumerate(stack):
            if is_large(blk) and (c, k) not in covered:
                problems.append(f"{c} layer {k}: large block without a span record")
    return problems


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def project_view(board: Board, director: str) -> Dict[str, List[Dict]]:
    """
    Paper App. B.4: row_k = layer k, cells left-to-right from the Director's
    seat; a large block is size 2 only when BOTH of its cells are on this wall.
    Empty cells are {"color": "none", "size": 0}.
    """
    wall = WALLS[director]
    view = {}
    for k in range(N_LAYERS):
        row = []
        for c in wall:
            stack = board.stacks[c]
            if k >= len(stack):
                row.append({"color": "none", "size": 0})
                continue
            blk = stack[k]
            size = 1
            if is_large(blk):
                p = board.partner(c, k)
                if p is not None and p in wall:
                    size = 2
            row.append({"color": COLORS.get(blk[0], "unknown"), "size": size})
        view[f"row_{k}"] = row
    return view


def project_all_views(board: Board) -> Dict[str, Dict]:
    return {d: project_view(board, d) for d in DIRECTORS}


def _cell_eq(a: Dict, b: Dict) -> bool:
    if a["color"] == "none" or b["color"] == "none":
        return a["color"] == b["color"]
    return a["color"] == b["color"] and int(a["size"]) == int(b["size"])


def view_match(board: Board, target_views: Dict[str, Dict]) -> Dict:
    """Fraction of the 81 (director, layer, cell) view entries that match."""
    per_dir = {}
    total = hits = 0
    for d in DIRECTORS:
        cur = project_view(board, d)
        n = m = 0
        for k in range(N_LAYERS):
            for a, b in zip(cur[f"row_{k}"], target_views[d][f"row_{k}"]):
                n += 1
                m += int(_cell_eq(a, b))
        per_dir[d] = {"match": m / n, "exact": m == n}
        total += n
        hits += m
    return {"view_match": hits / total,
            "views_exact": all(v["exact"] for v in per_dir.values()),
            "per_director": per_dir}


def infer_structure_from_views(views: Dict[str, Dict]) -> Board:
    """
    The structure a perfect reasoner can recover from the three views alone:
    wall cells get the colours shown; a domino is placed only where some view
    shows it as size 2 (both halves on that wall); everything else is a small
    block. Interior cells stay empty. This is a view-consistent build but not
    necessarily the true target (interior-spanning dominoes are unrecoverable).
    """
    colors: Dict[Tuple[str, int], str] = {}
    pairs: Dict[int, List[Tuple[str, str]]] = {}
    for d in DIRECTORS:
        wall = WALLS[d]
        for k in range(N_LAYERS):
            row = views[d][f"row_{k}"]
            i = 0
            while i < 3:
                cell = row[i]
                if cell["color"] != "none":
                    colors.setdefault((wall[i], k), cell["color"])
                if (int(cell.get("size", 1)) == 2 and i + 1 < 3
                        and int(row[i + 1].get("size", 1)) == 2
                        and row[i + 1]["color"] == cell["color"]):
                    pair = (wall[i], wall[i + 1])
                    colors.setdefault((wall[i + 1], k), row[i + 1]["color"])
                    if all(set(pair) != set(p) for p in pairs.get(k, [])):
                        pairs.setdefault(k, []).append(pair)
                    i += 2
                    continue
                i += 1
    b = Board()
    in_pair = {(x, k) for k, ps in pairs.items() for p in ps for x in p}
    for c in VISIBLE_CELLS:
        for k in range(N_LAYERS):
            col = colors.get((c, k))
            if col is None:
                break
            code = COLOR_TO_CODE[col] + ("l" if (c, k) in in_pair else "s")
            b.stacks[c].append(code)
    b.spans = {k: list(v) for k, v in pairs.items()}
    return b


# --------------------------------------------------------------------------- #
# Metrics (paper App. B.5) + visible-cell variants
# --------------------------------------------------------------------------- #
def _iou(cur: Board, tgt: Board, cells) -> float:
    inter = union = 0
    for c in cells:
        a, b = set(cur.stacks[c]), set(tgt.stacks[c])
        inter += len(a & b)
        union += len(a | b)
    return inter / union if union else 0.0


def _cp(cur: Board, tgt: Board, cells) -> float:
    total = hit = 0
    for c in cells:
        t, s = tgt.stacks[c], cur.stacks[c]
        total += len(t)
        hit += sum(1 for k in range(len(t)) if k < len(s) and s[k] == t[k])
    return hit / total if total else 0.0


def _pa(cur: Board, tgt: Board, cells) -> float:
    cells = list(cells)
    return sum(set(cur.stacks[c]) == set(tgt.stacks[c]) for c in cells) / len(cells)


def compute_metrics(cur: Board, tgt: Board, target_views: Optional[Dict] = None) -> Dict:
    iou, cp, pa = _iou(cur, tgt, CELLS), _cp(cur, tgt, CELLS), _pa(cur, tgt, CELLS)
    viou, vcp, vpa = (_iou(cur, tgt, VISIBLE_CELLS), _cp(cur, tgt, VISIBLE_CELLS),
                      _pa(cur, tgt, VISIBLE_CELLS))
    out = {
        "iou": iou, "completion": cp, "position_accuracy": pa, "progress": (iou + cp + pa) / 3,
        "visible_iou": viou, "visible_completion": vcp, "visible_position_accuracy": vpa,
        "visible_progress": (viou + vcp + vpa) / 3,
        "n_blocks": cur.n_blocks(),
    }
    if target_views is not None:
        vm = view_match(cur, target_views)
        out["view_match"] = vm["view_match"]
        out["views_exact"] = vm["views_exact"]
        out["view_match_by_director"] = {d: v["match"] for d, v in vm["per_director"].items()}
    return out


def min_moves(board: Board, start: Optional[Board] = None) -> int:
    """
    Moves needed to build `board` (a domino is one move): from empty by
    default, or from a prefix-correct `start` such as Board.from_partial().
    """
    return board.n_blocks() - (start.n_blocks() if start is not None else 0)


# --------------------------------------------------------------------------- #
# Oracle (paper App. B.1)
# --------------------------------------------------------------------------- #
def _is_prefix(stack: List[str], target: List[str]) -> bool:
    return len(stack) <= len(target) and stack == target[:len(stack)]


def enumerate_oracle_moves(board: Board, target: Board) -> List[Dict]:
    """
    All verified forward-progress moves from `board` toward `target`:
      (i)   placement of the next required block on a prefix-correct stack
            (dominoes only if the partner is also prefix-correct at the same height);
      (ii)  removal of the top block of a stack taller than its target;
      (iii) removal of the top block of a stack containing a wrong block.
    Each candidate is verified by simulating it on a copy of the board.
    """
    cands: List[Dict] = []
    seen = set()

    for c in CELLS:
        cur, tgt = board.stacks[c], target.stacks[c]
        h = len(cur)
        if _is_prefix(cur, tgt):
            if h == len(tgt):
                continue
            blk = tgt[h]
            if is_large(blk):
                p = target.partner(c, h)
                if p is None or not _is_prefix(board.stacks[p], target.stacks[p]) or board.height(p) != h:
                    continue
                key = ("place", blk, frozenset((c, p)), h)
                move = {"action": "place", "block": blk, "position": c, "layer": h, "span_to": p}
            else:
                key = ("place", blk, frozenset((c,)), h)
                move = {"action": "place", "block": blk, "position": c, "layer": h, "span_to": None}
        else:
            top = cur[-1]
            if is_large(top):
                p = board.partner(c, h - 1)
                if p is None:
                    continue
                key = ("remove", frozenset((c, p)), h - 1)
                move = {"action": "remove", "block": top, "position": c, "layer": h - 1, "span_to": p}
            else:
                key = ("remove", frozenset((c,)), h - 1)
                move = {"action": "remove", "block": top, "position": c, "layer": h - 1, "span_to": None}
        if key in seen:
            continue
        sim = board.copy()
        if not sim.apply(move).ok:
            continue
        if move["action"] == "place":
            cells = [c] + ([move["span_to"]] if move["span_to"] else [])
            if not all(_is_prefix(sim.stacks[x], target.stacks[x]) for x in cells):
                continue
        seen.add(key)
        cands.append(move)
    return cands


def sample_oracle_moves(cands: List[Dict], n: int, seed_key: str) -> List[Dict]:
    """Deterministic subsample of up to n candidates (seeded by structure+turn)."""
    if len(cands) <= n:
        return list(cands)
    rng = random.Random(seed_key)
    return rng.sample(cands, n)


def _endpoints(m: Dict) -> frozenset:
    pts = {norm_pos(m.get("position"))}
    if m.get("span_to"):
        pts.add(norm_pos(m.get("span_to")))
    return frozenset(pts)


def move_matches(move: Dict, cand: Dict) -> bool:
    """Endpoint-order-insensitive match; block is ignored for removals."""
    if move.get("action") != cand.get("action"):
        return False
    try:
        if int(move.get("layer")) != int(cand.get("layer")):
            return False
    except (TypeError, ValueError):
        return False
    if _endpoints(move) != _endpoints(cand):
        return False
    if move["action"] == "place" and move.get("block") != cand.get("block"):
        return False
    return True


def classify_turn(move: Dict, result: Optional[MoveResult], cands: List[Dict]) -> str:
    """Paper App. C failure taxonomy, applied to turns with >=1 oracle move."""
    if not cands:
        return "no-oracle"
    action = move.get("action")
    if action not in ("place", "remove"):
        return action or "other"                       # done / parse_error
    if result is not None and not result.ok:
        et = result.error_type or ""
        if et == "layer":
            return "engine-layer"
        if et == "span":
            return "engine-span"
        return "engine-other"
    if any(move_matches(move, c) for c in cands):
        return "correct"
    pos = norm_pos(move.get("position"))
    same_pos = [c for c in cands if c["action"] == action and pos in _endpoints(c)]
    if not same_pos:
        return "wrong-position"
    if action == "place" and all(c.get("block") != move.get("block") for c in same_pos):
        return "wrong-color"
    return "wrong-span"


# --------------------------------------------------------------------------- #
# Scripted builders (no LLM) — smoke tests and information ceilings
# --------------------------------------------------------------------------- #
class OraclePolicy:
    """Always takes the first oracle move toward the TRUE target (upper bound)."""
    name = "scripted-oracle"

    def decide(self, ctx: Dict) -> Dict:
        cands = enumerate_oracle_moves(ctx["board"], ctx["target"])
        if not cands:
            return {"move": {"action": "done", "reason": "oracle: nothing left"}}
        return {"move": dict(cands[0], confirmation="oracle")}


class ViewsPolicy:
    """
    Builds infer_structure_from_views(views): the best a perfect reasoner can
    do from the three views alone. Uses only the views, never the target.
    """
    name = "scripted-views"

    def decide(self, ctx: Dict) -> Dict:
        inferred = infer_structure_from_views(ctx["views"])
        cands = enumerate_oracle_moves(ctx["board"], inferred)
        if not cands:
            return {"move": {"action": "done", "reason": "views satisfied"}}
        return {"move": dict(cands[0], confirmation="views")}


def views_ceiling(entry: Dict, views: Dict) -> Dict:
    """Metrics of the view-inferred build against the true target."""
    target = Board.from_structure(entry)
    inferred = infer_structure_from_views(views)
    m = compute_metrics(inferred, target, views)
    m["min_moves"] = min_moves(inferred)
    return m


def target_views(entry: Dict, source: str = "recompute") -> Dict[str, Dict]:
    """'recompute' (default) projects from structure+spans; 'dataset' uses the stored views."""
    if source == "dataset":
        return entry["director_views"]
    return project_all_views(Board.from_structure(entry))


def load_structures(path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)