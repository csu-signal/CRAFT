import random
import json
import copy
from collections import defaultdict
# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

COLORS     = ['g', 'b', 'r', 'y', 'o']
BLOCK_TYPES = [f"{c}{s}" for c in COLORS for s in ['s', 'l']]

ALL_COORDS     = [f"({i},{j})" for i in range(3) for j in range(3)]
REQUIRED_FULL  = [c for c in ALL_COORDS if c not in ["(1,1)", "(2,1)"]]
OPTIONAL       = ["(1,1)", "(2,1)"]

COLOR_NAMES = {'g':'green','b':'blue','r':'red','y':'yellow','o':'orange','n':'none'}

def coord_ij(coord):
    """'(i,j)' → (i, j) as ints"""
    return int(coord[1]), int(coord[3])

def ij_coord(i, j):
    return f"({i},{j})"

def orthogonal_neighbors(coord):
    """Return valid orthogonal neighbors on the 3x3 grid."""
    i, j = coord_ij(coord)
    result = []
    if j < 2: result.append(ij_coord(i, j+1))  # right
    if i < 2: result.append(ij_coord(i+1, j))  # down
    if j > 0: result.append(ij_coord(i, j-1))  # left
    if i > 0: result.append(ij_coord(i-1, j))  # up
    return result


def get_block_encoding_reference():
    return """BLOCK ENCODING:
Colors: g=green, b=blue, r=red, y=yellow, o=orange
Sizes:  s=small (1 cell), l=large (2 adjacent cells)
Examples: gs=green small, gl=green large, bs=blue small, yl=yellow large, ol=orange large"""

def get_coordinate_system_reference():
    return """COORDINATE SYSTEM (3x3 grid):
(0,0) (0,1) (0,2)   ← top row    (i=0)
(1,0) (1,1) (1,2)   ← middle row (i=1)
(2,0) (2,1) (2,2)   ← bottom row (i=2)
j=0 = left column, j=1 = center column, j=2 = right column"""


# ─────────────────────────────────────────
# LAYER TILING
# ─────────────────────────────────────────

def generate_layer_tiling(rng, positions_needed, prev_layer=None):
    """prev_layer: dict {coord: block} from previous layer, or None"""
    grid  = {}
    spans = []
    needed = set(positions_needed)
    order  = list(needed)
    rng.shuffle(order)

    for coord in order:
        if coord in grid:
            continue

        prev_block = (prev_layer or {}).get(coord)

        if rng.random() < 0.5:
            free_nbrs = [n for n in orthogonal_neighbors(coord)
                         if n in needed and n not in grid]
            if free_nbrs:
                neighbor   = rng.choice(free_nbrs)
                prev_nbr   = (prev_layer or {}).get(neighbor)
                # Pick color that doesn't repeat for EITHER cell
                color = rng.choice(COLORS)
                block = f"{color}l"
                # Retry once if both cells would repeat
                attempts = 0
                while attempts < 10 and (block == prev_block or block == prev_nbr):
                    color = rng.choice(COLORS)
                    block = f"{color}l"
                    attempts += 1
                grid[coord]    = block
                grid[neighbor] = block
                spans.append((coord, neighbor))
                continue

        # Small block — avoid repeat with previous layer
        color = rng.choice(COLORS)
        block = f"{color}s"
        attempts = 0
        while attempts < 10 and block == prev_block:
            color = rng.choice(COLORS)
            block = f"{color}s"
            attempts += 1
        grid[coord] = block

    return grid, spans

# ─────────────────────────────────────────
# STRUCTURE GENERATION
# ─────────────────────────────────────────

def generate_valid_structure(rng=None):
    """
    Generate a valid CRAFT target structure.

    Constraints:
    - REQUIRED_FULL positions (7): exactly 3 layers each
    - OPTIONAL positions (1,1) and (2,1): 0–2 layers each
    - Large blocks span two orthogonal cells on the SAME layer (domino rule)
    - No consecutive identical block at the same position across layers

    Returns:
        structure  : dict {coord: [block_layer0, block_layer1, ...]}
        spans      : dict {layer: [(coord_a, coord_b), ...]}
    """
    if rng is None:
        rng = random.Random()

    # Decide heights for optional positions
    opt_heights = {coord: rng.randint(0, 2) for coord in OPTIONAL}

    structure  = {coord: [] for coord in ALL_COORDS}
    all_spans  = {}
    prev_layer = None
    
    for layer in range(3):
        # Which positions need a block at this layer?
        positions_this_layer = list(REQUIRED_FULL)
        for coord in OPTIONAL:
            if opt_heights[coord] > layer:
                positions_this_layer.append(coord)

        layer_grid, spans = generate_layer_tiling(rng, positions_this_layer)
        all_spans[layer]  = spans

        # Write into structure, enforcing no consecutive identical block
        for coord in positions_this_layer:
            block = layer_grid.get(coord, f"{rng.choice(COLORS)}s")
            # If same as previous layer at this position, flip size
            # if structure[coord] and structure[coord][-1] == block:
            #     other_size = 'l' if block.endswith('s') else 's'
            #     block = block[0] + other_size
            structure[coord].append(block)
            prev_layer = layer_grid  

    return structure, all_spans

 

    

# ─────────────────────────────────────────
# VALIDATION — STRUCTURE
# ─────────────────────────────────────────

def validate_structure(structure, spans, strict=True):
    """
    Full structure validation including domino span consistency.
    Returns (is_valid, errors)
    """
    errors = []

    for coord in ALL_COORDS:
        if coord not in structure:
            errors.append(f"Missing coordinate: {coord}")
            continue

        stack = structure[coord]

        if len(stack) > 3:
            errors.append(f"{coord}: stack height {len(stack)} exceeds 3")

        for layer, block in enumerate(stack):
            if block not in BLOCK_TYPES:
                errors.append(f"{coord} layer {layer}: invalid block '{block}'")

        if strict and coord in REQUIRED_FULL and len(stack) != 3:
            errors.append(f"{coord}: required position has {len(stack)} blocks (need 3)")

        if strict and coord in OPTIONAL and len(stack) > 2:
            errors.append(f"{coord}: optional position has {len(stack)} blocks (max 2)")

    # Domino span validation
    for layer, layer_spans in spans.items():
        seen_in_span = set()

        for coord_a, coord_b in layer_spans:
            # Must be orthogonal neighbors
            if coord_b not in orthogonal_neighbors(coord_a):
                errors.append(f"Layer {layer}: span {coord_a}↔{coord_b} not orthogonal")

            # Both cells must have same block string
            stack_a = structure.get(coord_a, [])
            stack_b = structure.get(coord_b, [])
            if layer < len(stack_a) and layer < len(stack_b):
                if stack_a[layer] != stack_b[layer]:
                    errors.append(
                        f"Layer {layer}: span {coord_a}={stack_a[layer]} "
                        f"vs {coord_b}={stack_b[layer]} mismatch"
                    )
                if not stack_a[layer].endswith('l'):
                    errors.append(
                        f"Layer {layer}: span {coord_a}↔{coord_b} "
                        f"block '{stack_a[layer]}' is not large"
                    )
            seen_in_span.add(coord_a)
            seen_in_span.add(coord_b)

        # Every large block must be in a span
        for coord in ALL_COORDS:
            stack = structure.get(coord, [])
            if layer < len(stack) and stack[layer].endswith('l'):
                if coord not in seen_in_span:
                    errors.append(
                        f"Layer {layer}: large block at {coord} "
                        f"has no span partner"
                    )

    return len(errors) == 0, errors


# ─────────────────────────────────────────
# VALIDATION — SINGLE ACTION PLACEMENT
# ─────────────────────────────────────────

def validate_placement_action(action, structure, spans, target_structure, target_spans):
    """
    Validate a single block placement action against current state and target.

    action = {
        'coord'    : '(i,j)',
        'block'    : 'yl',       # e.g. yellow large
        'span_to'  : '(i,j2)'   # required if block is large, None if small
    }

    Checks:
    1. Coord is valid
    2. Stack not already at max height (3)
    3. Block type is valid
    4. Layer order respected (no gaps)
    5. If large: span_to provided, is orthogonal neighbor, 
                 neighbor at same layer is free, 
                 neighbor stack same height (same layer being built)
    6. If small: no span_to provided (or ignored)
    7. Action matches target at this coord+layer
    8. If large: action matches target span

    Returns (is_valid, reason, layer_placed)
    """
    coord   = action.get('coord')
    block   = action.get('block')
    span_to = action.get('span_to')  # None for small blocks

    # ── Basic checks ──────────────────────────────────────────
    if coord not in ALL_COORDS:
        return False, f"Invalid coordinate: {coord}", None

    if block not in BLOCK_TYPES:
        return False, f"Invalid block type: {block}", None

    current_stack = structure.get(coord, [])
    layer         = len(current_stack)  # next available layer

    if layer >= 3:
        return False, f"{coord} stack is full (height 3)", None

    # ── Large block checks ────────────────────────────────────
    if block.endswith('l'):
        if span_to is None:
            return False, f"Large block '{block}' requires span_to neighbor", None

        if span_to not in orthogonal_neighbors(coord):
            return False, (
                f"span_to {span_to} is not an orthogonal neighbor of {coord}"
            ), None

        neighbor_stack = structure.get(span_to, [])
        neighbor_layer = len(neighbor_stack)

        # Neighbor must be at same layer (both being placed together)
        if neighbor_layer != layer:
            return False, (
                f"Neighbor {span_to} is at layer {neighbor_layer}, "
                f"expected {layer} to match {coord}"
            ), None

        # Neighbor must not already have a block at this layer
        if neighbor_layer >= 3:
            return False, f"Neighbor {span_to} stack is already full", None

        # Check neighbor is also free in current spans at this layer
        layer_spans = spans.get(layer, [])
        neighbor_occupied = any(
            span_to in (a, b) for a, b in layer_spans
        )
        if neighbor_occupied:
            return False, (
                f"Neighbor {span_to} at layer {layer} already occupied by a span"
            ), None

    # ── Small block: must not claim a cell needed for a span ──
    else:
        # If span_to provided for small block, warn but allow
        span_to = None

    # ── Target match check ────────────────────────────────────
    target_stack = target_structure.get(coord, [])

    if layer >= len(target_stack):
        return False, (
            f"{coord} layer {layer} not in target "
            f"(target height={len(target_stack)})"
        ), None

    expected_block = target_stack[layer]

    if block != expected_block:
        return False, (
            f"{coord} layer {layer}: placed '{block}' "
            f"but target expects '{expected_block}'"
        ), None

    # ── If large: check target span matches intended span ─────
    if block.endswith('l') and span_to is not None:
        target_layer_spans = target_spans.get(layer, [])
        intended_span = tuple(sorted([coord, span_to]))
        target_span_tuples = [
            tuple(sorted([a, b])) for a, b in target_layer_spans
        ]
        if intended_span not in target_span_tuples:
            return False, (
                f"Span {coord}↔{span_to} at layer {layer} "
                f"does not match any target span"
            ), None

    return True, "ok", layer


# ─────────────────────────────────────────
# APPLY ACTION (update state)
# ─────────────────────────────────────────

def apply_placement_action(action, structure, spans):
    """
    Apply a validated placement action to structure and spans (in place).
    Call validate_placement_action first.
    """
    coord   = action['coord']
    block   = action['block']
    span_to = action.get('span_to')
    layer   = len(structure[coord])

    structure[coord].append(block)

    if block.endswith('l') and span_to:
        structure[span_to].append(block)
        spans.setdefault(layer, []).append((coord, span_to))


# ─────────────────────────────────────────
# DIRECTOR VIEWS
# ─────────────────────────────────────────

# def get_director_views(structure):
#     def cell(coord, layer):
#         stack = structure.get(coord, [])
#         if layer >= len(stack):
#             return {"color": "none", "size": 1}
#         block = stack[layer]
#         return {
#             "color": COLOR_NAMES.get(block[0], "none"),
#             "size" : 2 if block.endswith('l') else 1
#         }

#     return {
#         "D1": {f"row_{l}": [cell("(0,0)",l), cell("(1,0)",l), cell("(2,0)",l)] for l in range(3)},
#         "D2": {f"row_{l}": [cell("(0,0)",l), cell("(0,1)",l), cell("(0,2)",l)] for l in range(3)},
#         "D3": {f"row_{l}": [cell("(0,2)",l), cell("(1,2)",l), cell("(2,2)",l)] for l in range(3)},
#     }


def get_director_views(structure, spans=None):
    director_coords = {
        "D1": ["(0,0)", "(1,0)", "(2,0)"],
        "D2": ["(0,0)", "(0,1)", "(0,2)"],
        "D3": ["(0,2)", "(1,2)", "(2,2)"]
    }

    def cell(coord, layer, visible_coords):
        stack = structure.get(coord, [])
        if layer >= len(stack):
            return {"color": "none", "size": 1}
        block = stack[layer]
        color = COLOR_NAMES.get(block[0], "none")

        if block.endswith('l') and spans:
            layer_spans = spans.get(layer, [])
            partner = next(
                (b if a == coord else a
                 for a, b in layer_spans if coord in (a, b)),
                None
            )
            # Large only if partner also visible — otherwise appears small
            size = 2 if (partner and partner in visible_coords) else 1
        else:
            size = 1

        return {"color": color, "size": size}

    views = {}
    for did, coords in director_coords.items():
        views[did] = {
            f"row_{l}": [cell(c, l, coords) for c in coords]
            for l in range(3)
        }
    return views

# ─────────────────────────────────────────
# 3D ASCII VISUALIZATION
# ─────────────────────────────────────────

def block_str(block):
    if not block or block == 'ns':
        return '   '
    return f"{block[0].upper()}{'L' if block.endswith('l') else 's'} "

def print_3d_structure(structure, spans=None, title="Structure"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"\n  Grid view (layers shown top→bottom within each row):")
    print(f"         j=0      j=1      j=2")
    print(f"       ┌────────┬────────┬────────┐")

    for i in range(3):
        for layer in range(2, -1, -1):
            row_parts = []
            for j in range(3):
                coord = ij_coord(i, j)
                stack = structure.get(coord, [])
                if layer < len(stack):
                    row_parts.append(f" {block_str(stack[layer]):3s}   ")
                else:
                    row_parts.append("        ")
            suffix = "  ← top" if layer == 2 else ("  ← base" if layer == 0 else "")
            label  = f"  i={i}" if layer == 2 else "      "
            print(f"  {label} │{'│'.join(row_parts)}│{suffix}")

        print(f"       ├────────┼────────┼────────┤" if i < 2
              else f"       └────────┴────────┴────────┘")

    # Span info
    if spans:
        print(f"\n  Large block spans:")
        for layer, layer_spans in sorted(spans.items()):
            if layer_spans:
                span_strs = [f"{a}↔{b}" for a, b in layer_spans]
                print(f"    Layer {layer}: {', '.join(span_strs)}")

    total  = sum(len(s) for s in structure.values())
    filled = sum(1 for s in structure.values() if s)
    print(f"\n  Total blocks: {total} | Filled positions: {filled}/9")

    print(f"\n  Stack detail:")
    for i in range(3):
        for j in range(3):
            coord = ij_coord(i, j)
            stack = structure.get(coord, [])
            if stack:
                detail = " → ".join(
                    f"{block_str(b).strip()}[{'base' if l==0 else 'mid' if l==1 else 'top'}]"
                    for l, b in enumerate(stack)
                )
                print(f"    {coord}: {detail}")
            else:
                print(f"    {coord}: (empty)")


# ─────────────────────────────────────────
# PARTIAL STRUCTURE
# ─────────────────────────────────────────

def generate_partial_structure(target, target_spans, n_pre_placed=None, rng=None):
    """
    Pre-place blocks respecting layer order AND domino integrity
    (never place one half of a large block without the other).
    """
    if rng is None:
        rng = random.Random()

    # Build ordered placement events — large blocks count as one event (two cells)
    placement_events = []
    processed = set()

    for layer in range(3):
        layer_spans = target_spans.get(layer, [])
        span_cells  = {c for a, b in layer_spans for c in (a, b)}

        # Large block events (domino pairs)
        for coord_a, coord_b in layer_spans:
            key = tuple(sorted([coord_a, coord_b, str(layer)]))
            if key not in processed:
                placement_events.append({
                    'type'   : 'large',
                    'coords' : (coord_a, coord_b),
                    'block'  : target[coord_a][layer],
                    'layer'  : layer
                })
                processed.add(key)

        # Small block events
        for coord in ALL_COORDS:
            t_stack = target.get(coord, [])
            if layer < len(t_stack) and coord not in span_cells:
                placement_events.append({
                    'type'  : 'small',
                    'coord' : coord,
                    'block' : t_stack[layer],
                    'layer' : layer
                })

    total = len(placement_events)
    if n_pre_placed is None:
        n_pre_placed = rng.randint(int(total * 0.6), int(total * 0.7))
    n_pre_placed = min(n_pre_placed, total)

    # Greedily pick events whose layer precondition is met
    partial       = {coord: [] for coord in ALL_COORDS}
    partial_spans = {}
    remaining     = list(placement_events)

    for _ in range(n_pre_placed):
        available = []
        for event in remaining:
            if event['type'] == 'small':
                if len(partial[event['coord']]) == event['layer']:
                    available.append(event)
            else:
                ca, cb = event['coords']
                if len(partial[ca]) == event['layer'] and \
                   len(partial[cb]) == event['layer']:
                    available.append(event)
        if not available:
            break

        pick = rng.choice(available)
        remaining.remove(pick)

        if pick['type'] == 'small':
            partial[pick['coord']].append(pick['block'])
        else:
            ca, cb = pick['coords']
            partial[ca].append(pick['block'])
            partial[cb].append(pick['block'])
            partial_spans.setdefault(pick['layer'], []).append((ca, cb))

    return partial, partial_spans, remaining


# ─────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────

def generate_dataset(n=100, seed=42):
    rng        = random.Random(seed)
    structures = []

    for i in range(n):
        structure, spans = generate_valid_structure(rng=rng)
        is_valid, errors = validate_structure(structure, spans, strict=True)

        if not is_valid:
            print(f"  WARNING structure {i}: {errors}")
            continue

        views  = get_director_views(structure)
        total  = sum(len(s) for s in structure.values())
        filled = sum(1 for s in structure.values() if s)
        complexity = 'simple' if total <= 22 else 'medium' if total <= 24 else 'complex'

        structures.append({
            'id'            : f"structure_{i+1:03d}",
             'complexity': complexity,   
            'structure'     : structure,
            'spans'         : {str(k): v for k, v in spans.items()},
            'director_views': views,
            'metadata'      : {
                'total_blocks'    : total,
                'filled_positions': filled,
                'optional_heights': {c: len(structure[c]) for c in OPTIONAL}
            }
        })

    return structures




# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=== CRAFT Structure Generator v2 ===\n")
    rng = random.Random(42)

    for trial in range(2):
        structure, spans = generate_valid_structure(rng=rng)
        is_valid, errors = validate_structure(structure, spans, strict=True)

        print_3d_structure(structure, spans, title=f"Example {trial+1}")
        print(f"\n  Validation: {'✓ passed' if is_valid else '✗ FAILED'}")
        for e in errors:
            print(f"    - {e}")

        views = get_director_views(structure)
        print(f"\n  Director Views:")
        for d, dv in views.items():
            print(f"    {d}:")
            for row, cells in dv.items():
                cell_str = " | ".join(
                    f"{c['color'][0].upper()}{'L' if c['size']==2 else 's'}"
                    for c in cells
                )
                print(f"      {row}: [{cell_str}]")

        # Test placement validation
        print(f"\n  Testing validate_placement_action:")
        partial, partial_spans, remaining = generate_partial_structure(
            structure, spans, rng=rng
        )
        print_3d_structure(partial, partial_spans, title=f"Example {trial+1} Partial")

        if remaining:
            # Test valid action
            next_event = remaining[0]
            if next_event['type'] == 'small':
                action = {'coord': next_event['coord'], 'block': next_event['block'], 'span_to': None}
            else:
                ca, cb = next_event['coords']
                action = {'coord': ca, 'block': next_event['block'], 'span_to': cb}

            ok, reason, layer = validate_placement_action(
                action, partial, partial_spans, structure, spans
            )
            print(f"\n  Next valid action: {action}")
            print(f"  Result: {'✓' if ok else '✗'} {reason} (layer={layer})")

            # Test invalid action — wrong block
            bad_action = dict(action)
            bad_action['block'] = 'rs' if action['block'] != 'rs' else 'gl'
            ok2, reason2, _ = validate_placement_action(
                bad_action, partial, partial_spans, structure, spans
            )
            print(f"\n  Bad action (wrong block): {bad_action}")
            print(f"  Result: {'✓' if ok2 else '✗'} {reason2}")

    # Full dataset
    print(f"\n{'='*60}")
    n = 20
    print(f"Generating dataset of {n} structures...")
    dataset = generate_dataset(n=n, seed=42)
    print(f"Generated {len(dataset)} valid structures")

    with open("structures_dataset_v2.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Saved to structures_dataset_v2.json")