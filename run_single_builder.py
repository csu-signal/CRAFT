from enum import Enum
import json
from pathlib import Path
import re
import os
from typing import Dict
from openai import OpenAI
from agents.environment import (
    EnhancedGameState, get_oracle_moves
)
from structure_generator_v2 import (
    get_block_encoding_reference,       
    get_coordinate_system_reference     
)
from datasets import load_dataset
# from oracle import enumerate_correct_actions
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
 
from agents.builder_tools import simulate_move

def calculate_iou_board(current, target):
        """
        Calculate Intersection over Union (IoU) for block positions
        """
        intersection = 0
        union = 0
        
        for coord in current.keys():
            current_blocks = set(current[coord])
            target_blocks = set(target[coord])
            
            intersection += len(current_blocks.intersection(target_blocks))
            #print(f"I: {intersection} {current_blocks} {target_blocks}")
            union += len(current_blocks.union(target_blocks))
            #print(f"{union}")
        
        return intersection / union if union > 0 else 0.0

def normalize_structure(structure):
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

def calculate_iou(list1: list[str], list2: list[str]) -> float:
    """
    Calculates the IoU for two lists of strings.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The IoU as a float between 0 and 1.
    """
    set1 = set(list1)
    set2 = set(list2)

    intersection = len(set1.intersection(set2)) # or len(set1 & set2)
    union = len(set1.union(set2))               # or len(set1 | set2)

    if union == 0:
        return 0.0  # Avoid division by zero if both sets are empty

    return float(intersection) / float(union)

def getParticalView(director, board):
    partBoard = {}
    if director == "D1":
        for k in board:
            if k != '(0,0)' and k != '(1,0)' and k != '(2,0)':
                partBoard[k] = []
            else:
                partBoard[k] = board[k]
        return partBoard
    
    if director == "D2":
        for k in board:
            if k != '(0,0)' and k != '(0,1)' and k != '(0,2)':
                partBoard[k] = []
            else:
                partBoard[k] = board[k]
        return partBoard
        
    if director == "D3":
        for k in board:
            if k != '(0,2)' and k != '(1,2)' and k != '(2,2)':
                partBoard[k] = []
            else:
                partBoard[k] = board[k]
        return partBoard

def _encode_block(block_name) -> str:
        """Convert block name to encoded format"""
        # block_name is already in correct format (e.g., 'os', 'bs', 'ys')
        if block_name in ["gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol"]:
            return block_name
        else:
            return "unknown"

def _place_block(current_structure, current_spans, position, block, layer, span_to=None):
        """
        Place a block at position.
        If block is large, places on both position and span_to atomically.
        """
        try:
            block_encoded = _encode_block(block)

            # ── Large block: place on both cells atomically ────────
            if block_encoded.endswith('l'):
                if not span_to:
                    print(f"DEBUG: Large block '{block_encoded}' requires span_to")
                    return False, current_structure, current_spans

                if position not in current_structure:
                    current_structure[position] = []
                if span_to not in current_structure:
                    current_structure[span_to] = []
                # optional safety: neighbor must be same height
                if len(current_structure[position]) != len(current_structure[span_to]):
                    print("DEBUG: large block placement requires equal stack heights")
                    return False, current_structure, current_spans

                current_structure[position].append(block_encoded)
                current_structure[span_to].append(block_encoded)

                # Record span
                current_spans.setdefault(layer, []).append((position, span_to))

                print(f"DEBUG: Placed large '{block_encoded}' "
                    f"at {position}↔{span_to} layer {layer}")

            # ── Small block: single cell ───────────────────────────
            else:
                if position not in current_structure:
                    current_structure[position] = []
                current_structure[position].append(block_encoded)
                print(f"DEBUG: Placed small '{block_encoded}' "
                    f"at {position} layer {layer}")

            return True, current_structure, current_spans

        except Exception as e:
            print(f"Error placing block: {e}")
            return False, current_structure, current_spans
    
def _remove_block(current_structure, current_spans, position, layer, span_to=None):
    """
    Remove the top block from position.
    If block is large, removes from both position and span partner atomically.
    """
    try:
        stack = current_structure.get(position, [])

        if len(stack) == 0:
            print(f"DEBUG: Cannot remove — stack at {position} is empty")
            return False, current_structure, current_spans

        if layer != len(stack) - 1:
            print(
                f"DEBUG: Cannot remove layer {layer} — "
                f"must remove top (layer {len(stack)-1}) first"
            )
            return False, current_structure, current_spans

        top_block = stack[-1]

        # ── Large block: remove both halves atomically ─────────
        if top_block.endswith('l'):
            if not span_to:
                print(f"DEBUG: Large block removal requires span_to")
                return False, current_structure, current_spans

            neighbor_stack = current_structure.get(span_to, [])

            # Neighbor must also have this block at same layer
            if len(neighbor_stack) == 0 or neighbor_stack[-1] != top_block:
                print(
                    f"DEBUG: Span partner {span_to} does not have matching "
                    f"block '{top_block}' at top"
                )
                return False, current_structure, current_spans

            if len(neighbor_stack) - 1 != layer:
                print(
                    f"DEBUG: Span partner {span_to} top layer "
                    f"{len(neighbor_stack)-1} != {layer}"
                )
                return False, current_structure, current_spans

            # Remove from both cells
            current_structure[position].pop()
            current_structure[span_to].pop()

            # Remove span record
            layer_spans = current_spans.get(layer, [])
            current_spans[layer] = [
                (a, b) for a, b in layer_spans
                if not (a == position and b == span_to)
                and not (a == span_to and b == position)
            ]

            print(f"DEBUG: Removed large block '{top_block}' "
                f"from {position}↔{span_to} layer {layer}")

        # ── Small block: remove single cell ───────────────────
        else:
            current_structure[position].pop()
            print(f"DEBUG: Removed small block '{top_block}' "
                f"from {position} layer {layer}")

        return True, current_structure, current_spans

    except Exception as e:
        print(f"Error removing block: {e}")
        return False, current_structure, current_spans

class BuilderType(Enum):
    Base = "Base"
    Literal1 = "Literal1"
    Pragmatic1 = "Pragmatic1"
    Literal2 = "Literal2"
    Pragmatic2 = "Pragmatic2"
    Literal3 = "Literal3"
    Pragmatic3 = "Pragmatic3"

pragmatic_addIns = {
    BuilderType.Base: '',
    BuilderType.Literal1: """
Read each director's message at face value.
Use only the semantic meaning of the directors' utterances and your prior beliefs in your interpretation. 
Disregard director intent or informativeness and consider only the literal interpretation.
""",
    BuilderType.Pragmatic1: '''
Infer what you can from the director's message.

Before deciding on a move, consider the following and make inferences based on your own thinking:
1. Identify which director gave the most informative description in order to take an action and further the task.
2. Do you think any of the dialogue is false or misleading?
3. Do you find the dialogue to be appropriate and focused on the task?
4. Do you find the meaning of any words used in the dialogue to be unclear or obscure?
''',
    BuilderType.Literal2: """
Interpret each director's messages with strict literalism.
Use only the semantic meaning of the directors' utterances and your prior beliefs in your interpretation. 
Treat the director statements as a series of literal propositions, independent of the speaker's goals.
""",
    BuilderType.Pragmatic2: '''
Infer what you can from the director's message.

Before deciding on a move, consider the following and make inferences based on your own thinking:
1. Do you find any of the dialogue to be unnecessary in order for you to understand the perspectives of the directors?
2. Do find that any of the observations or conclusions made in the dialogue lack adequate evidence to support their ideas? 
3. Do you find the dialogue to be appropriate and focused on the task?
4. Do you find the organization of the dialogue to be clear and easy to follow?
''',
    BuilderType.Literal3: """
Restrict interpretation to the compositional semantics of each utterance.
Disregard pragmatic inference or speaker intent; evaluate statements solely as independent logical propositions.
Base interpretations exclusively on literal meaning and established prior knowledge.
""",
    BuilderType.Pragmatic3: '''
Infer what you can from the director's message.

Before deciding on a move, consider the following and make inferences based on context, intent, and shared knowledge
1. Do you find the dialogue to be informative enough to take an action and further the task?
2. Is any of this phrasing potentially misrepresentative?
3. Does the exchange remain focused, or does it drift into irrelevant details?
4. Is the structure of this conversation easy to navigate?
''',
}

class BuilderAgent:
    """Builder agent that uses API calls"""
    
    def __init__(self, api_key=None, model_name="gpt-4o-mini", oracle_moves=None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model_name = model_name
        self.oracle_moves = oracle_moves

    def format_oracle_moves_for_prompt(self, moves):
        """
        Format correct moves for builder prompt.
        Large blocks include span_to, small blocks do not.
        """
        lines = []
        for m in moves:
            action = m["action"]
            block  = m.get("block", "")
            pos    = m["position"]
            layer  = m["layer"]
            span   = m.get("span_to")
            
            if action == "place":
                if span:
                    lines.append(f"  PLACE {block} at {pos} layer {layer} spanning to {span}")
                else:
                    lines.append(f"  PLACE {block} at {pos} layer {layer}")
            elif action == "remove":
                if span:
                    lines.append(f"  REMOVE from {pos} layer {layer} spanning to {span}")
                else:
                    lines.append(f"  REMOVE from {pos} layer {layer}")
        
        return "\n".join(lines)

    def compute_builder_prompt_section_lengths(self, prompt: str, tokenizer=None) -> dict:
        """
        Breaks down token length of each major section of the builder prompt.
        """
        import re
        model_name_dummy = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer_dummy = AutoTokenizer.from_pretrained(model_name_dummy)

        def tok(text):
            if tokenizer_dummy:
                return len(tokenizer_dummy.encode(text, add_special_tokens=False))
            return len(text.split())  # fallback: word count

        sections = {}
        sections["total_prompt_tokens"] = tok(prompt)

        def extract(pattern):
            m = re.search(pattern, prompt, re.DOTALL)
            return tok(m.group(0)) if m else 0

        sections["spatial_orientation"]     = extract(r"SPATIAL ORIENTATION.*?DIRECTOR PERSPECTIVE GUIDE")
        sections["director_perspective"]    = extract(r"DIRECTOR PERSPECTIVE GUIDE.*?EXAMPLE FRAME OF REFERENCE")
        sections["frame_of_reference_ex"]   = extract(r"EXAMPLE FRAME OF REFERENCE.*?CURRENT BOARD STATE")
        sections["current_board_state"]     = extract(r"CURRENT BOARD STATE.*?AVAILABLE BLOCKS")
        sections["available_blocks"]        = extract(r"AVAILABLE BLOCKS.*?(?:CANDIDATE MOVES|DIRECTOR DISCUSSION)")
        sections["oracle_section"]          = extract(r"CANDIDATE MOVES.*?DIRECTOR DISCUSSION") if "CANDIDATE MOVES" in prompt else 0
        sections["director_discussion"]     = extract(r"DIRECTOR DISCUSSION.*?DECISION RULE")
        sections["decision_rule"]           = extract(r"DECISION RULE.*?STACKING RULES")
        sections["stacking_rules"]          = extract(r"STACKING RULES.*?FRAME OF REFERENCE RULE")
        sections["frame_of_reference_rule"] = extract(r"FRAME OF REFERENCE RULE.*?LARGE BLOCK RULE")
        sections["large_block_rule"]        = extract(r"LARGE BLOCK RULE.*?EXAMPLE SPAN ANALYSIS")
        sections["span_example"]            = extract(r"EXAMPLE SPAN ANALYSIS.*?WHEN MOVES FAIL")
        sections["when_moves_fail"]         = extract(r"WHEN MOVES FAIL.*?OUTPUT FORMAT")
        sections["output_format"]           = extract(r"OUTPUT FORMAT.*?$")

        print(f"\n{'='*55}")
        print(f"  BUILDER PROMPT TOKEN BREAKDOWN")
        print(f"{'='*55}")
        for k, v in sections.items():
            if k != "total_prompt_tokens":
                bar = "█" * (v // 20)
                print(f"  {k:<30} {v:>5} {'tokens' if tokenizer else 'words'}  {bar}")
        print(f"  {'─'*50}")
        print(f"  {'total':<30} {sections['total_prompt_tokens']:>5} {'tokens' if tokenizer else 'words'}")
        print(f"{'='*55}\n")

        return sections

    def create_builder_prompt(self, director_discussion, current_state, 
                          available_blocks, oracle_moves=None, builderPromptAddin = BuilderType.Base):  
        """Create builder prompt with confirmation system"""
        

        block_reference = get_block_encoding_reference()
        coordinate_reference = get_coordinate_system_reference()
        print(pragmatic_addIns[builderPromptAddin])
        print(f"  [PROMPT] oracle_moves received: {oracle_moves is not None}, count={len(oracle_moves) if oracle_moves else 0}")

        oracle_section = ""
        if oracle_moves:
            formatted = self.format_oracle_moves_for_prompt(oracle_moves)

            # oracle_section = f"""
            # CANDIDATE MOVES (verified physically valid for this turn):
            # {formatted}

            # Select the candidate that best matches the directors' descriptions.
            # In your CONFIRM field, write 2-3 sentences covering:
            # 1. Which director(s) you are following and why their instruction matches this candidate.
            # 2. What the other director(s) said and whether they agreed, conflicted, or addressed a different part of the structure.
            # 3. Why you chose this candidate over any others in the list.
            # """

            oracle_section = f"""
            CANDIDATE MOVES (verified physically valid for this turn):
            {formatted}

            From this list, select the move that you believe at least one director is asking for based on their discussion.
            If no candidate clearly matches what any director is describing, CLARIFY.
            """
           
         

        
        return f"""You are a Builder in a collaborative LEGO construction task.

The three Directors (D1, D2, and D3) have to instruct you to build a single structure that is consistent with the private views of the structure they have.
Your job is to place, move, or remove blocks on the board to build the structure.
From a top-down view of the target structure, D1's private view is of the left wall of the structure, D2's view is of the top wall of the structure, and D3's view of the right wall of the structure.
From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.

{pragmatic_addIns[builderPromptAddin]}

SPATIAL ORIENTATION (use only in your thinking)
The coordinate grid from above:
  (0,0) (0,1) (0,2)   ← this is the "far" / "back" row
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)   ← this is the "near" / "front" row
Large blocks span SIDEWAYS or FORWARD/BACK — never stacked vertically.

DIRECTOR PERSPECTIVE GUIDE:
D1: From left to right, sees cells (0,0), (1,0), (2,0) across all layers.
D2: From left to right, sees cells (0,0), (0,1), (0,2) across all layers.
D3: From left to right, sees cells (0,2), (1,2), (2,2) across all layers.

When interpreting the instructions from D1, D2, or D3 instructions, you MUST adopt the frame of reference of the speaker. 
For instance, to D1, "my bottom left corner" is coordinate (0,0) at layer 0 and "my top right corner" is coordinate (2,0) at layer 2. 
To D2, "my bottom left corner" is coordinate (0,0) at layer 0 and "my top right corner" is coordinate (0,2) at layer 2. 
To D3, "my bottom left corner" is coordinate (0,2) at layer 0 and "my top right corner" is coordinate (2,2) at layer 2.

EXAMPLE FRAME OF REFERENCE ANALYSIS:
    Given board state
    {{
        "(0,0)": [],
        "(0,1)": [],
        "(0,2)": [],
        "(1,0)": [],
        "(1,1)": [],
        "(1,2)": [],
        "(2,0)": [],
        "(2,1)": [],
        "(2,2)": []
    }}
    Given utterance
    [D1: Could you please place a small orange block in my bottom left corner?]
    Correct move
    [PLACE:os:(0,0):0:CONFIRM:Placing small orange block at bottom-left of D1's side as requested".]
    Given utterance
    [D2: Please remove the large orange block from my bottom left and middle cells.]
    Correct move
    [REMOVE:(0,0):0:(0,1):CONFIRM:Removing the large orange block from bottom-left+bottom-middle of D2's side as requested.]
    Given utterance
    [D3: Let's begin by placing a large green block across the left and middle cells of my bottom layer.]
    Correct move
    [PLACE:gl:(0,2):0:(1,2):CONFIRM:Placing large green block across the left and middle cells of D3's bottom layer as requested.]

Positions invisible to ALL directors: (1,1) and (2,1)
A large block that is visible to ANY of the directors CANNOT span EITHER (1,1) or (2,1)
— only inferred from what's missing in other views
    CURRENT BOARD STATE: {json.dumps(current_state, indent=2)}
    AVAILABLE BLOCKS: {', '.join(available_blocks)}

    {block_reference} 
    {coordinate_reference}
{oracle_section}
DIRECTOR DISCUSSION:
{director_discussion}
DECISION RULE: If 2+ directors agree on a block or position, do that first.
If all three disagree, pick the most specific instruction.
STACKING RULES:
- "layer" means stack depth, NOT grid row
- ALWAYS calculate layer from CURRENT BOARD STATE, never trust director-specified layers
- Before ANY place: count blocks at target position from CURRENT BOARD STATE
  → You MUST place new blocks one layer above the number of blocks at that position (e.g., if position (0,1) has ['gs', 'ol'] → next block goes at layer 2; if position (0,1) has [''] → next block goes at layer 0
- Before ANY remove: verify position is non-empty in CURRENT BOARD STATE
  → If empty, do NOT attempt removal — tell directors and suggest placing instead
  
FRAME OF REFERENCE RULE:
IMPORTANT: When choosing where to place a block, you MUST adopt the frame of reference of the director whose instruction you are following.
REMINDER: "The left" of D1's view is coordinate (0,0) and "the right" is coordinate (2,0). 
"The left" of D2's view is coordinate (0,0) and "the right" is coordinate (0,2). 
"The left" of D3's view is coordinate (0,2) and "the right" is coordinate (2,2).
NEVER deviate from these frames of reference when executing instructions.

LARGE BLOCK RULE:
Large blocks span TWO adjacent cells — you MUST specify both endpoints.

To choose span_to:
- Identify the TWO director-relative cells explicitly referenced (e.g., "left+middle", "middle+right", "bottom left+bottom middle").
- Convert those two cells into global coordinates using the DIRECTOR PERSPECTIVE GUIDE.
- Ensure BOTH cells lie on the correct wall for that director.
- Set position to one endpoint and span_to to the other endpoint.
- Before outputting, verify:
  (a) position and span_to are orthogonal neighbors,
  (b) both endpoint stacks have the SAME height (so placement/removal is on the same layer),
  (c) neither endpoint is an invisible cell ((1,1) or (2,1)).

NEVER place OR remove a large block if span_to is None — it will always fail.

Format: PLACE:block:position:layer:span_to:CONFIRM:reason
Example: PLACE:gl:(0,0):0:(1,0):CONFIRM:Placing large green block across the left and middle cells of D1's bottom layer as requested

If a director says "green large in the corner", you must figure out 
which two adjacent cells it spans from the CURRENT BOARD STATE.
NEVER place OR remove a large block if span_to is None — it will always fail.
If you try to remove a large block, you MUST check the board state to see where spans contain the same block.

EXAMPLE SPAN ANALYSIS:
    Given board state
    {{
        "(0,0)": [],
        "(0,1)": [],
        "(0,2)": [],
        "(1,0)": [],
        "(1,1)": [],
        "(1,2)": [],
        "(2,0)": [],
        "(2,1)": [
            "gl"
        ],
        "(2,2)": [
            "gl"
        ]
    }}
    Given raw move
    [REMOVE:(2,2):0:CONFIRM:Removing the large green block from the bottom layer as requested by D3.]
    Correct move
    [REMOVE:(2,2):0:(2,1):CONFIRM:Removing the large green block from the bottom layer as requested by D3.]

WHEN MOVES FAIL:
- Explain WHY: e.g., "I can't remove any block from the middle cell on the bottom layer. There is no block there. Suggest placing [block] instead."
- Never silently retry the same failed move

BEFORE PLACING: Think step by step to make sure that you have interpreted the instructions, including block color and size, and the director's frame of reference, correctly.

    Do not place a block at the same place where you have previously removed a block of the same color.

    Count blocks at target position from CURRENT BOARD STATE
    
    EXAMPLE BLOCK COUNT AT TARGET POSITION:
        Given board state:
        {{
            "(0,0)": [
              "os"
            ],
            "(0,1)": [],
            "(0,2)": [],
            "(1,0)": [],
            "(1,1)": [],
            "(1,2)": [
              "bl"
            ],
            "(2,0)": [
              "gl",
              "bl"
            ],
            "(2,1)": [
              "gl",
              "bl"
            ],
            "(2,2)": [
              "bl"
            ]
        }}
        Given raw move
        [PLACE:gl:(2,2):0:(2,1):CONFIRM:Placing large green block across the left and middle cells of D3's bottom layer as requested.]
        Correct move
        [PLACE:gl:(2,2):2:(2,1):CONFIRM:Placing large green block across the left and middle cells of D3's bottom layer as requested.]

    OUTPUT FORMAT - Choose ONE of these exact formats:

    1. To place small block: PLACE:block_code:position:layer:CONFIRM:interpretation
    Example: PLACE:bs:(0,0):0:CONFIRM:Placing blue small block at bottom-left of D1's side as requested

    2. To place large block: PLACE:block_code:position:layer:span_to:CONFIRM:interpretation
    Example: PLACE:gl:(0,0):0:(1,0):CONFIRM:Placing large green block across left and middle cells of D1's bottom layer

    3. To remove small block: REMOVE:position:layer:CONFIRM:interpretation
    Example: REMOVE:(1,2):0:CONFIRM:Removing the block from middle-right of D3's side as requested

    4. To remove large block: REMOVE:position:layer:span_to:CONFIRM:interpretation
    Example: REMOVE:(2,2):0:(2,1):CONFIRM:Removing large green block from D3's bottom layer as requested
    NOTE: REMOVE never includes block code — do NOT write REMOVE:bl:(0,0):...

    5. To clarify: CLARIFY:your specific question
    Example: CLARIFY:Which blue block should I move - the one on top or bottom?
    Always include CONFIRM section to show what you understood from their instructions."""


    def create_builder_prompt_with_tools(self, director_discussion, current_state,
                                     available_blocks, max_simulations=3,
                                     oracle_moves=None, builderPromptAddin = BuilderType.Base):  
        """
        Thin wrapper: keep the base prompt exactly as-is, and append only tool policy.
        (No restatement of schemas / long contracts.)
        """
        base = self.create_builder_prompt(
        director_discussion=director_discussion,
        current_state=current_state,
        available_blocks=available_blocks,
        oracle_moves=oracle_moves,
        builderPromptAddin = builderPromptAddin)

        thin_addendum = f"""
---

TOOL MODE — simulate_move available ({max_simulations} calls max):

WORKFLOW:
1. Simulate each director's instruction once directly and literally.
2. Pick the result with greatest value for "progress".
3. Submit that exact move as your FINAL answer. DO NOT INVENT NEW MOVE AFTER SIMULATING.
4. If a sim fails (ok=False) → fix ONLY the field the hint specifies, retry once.
5. NEVER submit a move that returned ok=False.
6. NEVER submit a remove move where simulate shows structurePlacement=False — 
   even if it's the only ok=True simulation. In that case, CLARIFY instead.
7. NEVER clarify just because directors disagree — simulate and pick the best.
8. NEVER remove a block where simulate shows structurePlacement=False for that remove.
"""

        return base + thin_addendum



    def parse_builder_response(self, response_text):
        """Parse builder response including confirmation"""
        try:
            # response_text = response_text.strip()
            response_text = response_text.strip().strip("[]")  # add this

            # ── PLACE ─────────────────────────────────────────────
            if response_text.startswith("PLACE:"):
                parts = response_text.split(":", 6)

                # Large block: PLACE:block:position:layer:span_to:CONFIRM:interpretation
                if len(parts) == 7 and parts[5] == "CONFIRM":
                    return {
                        "action"      : "place",
                        "block"       : parts[1],
                        "position"    : parts[2],
                        "layer"       : int(parts[3]),
                        "span_to"     : parts[4],
                        "confirmation": parts[6]
                    }

                # Small block: PLACE:block:position:layer:CONFIRM:interpretation
                elif len(parts) >= 6 and parts[4] == "CONFIRM":
                    return {
                        "action"      : "place",
                        "block"       : parts[1],
                        "position"    : parts[2],
                        "layer"       : int(parts[3]),
                        "span_to"     : None,
                        "confirmation": parts[5]
                    }

                # Fallback: PLACE:block:position:layer
                elif len(parts) >= 4:
                    return {
                        "action"      : "place",
                        "block"       : parts[1],
                        "position"    : parts[2],
                        "layer"       : int(parts[3]),
                        "span_to"     : None,
                        "confirmation": f"Placing {parts[1]} at {parts[2]}"
                    }

            # ── REMOVE ────────────────────────────────────────────
            elif response_text.startswith("REMOVE:"):
                parts6 = response_text.split(":", 5)
                parts5 = response_text.split(":", 4)

                # Detect if second field is a block code (e.g. 'bl', 'os') vs a position '(x,y)'
                def is_position(s):
                    return s.strip().startswith("(")

                def is_block_code(s):
                    return len(s.strip()) <= 3 and not s.strip().startswith("(")

                # Strip optional block code if present: REMOVE:bl:(0,0):0:span_to:CONFIRM:reason
                raw_parts = response_text.split(":")
                if len(raw_parts) >= 2 and is_block_code(raw_parts[1]):
                    # Rebuild without the block code
                    response_text = "REMOVE:" + ":".join(raw_parts[2:])
                    parts6 = response_text.split(":", 5)

                # Large block: REMOVE:position:layer:span_to:CONFIRM:reason
                if len(parts6) == 6 and parts6[4] == "CONFIRM":
                    return {
                        "action": "remove",
                        "position": parts6[1],
                        "layer": int(parts6[2]),
                        "span_to": parts6[3],
                        "confirmation": parts6[5]
                    }
                # Small block: REMOVE:position:layer:CONFIRM:reason
                parts5 = response_text.split(":", 4)
                if len(parts5) >= 5 and parts5[3] == "CONFIRM":
                    return {
                        "action": "remove",
                        "position": parts5[1],
                        "layer": int(parts5[2]),
                        "span_to": None,
                        "confirmation": parts5[4]
                    }
            # ── CLARIFY ───────────────────────────────────────────
            elif response_text.startswith("CLARIFY:"):
                return {
                    "action"       : "clarify",
                    "clarification": response_text[8:]
                }

            # ── Fallback ──────────────────────────────────────────
            return {
                "action"       : "clarify",
                "clarification": f"Could not parse response: {response_text}"
            }

        except Exception as e:
            return {
                "action"       : "clarify",
                "clarification": f"Parse error: {str(e)}"
            }
            

    def get_builder_prompt(self, director_discussion, current_state, available_blocks,
                       use_tools=False, max_simulations=3, oracle_moves=None, builderPromptAddin = BuilderType.Base):  # ← add
        if use_tools:
                return self.create_builder_prompt_with_tools(
                    director_discussion, current_state, available_blocks,
                    max_simulations=max_simulations,
                    oracle_moves=oracle_moves, 
                    builderPromptAddin = builderPromptAddin 
                )
        return self.create_builder_prompt(
                director_discussion, current_state, available_blocks,
                oracle_moves=oracle_moves, builderPromptAddin = builderPromptAddin  
            )

    def generate_move(self, director_discussion, current_state, available_blocks, oracle_moves=None, check_prompt_tokens = False) -> Dict:
        """Generate builder move with improved parsing"""
        try:
            prompt = self.create_builder_prompt(director_discussion, current_state, available_blocks, oracle_moves=oracle_moves,  )
            if check_prompt_tokens: 
                self.compute_builder_prompt_section_lengths(prompt)
 
            system_content_oracle = (
                "You are a Builder in a collaborative LEGO task. "
                "You have been given VERIFIED CANDIDATE MOVES — you MUST choose exactly one from the list. "
                "Respond in the specified PLACE/REMOVE/CLARIFY format. "
                "In your CONFIRM field, write 2-3 sentences: which director(s) you followed, "
                "whether others agreed or conflicted, and why you chose this candidate."
            )

            system_content_base = (
                "You are a Builder in a collaborative LEGO task. "
                "Respond in the specified PLACE/REMOVE/CLARIFY format. "
                "In your CONFIRM field, write 2-3 sentences: which director(s) you are following, "
                "what the other directors said and whether they agreed or conflicted, "
                "and why you chose this move."
            )

            system_content = system_content_oracle if oracle_moves else system_content_base
            # "You are a Builder. Respond with exactly one line in the specified format. No additional text or explanation. OLD block, slight change. 
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent formatting
                max_tokens=250    # Shorter to discourage extra text
            )
            
            if completion and completion.choices:
                response_text = completion.choices[0].message.content.strip()
                
                # Extract first line only (ignore any extra text)
                first_line = response_text.split('\n')[0].strip()
                
                return self.parse_builder_response(first_line)
            
            return {"action": "clarify", "clarification": "No response from API"}
            
        except Exception as e:
            return {"action": "clarify", "clarification": f"Error in move generation: {str(e)}"}


    ## suggestions for further changes in builder exploration with tools;
    ## builder currently makes repetitive calls to save move; requires a change in instruction for tool use; and restrict repeated moves.
    # 0. Before simulating, check if you already simulated this exact move — if yes, skip it and try a different move instead. 
    # recorded log to verify:repeated move:  {'action': 'place', 'block': 'os', 'position': '(0,0)', 'layer': 0}

    # 1: I see my bottom left corner is empty, and I need a small yellow block there to s
    # [PROMPT] oracle_moves received: False, count=0
    # [PROMPT] oracle_moves received: False, count=0
    # [SIM input] {'action': 'place', 'block': 'os', 'position': '(0,0)', 'layer': 0}
    # DEBUG: Placed small 'os' at (0,0) layer 0
    # DEBUG: Progress calculation - target blocks: 23
    # DEBUG: Progress calculation - current blocks: 2
    # tool call results {'ok': True, 'blocks_correct': 1, 'blocks_total': 23, 'overall_progress': 0.06740280653324131, 'correctness': {'structurePlacement': False, 'sidePlacement': False}}
    # [SIM ok] blocks_correct=1/23 progress=0.067
    # [SIM input] {'action': 'place', 'block': 'os', 'position': '(0,0)', 'layer': 0}
    # DEBUG: Placed small 'os' at (0,0) layer 0
    # DEBUG: Progress calculation - target blocks: 23
    # DEBUG: Progress calculation - current blocks: 2
    # tool call results {'ok': True, 'blocks_correct': 1, 'blocks_total': 23, 'overall_progress': 0.06740280653324131, 'correctness': {'structurePlacement': False, 'sidePlacement': False}}
    # [SIM ok] blocks_correct=1/23 progress=0.067
    # [SIM input] {'action': 'place', 'block': 'ys', 'position': '(0,0)', 'layer': 0}


    ## simulated_moves = set()

    # code change suggestion: inside the tool call execution loop:

    # move_key = json.dumps(move, sort_keys=True)
    # if move_key in simulated_moves:
    #     result = {"ok": False, "error": "duplicate simulation", 
    #             "hint": "You already simulated this exact move. Try a different move."}
    # else:
    #     simulated_moves.add(move_key)
    #     result = simulate_move(game_state, move)

        
    def generate_move_with_tools(
        self,
        director_discussion: str,
        game_state,
        max_simulations: int = 3,
        oracle_moves=None,  
        check_prompt_tokens = False
    ) -> Dict:
        system_msg = {
        "role": "system",
        "content": (
            "You are a Builder agent. You may call simulate_move up to "
            f"{max_simulations} times to dry-run moves. "
            "After simulations, output ONE final move in the PLACE/REMOVE/CLARIFY text format. "
            "No JSON, no extra commentary."
        )
    }
        user_msg = {
            "role": "user",
            "content": self.get_builder_prompt(
                director_discussion=director_discussion,
                current_state=game_state.current_structure,
                available_blocks=game_state.available_blocks,
                use_tools=True,
                max_simulations=max_simulations,
                oracle_moves = oracle_moves
            )
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simulate_move",
                    "description": (
                        "Dry-run a proposed move against the environment. "
                        "Returns {ok, error, metrics, correctness, resulting_structure}. "
                        "Does NOT mutate game state."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "move": {
                                "type": "object",
                                "description": (
                                    "Proposed move with keys: action (place|remove|clarify), "
                                    "block (e.g. 'gs'), position (e.g. '(0,0)'), "
                                    "layer (int), span_to (e.g. '(1,0)' or null)."
                                ),
                                "properties": {
                                    "action":   {"type": "string", "enum": ["place", "remove", "clarify"]},
                                    "block":    {"type": "string"},
                                    "position": {"type": "string"},
                                    "layer":    {"type": "integer"},
                                    "span_to":  {"type": ["string", "null"]},
                                },
                                "required": ["action", "position", "layer"]
                            }
                        },
                        "required": ["move"]
                    }
                }
            }
        ]

        messages = [system_msg, user_msg]
        tool_calls_used = 0  # tracks total individual simulate_move calls
        choice = None


        while tool_calls_used < max_simulations:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
            choice = response.choices[0]
            messages.append(choice.message)

            # No tool calls → model produced its final answer, exit loop
            if not choice.message.tool_calls:
                break

            # Count how many simulate calls this round would consume
            new_calls = len(choice.message.tool_calls)

            # If adding these would exceed the budget, stop before executing them
            if tool_calls_used + new_calls > max_simulations:
                # Discard the tool-calling assistant turn and force a final answer
                messages.pop()  # remove the tool-calling assistant message
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have used {tool_calls_used}/{max_simulations} simulations. "
                        "No more simulate_move calls allowed. "
                        "Output your FINAL move now in PLACE/REMOVE/CLARIFY text format."
                    )
                })
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    # No tools passed → model cannot call any
                )
                choice = response.choices[0]
                break

            # Execute the tool calls
            tool_calls_used += new_calls
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    move = args.get("move", args)
                    print(f"  [SIM input] {move}")  # add this
                    result = simulate_move(game_state, move)
 
                    if not result["ok"]:
                        error_msg = result['error']
                          # Extract expected span_to if present in error message
                        
                        expected_match = re.search(r'expected (\(\d,\d\))', error_msg)
                        partner_hint = (
                            f" The correct span_to is {expected_match.group(1)}."
                            if expected_match else ""
                        )
                        
                        result["hint"] = (
                            f"Move failed: {error_msg}.{partner_hint} "
                            "Fix this specific move and try again. "
                            "Do NOT submit a move that already failed in simulation."
                        )
                except Exception as e:
                    result = {"ok": False, "error": f"simulate_move exception: {e}",
                            "hint": "Exception running simulation. Fix your move format."}
                print("tool call results", result)
                # Clean log — show only what matters
                if result["ok"]:
                    print(f"  [SIM ok] blocks_correct={result.get('blocks_correct')}/{result.get('blocks_total')} progress={result.get('overall_progress', 0):.3f}")
                else:
                    print(f"  [SIM fail] {result.get('hint', 'unknown')}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            # If budget exactly exhausted, force final answer now
            if tool_calls_used >= max_simulations:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Simulation budget exhausted ({max_simulations}/{max_simulations}). "
                        "Output your FINAL move now in PLACE/REMOVE/CLARIFY text format. "
                        "Do NOT call simulate_move again."
                    )
                })
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    # No tools → hard block on further calls
                )
                choice = response.choices[0]
                break

        print(f"  [TOOL] simulate called {tool_calls_used} times this turn")
        if choice is None:
            return {"action": "clarify", "clarification": "Builder loop did not execute."}

        final_text = (choice.message.content or "").strip()
        if not final_text:
            return {
                "action": "clarify",
                "clarification": "Builder produced no final text after tool calls."
            }

        first_line = final_text.split("\n")[0].strip()
        return self.parse_builder_response(first_line)

if __name__ == "__main__":
    # don't need tools
    # using oracle values
    # structure before 
    # as close as we can be to the existing prompt
    # saving off satisfication after new move (comparing to factual averages)
    with open(f"/home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json", 'r') as file:
        structData = json.load(file)

    folder_path = Path("/home/hannah/CRAFT/CRAFT/divergenceData/train")
    aggregated_data = []
    ds = load_dataset("Abhijnan/craft-benchmark-lean")

    for file_path in folder_path.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)   
            aggregated_data.extend(data)

    api_key = os.getenv('OPENAI_API_KEY')
    builder_model_name = "gpt-4o"
    builder_agent = BuilderAgent(api_key=api_key, model_name=builder_model_name)

    all_turns_counter = {}
    count = 0
    for turn in aggregated_data:    
        count += 1
        print(f"Turn {count}/{len(aggregated_data)}") 
        #factual = turn[turn["builderSelected"]]
        counterfact = turn[turn["predicted"]]

        available_blocks = ["gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol"]
        turnIndex = counterfact["timestamp"] 
        structure = counterfact["structure"]   
        utterance = turn["predicted"] + ": " + counterfact["utterance"]  
        currentStructure = json.loads(counterfact["structureBefore"].replace("\'", "\"")) 
        director = counterfact["modelCombo"].split("+")[0].strip().replace("-", "").lower()
        if(director == "deepseeklite"):
            director = "deepseekv2lite"

        filtered_ds = ds.filter(lambda example: example["turn_number"] == turnIndex and
                                example["structure_id"] == structure and example["director_model"].replace("-", "").lower() == director)

        if(len(filtered_ds['train']) == 0):
            print(f"Can't find value in dataset {turnIndex}, {structure},{director}")
            continue
        huggingFaceRow = filtered_ds['train'][0]
        builder_move = builder_agent.generate_move(
        director_discussion=utterance,
        current_state=currentStructure,  # pass state so tool can simulate
        available_blocks=available_blocks,
        oracle_moves= json.loads(huggingFaceRow["oracle_moves"]), 
        )
    
        print(f"  Builder move: {builder_move}")

        structure = currentStructure
        # ── Execute ───────────────────────────────────────
        if builder_move['action'] == 'place':
            success, structure, spans = _place_block(
                    currentStructure, 
                    json.loads(huggingFaceRow["spans_before"]),
                    builder_move["position"],
                    builder_move["block"],
                    builder_move["layer"],
                    builder_move["span_to"])
        elif builder_move['action'] == 'remove':
            success, structure, spans = _remove_block(
                    currentStructure, 
                    json.loads(huggingFaceRow["spans_before"]),
                    builder_move["position"],
                    builder_move["layer"],
                    builder_move["span_to"])

        particalViewBoard = getParticalView(turn["predicted"], structure)
        current_norm = normalize_structure(particalViewBoard)

        for s in structData:
            if(s['id'] == huggingFaceRow["structure_id"]):
                targetStruct = s["structure"]
                break

        target_norm = normalize_structure(targetStruct)
        iou_score = calculate_iou_board(current_norm, target_norm)

        key = f"{counterfact["modelCombo"]}_{huggingFaceRow["structure_id"]}"
        if(not all_turns_counter.__contains__(key)):
            all_turns_counter[key] = []
        all_turns_counter[key].append({
            "turn": turnIndex,
            "counterfactualDirector": turn["predicted"],
            "utterance": utterance,
            "structure_before": currentStructure,
            "structure_after": structure,
            "builder_move": builder_move,
            "satisfaction": iou_score
        })

    for k in all_turns_counter:
        with open(f"/home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/train/{k}.json", "a", encoding="utf-8") as f:
                json.dump(
                    all_turns_counter[k],
                    f,
                    indent=2,
                    default=lambda obj: (
                        obj.to_dict() if hasattr(obj, "to_dict") else str(obj)
                    ),
                )
        