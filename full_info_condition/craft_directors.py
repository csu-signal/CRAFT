#!/usr/bin/env python3
"""
craft_directors.py
==================

Director mode for run_full_info_builder.py (enabled with --directors).

Game loop as in CRAFT (paper Sec. 5.2): each turn a random 1-3 unique Directors
speak in random order, each seeing the full dialogue so far and the board; the
Builder then sees ONLY this turn's Director messages and the board, and makes
one PLACE / REMOVE / CLARIFY move.

--director-views selects the information condition:
  all  (default)  every Director gets all three target views (no information separation)
  own             every Director gets only its own wall (standard CRAFT; use as the control)

Prompts
-------
Ported from the CRAFT repo's agents so the only intended difference between the
two conditions is the information given to Directors:
  * Director prompt  <- director_agent.DirectorAgent.create_enhanced_director_prompt_with_references
  * Builder prompt   <- builder_agent.BuilderAgent.create_builder_prompt (BuilderType.Base)
In 'own' mode both are verbatim apart from whitespace. 'all' mode replaces only the
sections that assume one private view and adds one rule for referring to another
Director's wall (see ALL_* constants). The Builder never sees views, as in CRAFT.

get_block_encoding_reference / get_coordinate_system_reference come from
structure_generator_v2 when that module is importable (i.e. when run inside the
CRAFT repo); otherwise short stand-ins are used and recorded in the game log.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Dict, List, Optional

from craft_full_info_env import (
    BLOCKS, DIRECTORS, Board, enumerate_oracle_moves, compute_metrics, min_moves,
    sample_oracle_moves, target_views, views_ceiling,
)

try:   # real CRAFT reference strings when running inside the CRAFT repo
    from structure_generator_v2 import get_block_encoding_reference, get_coordinate_system_reference
    BLOCK_REFERENCE, COORD_REFERENCE = get_block_encoding_reference(), get_coordinate_system_reference()
    REFERENCE_SOURCE = "structure_generator_v2"
except Exception:                                                     # noqa: BLE001
    BLOCK_REFERENCE = ("BLOCK ENCODING: first letter = colour (g=green, b=blue, r=red, y=yellow, o=orange); "
                       "second letter = size (s=small, one cell; l=large, two adjacent cells on the same layer).")
    COORD_REFERENCE = ("COORDINATES: (row,col). Row 0 is the far/back row, row 2 the near/front row; column 0 is "
                       "the left side (D1's wall), column 2 the right side (D3's wall). Layer = stack depth, "
                       "0 = bottom, at most 3 blocks per cell.")
    REFERENCE_SOURCE = "fallback"

# --------------------------------------------------------------------------- #
# Archetypes (director_agent.DirectorAgent)
# --------------------------------------------------------------------------- #
TYPES = ["assertive", "cautious", "observant", "skeptical", "synthesizer"]
ARCHETYPES = {
    "assertive": (
        "You are confident and direct. You form hypotheses quickly from your data "
        "and share them, but you genuinely listen to other groups and update your "
        "thinking when their evidence is compelling. You sometimes move faster than "
        "the evidence warrants but you're not closed-minded."),
    "cautious": (
        "You are methodical and prefer to verify before claiming. You ask clarifying "
        "questions and often synthesize what others have said before adding your own "
        "interpretation. You can make claims when evidence is strong enough — you're "
        "not paralyzed, just careful."),
    "observant": (
        "You notice patterns and anomalies in your data that others might overlook. "
        "You tend to flag inconsistencies and ask 'does this match what you're seeing?' "
        "rather than broadcasting conclusions. You're collaborative by nature and "
        "often connect dots across groups."),
    "skeptical": (
        "You question assumptions including your own. When someone makes a claim you "
        "probe it — not to be difficult but because you want the group to get it right. "
        "You're comfortable with uncertainty and say so openly."),
    "synthesizer": (
        "You actively try to integrate what all groups are saying into a coherent "
        "picture. You summarize, reconcile contradictions, and push the group toward "
        "a shared understanding. You ask 'how does your data fit your view and what the other directors have said?'"),
}


def assign_archetype(structure_index: int, director_id: str, run: int, previousData = None) -> str:
    """Same rule as DirectorAgent.__init__: hash((structure_index, director_num, run)).
    Hashes of int tuples are stable across processes (PYTHONHASHSEED only affects str/bytes)."""
    if os.path.isdir(previousData):
        filepath = f"{previousData}/dpip_structure_{structure_index + 1:03d}_{run}.json"
        if not os.path.exists(filepath):
            filepath = f"{previousData}/craft_structure_{structure_index + 1:03d}_{run}.json"
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data['games'][0][f'{director_id} Archetype']

    director_num = {"D1": 0, "D2": 1, "D3": 2}[director_id]
    seed = hash((structure_index, director_num, run)) % (2 ** 32)
    return random.Random(seed).choice(TYPES)


# --------------------------------------------------------------------------- #
# Director prompt
# --------------------------------------------------------------------------- #
PERSPECTIVE = {
    "D1": "From left to right, you see cells (0,0), (1,0), (2,0) across all layers.",
    "D2": "From left to right, you see cells (0,0), (0,1), (0,2) across all layers.",
    "D3": "From left to right, you see cells (0,2), (1,2), (2,2) across all layers.",
}

D_SPATIAL = """
   ### SPATIAL ORIENTATION (use only in your thinking)
The coordinate grid from above:
  (0,0) (0,1) (0,2)   ← this is the "far" / "back" row
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)   ← this is the "near" / "front" row
Large blocks span SIDEWAYS or FORWARD/BACK — never stacked vertically.
"""

D_INTERPRET = """
    ### HOW TO INTERPRET YOUR TARGET VIEW
    - IMPORTANT: In the JSON, keys are named row_0/row_1/row_2, but they refer to LAYERS (vertical stack depth), not grid rows.
    - row_0 = layer_0 (bottom layer / stack depth 0)
    - row_1 = layer_1 (middle layer / stack depth 1)
    - row_2 = layer_2 (top layer / stack depth 2)
    - in each layer, blocks are listed from LEFT to RIGHT according to YOUR VIEW
    - In your PUBLIC message, say "bottom layer / middle layer / top layer" (avoid saying "bottom row").
    - color=none means that cell should be empty
    - size of 1 = the block is a small block, size of 2 = the block is a large block and spans two adjacent cells
    - if two adjacent cells in your target view, have the same color and BOTH are size 2, this means that a SINGLE large block occupies both those cells
"""

ALL_INTERPRET_EXTRA = """    - YOU HAVE ALL THREE TARGET VIEWS. Each view is listed LEFT to RIGHT from the seat of the Director who owns that wall:
        D1 (left wall):  (0,0), (1,0), (2,0)
        D2 (far wall):   (0,0), (0,1), (0,2)
        D3 (right wall): (0,2), (1,2), (2,2)
    - A large block can look size 2 on one wall and size 1 on another; cells (1,1) and (2,1) appear in no view
"""

D_EXAMPLE = """
    ### EXAMPLE ANALYSIS OF TARGET VIEW AND BOARD STATE
    D2's target view:
    "D2": {
      "row_0": [
        {"color": "blue", "size": 1},
        {"color": "orange", "size": 2},
        {"color": "orange", "size": 2}
      ],
      "row_1": [
        {"color": "yellow", "size": 1},
        {"color": "yellow", "size": 1},
        {"color": "orange", "size": 1}
      ],
      "row_2": [
        {"color": "yellow", "size": 1},
        {"color": "blue", "size": 1},
        {"color": "green", "size": 1}
      ]
    }
    Current board state:
    {
        "(0,0)": [], "(0,1)": [], "(0,2)": [],
        "(1,0)": [], "(1,1)": [], "(1,2)": [],
        "(2,0)": [], "(2,1)": [], "(2,2)": []
    }
    Correct D2 analysis:
[From my perspective, the current board state has all cells empty. 
My target view specifies that layer 0 should have a blue small block 
in my bottom left corner (0,0), and then a large orange block spanning 
the middle and right cells (0,1) and (0,2). 

Going left to right, layer 1 should have two small yellow blocks at 
(0,0) and (0,1), and a small orange block at (0,2). 

Finally, layer 2 should consist of a yellow small block at (0,0), 
a blue small block at (0,1), and a green small block at (0,2).

To start, I need the builder to place a large orange block spanning 
(0,1) and (0,2), which are the middle and right cells of my bottom layer. 
This is the first action to align with my target view.]
   Correct D2 utterance based on this analysis:
    [Put a large orange block across the middle and the right side of my bottom layer.]
"""

OWN_JOB = """
    ### YOUR JOB
    - Look at your target view and compare it to the current board state
    - First, look for any blocks on the board that are ALREADY consistent with your private target view - DO NOT TALK ABOUT THESE
    - Then, figure out what the builder needs to do to make the board look correct from your perspective - that may involve placing new blocks or removing incorrectly placed ones
    - Talk naturally to your team, like a real person would - use a wide diversity of phrasings to communicate your meaning
"""
ALL_JOB = """
    ### YOUR JOB
    - Look at ALL THREE target views and compare them to the current board state
    - First, look for any blocks on the board that are ALREADY consistent with the target views - DO NOT TALK ABOUT THESE
    - Then, figure out what the builder needs to do to make the board consistent with all three views - that may involve placing new blocks or removing incorrectly placed ones
    - Talk naturally to your team, like a real person would - use a wide diversity of phrasings to communicate your meaning
"""

REASONING_RULES = """
    ### RULES FOR REASONING (private, in think tags)
    - Think step by step
    - Use BOTH coordinates and layer numbers to work out what's missing
    - Examine the current game board closely - do not ask the builder to put a block on a layer that has no support underneath it
    - Check if other directors already covered what you need
    - {complete_rule}
    - If someone else instructs the builder to do something that would destroy part of {wall_scope}, say so
    - If you want to remove a block, check the current board state to make sure that there is actually a block at that position (e.g., if the board is empty, you cannot suggest removing any blocks, if cell (0,0) has no blocks in it, you cannot suggest removing anything from that position)
"""

OWN_SPEAKING = """
    ### RULES FOR SPEAKING (public message)
 - Your message should do TWO things naturally in one or two sentences:
  1. Briefly describe what you currently see from YOUR side that others may not see
     (focus on YOUR unique cells — what's there, what's missing, what's wrong)
  2. Give ONE specific instruction based on that observation
  VERY IMPORTANT: YOU HAVE SPEAK IN WAYS THAT MAKE YOUR PERSONALITY SHINE THROUGH!
 
- The description should flow naturally into the instruction, like a human would say it.
- Focus your description on what ONLY YOU can see from your angle:
  D1's unique view: the left wall — prioritize describing what you see there
  D2's unique view: the back wall — prioritize describing what you see there  
  D3's unique view: the right wall — prioritize describing what you see there
- If another director already described something visible from your side too, acknowledge in short and 
and describe something only you can see instead.
- One combined message, max 35 words total.
"""
ALL_SPEAKING = """
    ### RULES FOR SPEAKING (public message)
 - Your message should do TWO things naturally in one or two sentences:
  1. Briefly describe the part of the structure your instruction is about (what's there, what's missing, what's wrong)
  2. Give ONE specific instruction based on that observation
  VERY IMPORTANT: YOU HAVE SPEAK IN WAYS THAT MAKE YOUR PERSONALITY SHINE THROUGH!
 
- The description should flow naturally into the instruction, like a human would say it.
- In this game EVERY Director can see ALL THREE walls, so there is no information only you hold.
  Do not repeat an instruction another director already gave this turn; if you agree with it, say so briefly
  or give the next most useful instruction instead.
- If your instruction is about a cell that is NOT on your own wall, say whose wall it is on and describe it
  in THAT director's frame, e.g. "on D3's wall, the bottom left" (D3's bottom left is (0,2) at layer 0).
- One combined message, max 35 words total.
"""

D_SIZE_EXAMPLE = """    GIVEN THE FOLLOWING VIEWS OF THE BOTTOM LAYER (row_0 in the JSON):
        D1's row_0:[{"color": "yellow", "size": 1}, {"color": "green", "size": 1}, {"color": "orange", "size": 1}]
        D2's row_0:[{"color": "blue", "size": 1}, {"color": "yellow", "size": 2}, {"color": "yellow", "size": 2}]
        D3's row_0:[{"color": "green", "size": 2}, {"color": "green", "size": 2}, {"color": "blue", "size": 1}]
VERY IMPORTANT, HERE ARE SOME NATURAL UTTERANCES WHOSE STYLE TO EMULATE:
    [So, the second layer on top of the yellow will be orange, and then another orange, and then a yellow.]
    [And then it'll go blue, yellow, green.]
    [On top of green there goes a blue. And on top of red there goes a yellow.]

VERY IMPORTANT, HERE ARE THE RULES FOR SPEAKING:
    - Use natural spatial language: "on top of the green one", "the corner near me",
    "next to the blue block", "bottom left", "stack another one there"
    - Never say coordinate numbers or layer numbers out loud
    - Never use block codes like 'gs' or 'ol' — say "small green" or "large orange"
    - Speak from YOUR OWN frame of reference. Use phrases such "my bottom right" to communicate this. For instance, if you are D1,     "my bottom left corner" is coordinate (0,0) at layer 0 and "my top right corner" is coordinate (2,0) at layer 2.    If you are D2, "my bottom left corner" is (0,0) at layer 0 and "my top right corner" is (0,2) at layer 2.     If you are D3, "my bottom left corner" is (0,2) at layer 0 and "my top right corner" is (2,2) at layer 2.
    - NEVER deviate from these frames of reference when giving instructions.
    - If the builder asked a clarification question in the previous turn, answer it directly
    at the start of your message before giving your instruction.
    - If the builder said a move failed, acknowledge it and suggest a correction.
 

    ### RESPONSE FORMAT

<think>
    [Your private reasoning — use coordinates freely here to work out what's needed]
    </think>

    <message>
    [Natural human speech only — no coordinates, no codes, no layer numbers]
    </message>
"""


def director_prompt(director_id: str, archetype: str, mode: str, views: Dict, board: Board,
                    conversation_history: str) -> str:
    all_mode = mode == "all"
    header = (f"You are Director {director_id} ({director_id}) in a collaborative LEGO construction task.\n"
              "    You are sitting around a physical board with a Builder and two other Directors.\n"
              "    From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.\n\n"
              f"    YOU ARE {archetype}\n\n"
              "    ### YOUR PERSONALITY\n"
              f"    {ARCHETYPES[archetype]}\n    \n"
              "    VERY IMPORTANT: YOU MUST ADOPT THIS PERSONALITY IN YOUR INTERNAL REASONING AND PUBLIC UTTERANCES\n\n"
              "    ### YOUR PERSPECTIVE\n"
              f"    {PERSPECTIVE[director_id]}\n")
    if all_mode:
        header += ("    That is YOUR OWN wall. In this game there is NO information separation: every Director, "
                   "including you, has the target views of ALL THREE walls.\n")
    interpret = D_INTERPRET + (ALL_INTERPRET_EXTRA if all_mode else "")
    reasoning = REASONING_RULES.format(
        complete_rule=("If all three views are already complete, say so briefly" if all_mode
                       else "If your view is already complete, say so briefly"),
        wall_scope="the structure" if all_mode else "your wall")
    if all_mode:
        target = ("    ### ALL THREE TARGET VIEWS (what the structure needs to look like from each wall)\n    "
                  + json.dumps(views))
    else:
        target = ("    ### YOUR TARGET VIEW (what YOU need the structure to look like from your side)\n    "
                  + json.dumps(views[director_id]))
    return (header + D_SPATIAL + interpret + D_EXAMPLE + (ALL_JOB if all_mode else OWN_JOB) + reasoning
            + (ALL_SPEAKING if all_mode else OWN_SPEAKING) + D_SIZE_EXAMPLE
            + "\n    ### CURRENT BOARD STATE (full — what is actually built right now)\n    "
            + json.dumps(board.stacks) + "\n\n" + target + "\n        \n    ### CONVERSATION SO FAR\n    "
            + conversation_history)


def _parse_director_response_core(text: str) -> Dict:
    """Port of DirectorAgent.parse_director_response (same fallback order, no debug prints)."""
    text = text or ""
    think = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
    msg = re.search(r"<message>\s*(.*?)\s*</message>", text, re.DOTALL | re.IGNORECASE)
    if think and msg:
        return {"internal_thinking": think.group(1).strip(), "public_message": msg.group(1).strip()}
    msg_open = re.search(r"<message>\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    if msg_open and not msg:
        return {"internal_thinking": think.group(1).strip() if think else "No thinking provided",
                "public_message": msg_open.group(1).strip()}
    if think and not msg:
        after = text[think.end():].strip()
        if after:
            return {"internal_thinking": think.group(1).strip(),
                    "public_message": re.sub(r"<message>", "", after, flags=re.IGNORECASE).strip()}
    if "<think>" in text.lower() and "</think>" not in text.lower():
        content = re.sub(r"<think>", "", text, flags=re.IGNORECASE).strip()
        lines = [l.strip() for l in content.split("\n") if l.strip() and len(l.split()) > 4]
        last = re.sub(r"<[^>]*$", "", lines[-1]).strip() if lines else "No message provided"
        return {"internal_thinking": content, "public_message": last}
    cleaned = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL).strip()
    seen, kept = set(), []
    for p in cleaned.split("\n\n"):
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            kept.append(p)
    cleaned = "\n\n".join(kept).strip()
    return {"internal_thinking": "No thinking provided", "public_message": cleaned or "No message provided"}


# Small models sometimes emit a malformed opening tag ("=message>", ":message>", "message>") that the ported
# CRAFT parser then passes through as part of the public message. It would reach the Builder's prompt, so it is
# stripped here (the unmodified text stays in the game log under "raw").
_TAG_HEAD = re.compile(r"^\s*(?:[<=:\[]+\s*/?\s*message\s*[>\]:]+|message\s*>+)\s*", re.IGNORECASE)
_TAG_TAIL = re.compile(r"\s*<\s*/?\s*message\s*>?\s*$", re.IGNORECASE)


def strip_tag_fragments(text: str) -> str:
    for _ in range(3):
        new = _TAG_TAIL.sub("", _TAG_HEAD.sub("", text or ""))
        if new == text:
            break
        text = new
    return text.strip()


def parse_director_response(text: str) -> Dict:
    out = _parse_director_response_core(text)
    cleaned = strip_tag_fragments(out["public_message"])
    out["tag_fragment_stripped"] = cleaned != out["public_message"].strip()
    out["public_message"] = cleaned or "No message provided"
    return out


SILENT = {"", "no message provided", "none", "null", "n/a"}


def is_silent(message: Optional[str]) -> bool:
    return (message or "").strip().lower() in SILENT


# --------------------------------------------------------------------------- #
# CRAFT Builder prompt (builder_agent.create_builder_prompt, BuilderType.Base)
# --------------------------------------------------------------------------- #
ALL_BUILDER_NOTE = """
In this game there is NO information separation: every Director can see all three walls.
A Director may therefore describe a cell on ANOTHER Director's wall by naming it (e.g. "on D3's wall, the bottom left").
When a Director names another Director's wall, interpret that part of the instruction in THAT wall owner's frame of reference.
"""


def format_oracle_moves(moves: List[Dict]) -> str:
    lines = []
    for m in moves:
        span = m.get("span_to")
        if m["action"] == "place":
            lines.append(f"  PLACE {m['block']} at {m['position']} layer {m['layer']}"
                         + (f" spanning to {span}" if span else ""))
        else:
            lines.append(f"  REMOVE from {m['position']} layer {m['layer']}" + (f" spanning to {span}" if span else ""))
    return "\n".join(lines)


def craft_builder_prompt(director_discussion: str, current_state: Dict, mode: str,
                         oracle_moves: Optional[List[Dict]] = None) -> str:
    oracle_section = ""
    if oracle_moves:
        oracle_section = f"""
            CANDIDATE MOVES (verified physically valid for this turn):
            {format_oracle_moves(oracle_moves)}

            From this list, select the move that you believe at least one director is asking for based on their discussion.
            If no candidate clearly matches what any director is describing, CLARIFY.
            """
    note = ALL_BUILDER_NOTE if mode == "all" else ""
    frame_extra = ("\nEXCEPTION: if the Director explicitly names another Director's wall, use that wall owner's frame "
                   "for that part of the instruction.\n" if mode == "all" else "")
    empty = json.dumps({c: [] for c in current_state}, indent=4)
    return f"""You are a Builder in a collaborative LEGO construction task.

The three Directors (D1, D2, and D3) have to instruct you to build a single structure that is consistent with the private views of the structure they have.
Your job is to place, move, or remove blocks on the board to build the structure.
From a top-down view of the target structure, D1's private view is of the left wall of the structure, D2's view is of the top wall of the structure, and D3's view of the right wall of the structure.
From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.
{note}

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
    {empty}
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
    AVAILABLE BLOCKS: {', '.join(BLOCKS)}

    {BLOCK_REFERENCE} 
    {COORD_REFERENCE}
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
{frame_extra}
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
        "(0,0)": [], "(0,1)": [], "(0,2)": [],
        "(1,0)": [], "(1,1)": [], "(1,2)": [],
        "(2,0)": [], "(2,1)": ["gl"], "(2,2)": ["gl"]
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
            "(0,0)": ["os"], "(0,1)": [], "(0,2)": [],
            "(1,0)": [], "(1,1)": [], "(1,2)": ["bl"],
            "(2,0)": ["gl", "bl"], "(2,1)": ["gl", "bl"], "(2,2)": ["bl"]
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


BUILDER_SYSTEM_BASE = (
    "You are a Builder in a collaborative LEGO task. "
    "Respond in the specified PLACE/REMOVE/CLARIFY format. "
    "In your CONFIRM field, write 2-3 sentences: which director(s) you are following, "
    "what the other directors said and whether they agreed or conflicted, "
    "and why you chose this move.")
BUILDER_SYSTEM_ORACLE = (
    "You are a Builder in a collaborative LEGO task. "
    "You have been given VERIFIED CANDIDATE MOVES — you MUST choose exactly one from the list. "
    "Respond in the specified PLACE/REMOVE/CLARIFY format. "
    "In your CONFIRM field, write 2-3 sentences: which director(s) you followed, "
    "whether others agreed or conflicted, and why you chose this candidate.")


# --------------------------------------------------------------------------- #
# Game loop
# --------------------------------------------------------------------------- #
def _format_history(dialogue: List[Dict], current_turn: int, current: List[Dict]) -> str:
    lines = []
    for d in dialogue:
        lines.append(f"Turn {d['turn']}:")
        lines += [f"  {m['director']}: {m['message']}" for m in d["messages"]]
        lines.append(f"  Builder: {d['builder']}")
    lines.append(f"Turn {current_turn} (now):")
    lines += [f"  {m['director']}: {m['message']}" for m in current]
    if not current:
        lines.append("  (you are the first to speak this turn)")
    return "\n".join(lines)


def _builder_utterance(move: Dict, res) -> str:
    """What the Directors hear from the Builder next turn."""
    from run_full_info_builder import format_move
    a = move["action"]
    if a == "clarify":
        return f"CLARIFY: {move.get('clarification', '').strip()}"
    if a == "parse_error":
        return "(the builder did not make a valid move this turn)"
    conf = move.get("confirmation", "").strip()
    if res is not None and res.ok:
        return f"Executed {format_move(move)}." + (f" {conf}" if conf else "")
    return f"Attempted {format_move(move)} but the move failed: {res.error if res else 'unknown error'}."


def play_game_directors(entry: Dict, structure_index: int, builder_backend, director_backend, cfg,
                        run: int, previousData, checkpoint=None) -> Dict:
    from run_full_info_builder import (  # lazy: avoids a circular import
        decide_move, build_turn_record, format_move, token_totals, _seed_for, config_dict, BackendDown,
        turn_backend_failed, is_generation_abort,
    )
    sid = entry["id"]
    mode = cfg.director_views
    target = Board.from_structure(entry)
    views = target_views(entry, cfg.views_source)
    board = Board()
    archetypes = {d: assign_archetype(structure_index, d, run, previousData) for d in DIRECTORS}
    print(f"  archetypes: {archetypes}  | director views: {mode}")
    dialogue: List[Dict] = []
    turns: List[Dict] = []
    fail_streak = 0
    t0 = time.time()

    for t in range(1, cfg.turns + 1):
        rng = random.Random(f"{sid}|{run}|{t}|{cfg.seed}|speakers")
        speakers = rng.sample(list(DIRECTORS), rng.randint(1, 3))

        # ---- Directors speak sequentially -------------------------------
        current, dir_records = [], []
        for d in speakers:
            hist = _format_history(dialogue, t, current)
            prompt = director_prompt(d, archetypes[d], mode, views, board, hist)
            messages = [{"role": "system", "content": f"You are Director {d} in a collaborative LEGO construction task."},
                        {"role": "user", "content": prompt}]
            seed = None if cfg.seed < 0 else _seed_for(sid, run, t * 10 + DIRECTORS.index(d) + 1, cfg.seed)
            text, usage, err = "", None, None
            for k in range(3):
                try:
                    text, usage = director_backend.chat(messages, None if seed is None else seed + k)
                    err = None
                    break
                except Exception as exc:                              # noqa: BLE001
                    err = str(exc)
                    if is_generation_abort(exc) and k < 2:
                        print(f"    [{d}: generation aborted by server; retrying with a new seed ({k + 1}/2)]")
                        continue
                    break
            parsed = parse_director_response(text)
            # A reply cut off by the token cap with no complete <message> is unfinished reasoning, not an
            # instruction: the ported parser would pass its last line on to the Builder. Treat it as silent.
            truncated = bool(usage) and "length" in (usage.get("finish_reason"), usage.get("done_reason"))
            if truncated and not re.search(r"<message>.*?</message>", text or "", re.DOTALL | re.IGNORECASE):
                rt = (usage or {}).get("reasoning_tokens") or 0
                print(f"    [{d}: reply truncated at max tokens ({(usage or {}).get('completion_tokens')}"
                      + (f", {rt} of them hidden reasoning" if rt else "") + "); ignored. "
                      + ("Lower --director-reasoning-effort or raise --director-max-tokens]" if rt else
                         "Raise --director-max-tokens (the paper used 2000 for GPT-series Directors)]"))
                parsed = {"internal_thinking": parsed["internal_thinking"], "public_message": "No message provided",
                          "tag_fragment_stripped": False}
            else:
                truncated = False
            silent = is_silent(parsed["public_message"])
            rec = {"director": d, "archetype": archetypes[d], "public_message": parsed["public_message"],
                   "internal_thinking": parsed["internal_thinking"], "raw": text, "silent": silent,
                   "tag_fragment_stripped": parsed.get("tag_fragment_stripped", False),
                   "truncated": truncated,
                   "usage": usage, "error": err}
            if cfg.save_prompts or t == 1:
                rec["prompt"] = prompt
            dir_records.append(rec)
            if not silent:
                current.append({"director": d, "message": parsed["public_message"]})
            print(f"    {d} ({archetypes[d][:5]}): {parsed['public_message'][:150]}")

        discussion = "\n".join(f"{m['director']}: {m['message']}" for m in current) \
            or "(no director gave an instruction this turn)"

        # ---- Builder -----------------------------------------------------
        cands = enumerate_oracle_moves(board, target)
        shown = sample_oracle_moves(cands, cfg.oracle_n, f"{sid}:{t}") if cfg.oracle_in_prompt else None
        prompt = craft_builder_prompt(discussion, board.stacks, mode, shown)
        system = BUILDER_SYSTEM_ORACLE if shown else BUILDER_SYSTEM_BASE
        decision = decide_move(builder_backend, system, prompt, cfg, {"place", "remove", "clarify"},
                               None if cfg.seed < 0 else _seed_for(sid, run, t, cfg.seed),
                               {"board": board, "target": target, "views": views}, "directors")
        move = decision["move"]
        board_before = board.copy()
        res = board.apply(move) if move["action"] in ("place", "remove") else None
        metrics = compute_metrics(board, target, views)

        rec = build_turn_record(t, move, res, cands, shown, board_before, board, metrics, decision, cfg)
        rec["speakers"] = speakers
        rec["directors"] = dir_records
        rec["director_usage"] = [r["usage"] for r in dir_records if r["usage"]]
        rec["discussion"] = discussion
        turns.append(rec)
        builder_said = _builder_utterance(move, res)
        dialogue.append({"turn": t, "messages": current, "builder": builder_said})
        if checkpoint:
            checkpoint({"structure_id": sid, "run": run, "partial": True, "config": config_dict(cfg),
                        "archetypes": archetypes, "dialogue": dialogue, "turns": turns})
        fail_streak = fail_streak + 1 if turn_backend_failed(move, dir_records) else 0
        limit = getattr(cfg, "max_backend_fail_turns", 0)
        if limit and fail_streak >= limit:
            raise BackendDown(f"{sid} run{run}: {fail_streak} consecutive turns with backend errors "
                              f"(builder: {str(move.get('error'))[:200]})")
        print(f"  [{sid} run{run} t{t:02d}] {format_move(move):<48} "
              f"{'ok ' if rec['executed'] else 'ERR' if move['action'] in ('place', 'remove') else '   '} "
              f"OP={metrics['progress']:.3f} CP={metrics['completion']:.3f} views={metrics['view_match']:.3f}")
        if cfg.verbose:
            if rec["error"]:
                print(f"    [error] {rec['error']}")
            if move["action"] == "clarify":
                print(f"    [clarify] {move.get('clarification', '')[:200]}")

    b_usage = token_totals([u for t in turns for u in (t.get("usage") or [])])
    d_usage = token_totals([u for t in turns for u in t.get("director_usage", [])])
    print(f"  [tokens {sid} run{run}] builder in={b_usage['input_tokens']:,} out={b_usage['output_total_tokens']:,}"
          f" | directors in={d_usage['input_tokens']:,} out={d_usage['output_total_tokens']:,}"
          f" (reasoning={d_usage['reasoning_tokens']:,})")
    return {
        "structure_id": sid,
        "structure_index": structure_index,
        "complexity": entry.get("complexity"),
        "run": run,
        "policy": f"directors[{mode}]:{cfg.director_backend}:{cfg.director_model} -> builder:{cfg.backend}:{cfg.model}",
        "config": config_dict(cfg),
        "archetypes": archetypes,
        "reference_strings": REFERENCE_SOURCE,
        "target": target.to_json(),
        "target_views": views,
        "min_moves_true_target": min_moves(target),
        "views_ceiling": views_ceiling(entry, views),
        "done_at": None,
        "final_board": board.to_json(),
        "final_metrics": turns[-1]["metrics"],
        "dialogue": dialogue,
        "turns": turns,
        "elapsed_sec": round(time.time() - t0, 1),
        "token_usage": b_usage,
        "token_usage_directors": d_usage,
    }
