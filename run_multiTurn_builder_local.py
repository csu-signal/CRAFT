#!/usr/bin/env python3
"""
run_multiTurn_builder_local.py
===========================

Offline / no-API-cost version of `run_single_builder.py`, running BOTH arms of
each divergence turn through the same local Builder, and now rolling each arm
forward for H turns (default 3) instead of one.

Rollout design
--------------
For every divergence turn t the harness runs two arms:

  * counterfactual arm — step 0 uses the utterance of the director the POLICY
                         chose
  * factual arm        — step 0 uses the utterance of the director HISTORY
                         chose (`turn["builderSelected"]`)

Each arm then continues for steps 1..H-1 (turns t+1 .. t+H-1) on ITS OWN board
and its own conversation. Two independent switches control those turns:

  --followon-source generate (default)
      New D1/D2/D3 utterances every follow-on turn, from the DirectorAgent
      prompt and parser (imported from --director-module, served by Ollama),
      conditioned on the arm's board, each director's target view, and the
      arm's own conversation. Archetypes come from CRAFT's D*_archetype so the
      directors keep their historical personalities. Directors speak in order
      D1, D2, D3 and each sees who spoke before it. This gives a continuous
      causal chain from the divergence onward.
  --followon-source history
      Replay the logged CRAFT D1-D3 messages for turn t+k.

  --followon-select all (default)
      The Builder hears every non-silent director, as in the original game.
  --followon-select best
      The Builder hears each director alone; the move with the highest
      whole-board IoU vs target is kept. This uses the target, so it is an
      ORACLE selection rule. history+best isolates compounding returns from
      director selection, with no generation in the loop.

In every mode the ONLY difference between the arms is the step-0 director
choice; everything downstream is the consequence of that choice.

Ground-truth checks
-------------------
At every step each arm's board is compared against:

  1. the target structure (`satisfaction_full`, as before), and
  2. the historical board after the same turn in CRAFT (the ground-truth
     trajectory): `iou_vs_ground_truth_board`, plus
     `progress_vs_ground_truth = satisfaction_full - ground_truth_satisfaction_full`,
     i.e. how far ahead of / behind what actually happened the arm is.

Caveats that matter for interpretation
--------------------------------------
  * Follow-on utterances were produced in response to the HISTORICAL board, not
    the arm's board. The further an arm drifts from history, the less well the
    instructions fit it. That is inherent to replaying logged dialogue.
  * CRAFT `oracle_moves` for turn t+k were enumerated for the historical board,
    so with --use-oracle they are only passed at step 0 (where both arms share
    the historical pre-move board). Follow-on steps never get oracle moves.
  * Rollouts stop early when the game ends (no CRAFT row for the next turn).
    Later steps are therefore computed on a shrinking subset of turns; the
    per-step `n` in the summary shows this. Compare step 2 against step 0
    restricted to the same turns (`final_full_horizon`) rather than pooled.
  * Cost: 2 arms x H steps LLM calls per divergence turn (6x the old
    single-arm harness at H=3). Smoke-test with --limit first.

Everything else (`create_builder_prompt`, `parse_builder_response`,
`_place_block`, `_remove_block`, `getParticalView`, `calculate_iou_board`) is
imported from `run_single_builder` rather than copied, so the two harnesses
cannot drift apart.

Setup
-----
    ollama pull qwen2.5:7b-instruct
    OLLAMA_CONTEXT_LENGTH=8192 ollama serve

Usage
-----
    # 1) Check which CRAFT field holds the follow-on director discussion:
    python run_multiTurn_builder_local.py --divergence-dir ... --structures ... \
        --out-dir ... --inspect-row

    # 2) Run:
    python run_multiTurn_builder_local.py \
        --divergence-dir /home/hannah/CRAFT/CRAFT/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4/validation \
        --structures    /home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json \
        --out-dir       /home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/local_qwen7b_seed42_h3 \
        --model         qwen2.5:7b-instruct \
        --horizon 3 \
        --limit 20
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import importlib
import io
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from datasets import load_dataset

# Reuse the real prompt/parse/physics/scoring code -- do not reimplement it.
from run_single_builder import (
    BuilderAgent,
    BuilderType,
    _place_block,
    _remove_block,
    calculate_iou_board,
    getParticalView,
    normalize_structure,
)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
AVAILABLE_BLOCKS = ["gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol"]
DIRECTORS = ("D1", "D2", "D3")

SILENT_MARKERS = {"", "no message provided", "none", "null", "n/a"}
MOVE_PREFIXES = ("PLACE:", "REMOVE:", "CLARIFY:")

# Candidate CRAFT column names. The first one present in the dataset is used.
# Run with --inspect-row to see the actual columns; override the utterance
# field with --followon-utterance-field if auto-detection picks the wrong one.
UTTERANCE_KEYS = (
    "director_discussion", "selected_utterance", "builder_selected_utterance",
    "director_utterance", "utterance", "discussion", "dialogue",
)
# Special value for --followon-utterance-field: build the follow-on discussion
# from the per-director columns D1_message / D2_message / D3_message, formatted
# "D1: ...\nD2: ...\nD3: ..." with silent directors left out. This is the
# default whenever those three columns exist (as in craft-benchmark-lean).
DIRECTOR_MESSAGES = "directors"
BOARD_AFTER_KEYS = ("structure_after", "structureAfter", "board_after", "gt_structure_after")
BOARD_BEFORE_KEYS = ("structure_before", "structureBefore", "board_before")


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_ollama(base_url: str, model: str, timeout: float = 10.0) -> None:
    """Fail loudly and early rather than after 400 turns of CLARIFY."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"[preflight] Cannot reach Ollama at {root} ({exc}).\n"
            f"            Start it with:  OLLAMA_CONTEXT_LENGTH=8192 ollama serve"
        )

    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        sys.exit(
            f"[preflight] Model '{model}' is not pulled. Available: {sorted(names) or '(none)'}\n"
            f"            Run:  ollama pull {model}"
        )
    print(f"[preflight] Ollama reachable at {root}; model '{model}' present.")


def warn_if_prompt_may_be_truncated(prompt: str, assumed_ctx: int = 4096) -> None:
    est = len(prompt) // 4
    if est > assumed_ctx * 0.75:
        print(
            f"[warn] builder prompt ~{est} tokens. If OLLAMA_CONTEXT_LENGTH is at its "
            f"default (~{assumed_ctx}) the prompt may be TRUNCATED. "
            f"Restart the server with OLLAMA_CONTEXT_LENGTH=8192."
        )


# --------------------------------------------------------------------------- #
# Local builder agent
# --------------------------------------------------------------------------- #
def extract_move_line(text: str) -> Optional[str]:
    """Pull the PLACE/REMOVE/CLARIFY line out of a local model's response."""
    if not text:
        return None

    cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().strip("[]").strip()
        line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line).strip()
        line = re.sub(r"^(?:[-*>]\s+|\d+[.)]\s+)", "", line)
        line = line.strip("*").strip()
        upper = line.upper()
        for prefix in MOVE_PREFIXES:
            if upper.startswith(prefix):
                return prefix + line[len(prefix):]

    match = re.search(r"\b(PLACE|REMOVE|CLARIFY):", cleaned, flags=re.IGNORECASE)
    if match:
        tail = cleaned[match.start():].splitlines()[0].strip().strip("[]").strip().rstrip("*")
        return match.group(1).upper() + tail[len(match.group(1)):]
    return None


def is_parse_failure(move: Dict) -> bool:
    """A real CLARIFY is a valid answer; a parser giving up is not."""
    if move.get("action") != "clarify":
        return False
    text = str(move.get("clarification", ""))
    return text.startswith("Could not parse response") or text.startswith("Parse error")


# --------------------------------------------------------------------------- #
# Oracle adherence
# --------------------------------------------------------------------------- #
def _norm_pos_str(s) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        try:
            parts = [int(p.strip()) for p in s.strip("()").split(",")]
            return "(" + ",".join(str(p) for p in parts) + ")"
        except ValueError:
            return s
    return s


def _coerce_layer(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def move_matches_candidate(move: Dict, candidate_move: Dict) -> bool:
    if move.get("action") != candidate_move.get("action"):
        return False
    if _norm_pos_str(move.get("position")) != _norm_pos_str(candidate_move.get("position")):
        return False
    if _coerce_layer(move.get("layer")) != _coerce_layer(candidate_move.get("layer")):
        return False
    if _norm_pos_str(move.get("span_to")) != _norm_pos_str(candidate_move.get("span_to")):
        return False
    if move.get("action") == "place" and move.get("block") != candidate_move.get("block"):
        return False
    return True


def _candidate_move_dict(cand: Dict) -> Dict:
    if "action" in cand:
        return cand
    return cand.get("move", cand)


def compute_oracle_adherence(move: Dict, oracle_moves: Optional[List[Dict]]) -> Optional[bool]:
    if not oracle_moves:
        return None
    if move.get("action") == "clarify":
        return None
    return any(move_matches_candidate(move, _candidate_move_dict(cand)) for cand in oracle_moves)


class LocalBuilderAgent(BuilderAgent):
    """BuilderAgent pointed at Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 400,
        seed: Optional[int] = 0,
        max_retries: int = 2,
        request_timeout: float = 300.0,
    ):
        # Deliberately not calling BuilderAgent.__init__ (it binds to api.openai.com).
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self.model_name = model_name
        self.oracle_moves = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.last_raw_text: str = ""
        self.last_attempts: int = 0

    def _chat(self, messages: List[Dict]) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        completion = self.client.chat.completions.create(**kwargs)
        if not completion or not completion.choices:
            return ""
        return (completion.choices[0].message.content or "").strip()

    def generate_move(
        self,
        director_discussion,
        current_state,
        available_blocks,
        oracle_moves=None,
        check_prompt_tokens: bool = False,
    ) -> Dict:
        self.last_raw_text = ""
        self.last_attempts = 0
        try:
            prompt = self.create_builder_prompt(
                director_discussion,
                current_state,
                available_blocks,
                oracle_moves=oracle_moves,
                builderPromptAddin=BuilderType.Base,
            )
            if check_prompt_tokens:
                warn_if_prompt_may_be_truncated(prompt)

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
            system_content += (
                " Output EXACTLY ONE line and nothing else: no preamble, no markdown, "
                "no code fences, no explanation after the line."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            last_move: Dict = {"action": "clarify", "clarification": "No response from local model"}
            for attempt in range(self.max_retries + 1):
                self.last_attempts = attempt + 1
                raw = self._chat(messages)
                self.last_raw_text = raw
                if not raw:
                    last_move = {"action": "clarify", "clarification": "Empty response from local model"}
                    continue

                line = extract_move_line(raw)
                if line is not None:
                    move = self.parse_builder_response(line)
                    if not is_parse_failure(move):
                        return move
                    last_move = move
                else:
                    last_move = {
                        "action": "clarify",
                        "clarification": f"Could not parse response: {raw[:200]}",
                    }

                if attempt < self.max_retries:
                    print(f"  [retry {attempt + 1}] unparseable output, re-prompting for format")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            "That was not in the required format. Reply with ONE line only, "
                            "starting with PLACE:, REMOVE: or CLARIFY:, using exactly:\n"
                            "PLACE:block_code:position:layer:CONFIRM:interpretation\n"
                            "PLACE:block_code:position:layer:span_to:CONFIRM:interpretation\n"
                            "REMOVE:position:layer:CONFIRM:interpretation\n"
                            "REMOVE:position:layer:span_to:CONFIRM:interpretation\n"
                            "CLARIFY:your question\n"
                            "No other text."
                        ),
                    })

            return last_move

        except Exception as exc:  # noqa: BLE001
            return {"action": "clarify", "clarification": f"Error in move generation: {exc}"}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_divergence_turns(folder: Path) -> List[Dict]:
    turns: List[Dict] = []
    files = sorted(folder.glob("*.json"))
    if not files:
        sys.exit(f"[data] No divergence logs found in {folder}")
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            turns.extend(json.load(fh))
    print(f"[data] Loaded {len(turns)} divergence turns from {len(files)} files.")
    return turns


def normalize_director_model(name: str) -> str:
    director = name.replace("-", "").lower()
    if director == "deepseeklite":
        director = "deepseekv2lite"
    return director


def director_label_from_combo(model_combo: str) -> str:
    return model_combo.split("+")[0].strip()


def load_decidable_keys(path: Path) -> set:
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return {
        (rec["turn_number"], rec["structure_id"], normalize_director_model(rec["director_model"]))
        for rec in records if rec["decidable"]
    }


def build_craft_index(dataset_name: str) -> Dict:
    ds = load_dataset(dataset_name)
    split = ds["train"]
    index: Dict = {}
    for row in split:
        key = (row["turn_number"], row["structure_id"], normalize_director_model(row["director_model"]))
        index.setdefault(key, row)
    print(f"[data] Indexed {len(index)} CRAFT rows from '{dataset_name}'.")
    return index


def is_silent(message_obj: Dict) -> bool:
    if not isinstance(message_obj, dict):
        return True
    utterance = (message_obj.get("utterance") or "").strip().lower()
    return utterance in SILENT_MARKERS


def parse_board(value):
    """Boards arrive as dicts/lists, JSON strings, or Python-repr strings."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def first_present(row: Dict, keys) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return k
    return None


def inspect_craft_row(index: Dict) -> None:
    """Print one CRAFT row's columns so the follow-on fields can be verified."""
    row = next(iter(index.values()))
    print("\n[inspect] CRAFT row columns (values truncated):")
    for k, v in row.items():
        s = str(v).replace("\n", "\\n")
        print(f"  {k:<32} {s[:140]}{'...' if len(s) > 140 else ''}")
    print(f"\n[inspect] auto-detected follow-on utterance field: {detect_followon_field(row)}")
    print(f"[inspect] auto-detected board-after field:         {first_present(row, BOARD_AFTER_KEYS)}")
    print(f"[inspect] auto-detected board-before field:        {first_present(row, BOARD_BEFORE_KEYS)}")
    if detect_followon_field(row) == DIRECTOR_MESSAGES:
        print("\n[inspect] example follow-on discussion fed to the Builder:")
        print(followon_utterance(row, DIRECTOR_MESSAGES))


def has_director_message_columns(row: Dict) -> bool:
    return all(f"{d}_message" in row for d in DIRECTORS)


def detect_followon_field(row: Dict) -> Optional[str]:
    if has_director_message_columns(row):
        return DIRECTOR_MESSAGES
    return first_present(row, UTTERANCE_KEYS)


def followon_utterance(row: Dict, field: str) -> Optional[str]:
    if field == DIRECTOR_MESSAGES:
        lines = []
        for d in DIRECTORS:
            msg = row.get(f"{d}_message")
            if isinstance(msg, (list, dict)):
                msg = json.dumps(msg)
            msg = (str(msg) if msg is not None else "").strip()
            if msg.lower() in SILENT_MARKERS:
                continue
            lines.append(f"{d}: {msg}")
        return "\n".join(lines) or None

    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        value = json.dumps(value)
    value = str(value).strip()
    return value or None


def ground_truth_board_after(
    index: Dict, turn: int, structure_id, director_model: str, after_key: Optional[str]
):
    """
    Historical board after `turn`. Prefer an explicit after-column on the row;
    otherwise use the NEXT turn's before-board. None if neither exists
    (typically the last turn of a game with no after-column).
    """
    row = index.get((turn, structure_id, director_model))
    if row is not None and after_key and row.get(after_key) not in (None, ""):
        return parse_board(row[after_key])
    nxt = index.get((turn + 1, structure_id, director_model))
    if nxt is not None:
        before_key = first_present(nxt, BOARD_BEFORE_KEYS)
        if before_key:
            return parse_board(nxt[before_key])
    return None


# --------------------------------------------------------------------------- #
# Step / rollout execution
# --------------------------------------------------------------------------- #
def execute_step(
    agent: LocalBuilderAgent,
    arm: str,
    step_idx: int,
    director: Optional[str],
    utterance: str,
    board,
    spans,
    target_norm: Dict,
    oracle_moves,
    check_tokens: bool = False,
) -> Tuple[Dict, object, object]:
    """
    One Builder move on `board`. Returns (result_dict, board_after, spans_after).
    `board` and `spans` belong to this arm only; the caller deep-copies them at
    the start of the rollout so arms never share state.
    """
    board_before = copy.deepcopy(board)

    move = agent.generate_move(
        director_discussion=utterance,
        current_state=copy.deepcopy(board),
        available_blocks=AVAILABLE_BLOCKS,
        oracle_moves=oracle_moves,
        check_prompt_tokens=check_tokens,
    )
    raw_text, attempts = agent.last_raw_text, agent.last_attempts
    print(f"    [{arm:<14} t+{step_idx}] {director or 'history'}: {move}")

    # NOTE: spans is carried forward as the same object. This assumes
    # _place_block/_remove_block update spans in place (as they do the board).
    # If they instead return updated spans in the third slot, use it here.
    success = False
    if move["action"] == "place":
        success, board, _ = _place_block(
            board, spans, move["position"], move["block"], move["layer"], move["span_to"]
        )
    elif move["action"] == "remove":
        success, board, _ = _remove_block(
            board, spans, move["position"], move["layer"], move["span_to"]
        )

    oracle_adherent = compute_oracle_adherence(move, oracle_moves)
    partial_views = {
        d: calculate_iou_board(normalize_structure(getParticalView(d, board)), target_norm)
        for d in DIRECTORS
    }

    result = {
        "step": step_idx,
        "director": director,
        "utterance": utterance,
        "structure_before": board_before,
        "structure_after": copy.deepcopy(board),
        "satisfaction_full_before": calculate_iou_board(normalize_structure(board_before), target_norm),
        "builder_move": move,
        "move_executed": success,
        "parse_failed": is_parse_failure(move),
        "oracle_candidate_count": len(oracle_moves) if oracle_moves else 0,
        "oracle_adherent": oracle_adherent,
        "satisfaction_full": calculate_iou_board(normalize_structure(board), target_norm),
        "satisfaction_partial": partial_views.get(director) if director in DIRECTORS else None,
        "partial_views": partial_views,
        "llm_attempts": attempts,
        "raw_model_text": raw_text,
    }
    return result, board, spans


# --------------------------------------------------------------------------- #
# Follow-on director turns
# --------------------------------------------------------------------------- #
FOLLOWON_SOURCES = ("generate", "history")
FOLLOWON_SELECTS = ("all", "best")
CONVERSATION_KEYS = ("conversation_snapshot", "conversation_history", "conversation")


def load_director_module(module_name: str):
    """Import the module that defines DirectorAgent (and get_director_views_fn)."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"[directors] Could not import '{module_name}' ({exc}).\n"
            f"            Pass --director-module <name> (the .py file that defines DirectorAgent, "
            f"without '.py') and run from the folder that also has structure_generator_v2.py "
            f"and task_progress_tracker.py."
        )
    if not hasattr(module, "DirectorAgent"):
        sys.exit(f"[directors] Module '{module_name}' has no DirectorAgent class.")
    return module


def make_local_director_class(DirectorAgent):
    class LocalDirectorAgent(DirectorAgent):
        """
        DirectorAgent served by Ollama. The prompt
        (create_enhanced_director_prompt_with_references) and the parser
        (parse_director_response) are inherited unchanged; only the transport
        differs.

        DirectorAgent.__init__ is deliberately not called: it needs torch, it
        builds a hosted-API client, and it re-derives the archetype from
        hash(). Here the archetype comes from the CRAFT row (D*_archetype) so
        the generated directors keep the personalities they had in the
        historical game.
        """

        def __init__(self, director_id, archetype, client, model_name,
                     temperature=0.7, max_tokens=500, seed=None, fallback_key=""):
            self.director_id = director_id
            self.archetype_source = "craft"
            if archetype not in DirectorAgent.ARCHETYPES:
                archetype = random.Random(f"{fallback_key}-{director_id}").choice(DirectorAgent.TYPES)
                self.archetype_source = "fallback"
            self.archetype = archetype
            self.personality = DirectorAgent.ARCHETYPES[archetype]
            self.use_api = True
            self.provider = "ollama"
            self.local_model = None
            self.local_tokenizer = None
            self.client = client
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.seed = seed
            self.last_prompt = ""
            self.last_raw = ""

        def build_prompt(self, current_view, target_view, conversation_history):
            return self.create_enhanced_director_prompt_with_references(
                current_view, target_view, conversation_history, AVAILABLE_BLOCKS
            )

        def generate_response(self, current_view, target_view, conversation_history, available_blocks):
            prompt = self.create_enhanced_director_prompt_with_references(
                current_view, target_view, conversation_history, available_blocks
            )
            self.last_prompt = prompt
            kwargs = dict(
                model=self.model_name,
                messages=[
                    # same system message as DirectorAgent._generate_with_openai
                    {"role": "system",
                     "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if self.seed is not None:
                kwargs["seed"] = self.seed
            try:
                completion = self.client.chat.completions.create(**kwargs)
                raw = ""
                if completion and completion.choices:
                    raw = completion.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                self.last_raw = ""
                return {"internal_thinking": f"Error in generation: {exc}",
                        "public_message": "No message provided", "raw_response": ""}
            self.last_raw = raw
            # parse_director_response prints a lot of debug output; keep the log readable
            with contextlib.redirect_stdout(io.StringIO()):
                return self.parse_director_response(raw)

    return LocalDirectorAgent


def get_views_fn(director_module):
    fn = getattr(director_module, "get_director_views_fn", None)
    if fn is None:
        try:
            from structure_generator_v2 import get_director_views as fn  # type: ignore
        except ImportError:
            fn = None
    return fn


def compute_target_views(views_fn, struct_entry: Dict) -> Dict:
    """Per-director target views in the row_0/row_1/row_2 format the director prompt expects."""
    for k in ("director_views", "target_views", "views"):
        v = struct_entry.get(k)
        if isinstance(v, dict) and all(d in v for d in DIRECTORS):
            return v
    if views_fn is None:
        raise RuntimeError("no get_director_views function available")
    views = views_fn(struct_entry["structure"])
    if isinstance(views, (list, tuple)) and len(views) == 3:
        views = dict(zip(DIRECTORS, views))
    if not (isinstance(views, dict) and all(d in views for d in DIRECTORS)):
        raise RuntimeError(f"get_director_views returned {type(views).__name__}, expected dict keyed D1/D2/D3")
    return views


def is_silent_text(text: Optional[str]) -> bool:
    return (text or "").strip().lower() in SILENT_MARKERS


def historical_director_messages(row: Dict) -> Dict[str, str]:
    out = {}
    for d in DIRECTORS:
        msg = row.get(f"{d}_message")
        out[d] = "" if msg is None else str(msg).strip()
    return out


def snapshot_to_text(value) -> str:
    """CRAFT conversation_snapshot -> plain 'speaker: text' lines."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in "[{":
            try:
                return snapshot_to_text(json.loads(s))
            except json.JSONDecodeError:
                pass
        return s
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                speaker = (item.get("speaker") or item.get("role") or item.get("director")
                           or item.get("name") or "?")
                text = (item.get("message") or item.get("content") or item.get("utterance")
                        or item.get("text") or json.dumps(item))
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(value)


_SNAPSHOT_WARNED = False


def base_conversation(row: Dict, conv_field: Optional[str]) -> str:
    """
    Conversation up to and including the directors' turn-t messages (before the
    Builder's turn-t response, which each arm supplies itself).
    """
    global _SNAPSHOT_WARNED
    convo = snapshot_to_text(row.get(conv_field)) if conv_field else ""
    lines = [convo] if convo else []
    already_included = False
    for d, text in historical_director_messages(row).items():
        if is_silent_text(text):
            continue
        if convo and text[:60] in convo:
            already_included = True
            continue
        lines.append(f"{d}: {text}")
    if already_included and not _SNAPSHOT_WARNED:
        _SNAPSHOT_WARNED = True
        print(f"[warn] '{conv_field}' already contains the divergence turn's director messages, so it "
              f"may also contain the HISTORICAL Builder reply to them. Check with "
              f"--preview-director-prompt; use --conversation-field none to drop it.")
    return "\n".join(lines)


def builder_line(move: Dict, executed: bool) -> str:
    action = move.get("action")
    note = (move.get("confirmation") or move.get("interpretation") or "").strip()
    if action == "clarify":
        return f"Builder: {move.get('clarification', '')}".strip()
    if action == "place":
        desc = f"placed {move.get('block')} at {move.get('position')} on layer {move.get('layer')}"
    else:
        desc = f"removed the block at {move.get('position')} on layer {move.get('layer')}"
    if move.get("span_to"):
        desc += f" spanning to {move.get('span_to')}"
    status = "" if executed else " -- MOVE FAILED, the board did not change"
    return f"Builder: {desc}{status}." + (f" {note}" if note else "")


def tail(text: str, max_chars: Optional[int]) -> str:
    if max_chars and len(text) > max_chars:
        return "...\n" + text[-max_chars:]
    return text


def generate_director_messages(directors: Dict, board, views: Dict, conversation: str,
                               max_history_chars: Optional[int]) -> Dict[str, Dict]:
    """
    New D1/D2/D3 utterances for this arm's board. Directors speak in order and
    each sees the ones who spoke before it this turn, matching the prompt's
    'check if other directors already covered what you need'.
    """
    out: Dict[str, Dict] = {}
    running = conversation
    for d in DIRECTORS:
        parsed = directors[d].generate_response(
            copy.deepcopy(board), views[d], tail(running, max_history_chars), AVAILABLE_BLOCKS
        )
        text = (parsed.get("public_message") or "").strip()
        out[d] = {
            "text": text,
            "thinking": parsed.get("internal_thinking"),
            "source": "generated",
            "archetype": directors[d].archetype,
        }
        print(f"      [gen {d} ({directors[d].archetype})] {text[:160]}")
        if not is_silent_text(text):
            running = f"{running}\n{d}: {text}" if running else f"{d}: {text}"
    return out


def run_followon_step(agent, arm, k, msgs, speaking, board, spans, target_norm, select):
    """Builder turn at a follow-on step under the chosen selection rule."""
    if select == "all":
        utterance = "\n".join(f"{d}: {msgs[d]['text']}" for d in speaking)
        step, board, spans = execute_step(agent, arm, k, None, utterance, board, spans, target_norm, None)
        step["selection"] = {"mode": "all", "speaking": speaking}
        return step, board, spans

    # select == "best": the Builder hears each director alone; keep the move
    # with the highest whole-board IoU vs target. Uses the target, so this is
    # an oracle selection rule, not something a deployed policy could do.
    cands = []
    for d in speaking:
        res, b, s = execute_step(
            agent, f"{arm}/{d}", k, d, f"{d}: {msgs[d]['text']}",
            copy.deepcopy(board), copy.deepcopy(spans), target_norm, None,
        )
        cands.append((d, res, b, s))
    best_i = max(range(len(cands)), key=lambda i: (cands[i][1]["satisfaction_full"], -i))
    d, step, board, spans = cands[best_i]
    top = step["satisfaction_full"]
    step["selection"] = {
        "mode": "best",
        "chosen": d,
        "speaking": speaking,
        "candidate_satisfaction": {c[0]: c[1]["satisfaction_full"] for c in cands},
        "candidate_moves": {c[0]: c[1]["builder_move"] for c in cands},
        "n_tied_best": sum(1 for c in cands if c[1]["satisfaction_full"] == top),
    }
    print(f"    [{arm:<14} t+{k}] best-of-{len(cands)} chose {d} (sat={top:.4f})")
    return step, board, spans


def run_rollout(
    agent: LocalBuilderAgent,
    arm: str,
    director: str,
    message_obj: Dict,
    row: Dict,
    target_norm: Dict,
    oracle_moves,
    craft_index: Dict,
    turn_index: int,
    structure_id,
    director_model: str,
    horizon: int,
    after_key: Optional[str],
    followon: Dict,
    check_tokens: bool = False,
) -> Dict:
    """
    Step 0: this arm's director utterance on the shared pre-move board.
    Steps 1..H-1 depend on followon['source'] / followon['select']:
      generate: new D1-D3 utterances from the DirectorAgent prompt, conditioned
                on THIS arm's board and conversation (continuous causal chain)
      history:  the logged CRAFT D1-D3 messages for turn t+k
      all:      Builder hears every non-silent director
      best:     Builder hears each director alone; best-scoring move is kept
    Every step is scored against the target and the historical board.
    """
    source, select = followon["source"], followon["select"]
    board = copy.deepcopy(parse_board(message_obj["structureBefore"]))
    spans = copy.deepcopy(json.loads(row["spans_before"]))
    steps: List[Dict] = []
    stop_reason = None
    conversation = base_conversation(row, followon["conversation_field"])
    directors = followon["make_directors"](row) if (source == "generate" and horizon > 1) else None

    for k in range(horizon):
        t = turn_index + k
        if k == 0:
            utterance = f"{director}: {message_obj['utterance']}"
            step, board, spans = execute_step(
                agent, arm, 0, director, utterance, board, spans,
                target_norm, oracle_moves, check_tokens=check_tokens,
            )
            step["selection"] = {"mode": "divergence", "chosen": director}
            new_lines = []
        else:
            if source == "history":
                frow = craft_index.get((t, structure_id, director_model))
                if frow is None:
                    stop_reason = f"game ended: no CRAFT row for turn {t}"
                    break
                msgs = {d: {"text": txt, "source": "history"}
                        for d, txt in historical_director_messages(frow).items()}
            else:
                if steps and steps[-1]["satisfaction_full"] >= 1.0:
                    stop_reason = "target complete"
                    break
                msgs = generate_director_messages(
                    directors, board, followon["target_views"], conversation,
                    followon["max_history_chars"],
                )
            speaking = [d for d in DIRECTORS if not is_silent_text(msgs[d]["text"])]
            if not speaking:
                stop_reason = f"all directors silent at turn {t}"
                break
            step, board, spans = run_followon_step(
                agent, arm, k, msgs, speaking, board, spans, target_norm, select,
            )
            step["director_messages"] = msgs
            new_lines = [f"{d}: {msgs[d]['text']}" for d in speaking]

        step["turn"] = t
        new_lines.append(builder_line(step["builder_move"], step["move_executed"]))
        conversation = "\n".join([c for c in [conversation] + new_lines if c])

        gt_after = ground_truth_board_after(craft_index, t, structure_id, director_model, after_key)
        if gt_after is not None:
            gt_norm = normalize_structure(gt_after)
            gt_sat = calculate_iou_board(gt_norm, target_norm)
            step["ground_truth_satisfaction_full"] = gt_sat
            step["iou_vs_ground_truth_board"] = calculate_iou_board(normalize_structure(board), gt_norm)
            step["progress_vs_ground_truth"] = step["satisfaction_full"] - gt_sat
        else:
            step["ground_truth_satisfaction_full"] = None
            step["iou_vs_ground_truth_board"] = None
            step["progress_vs_ground_truth"] = None

        print(
            f"    [{arm:<14} t+{k}] sat={step['satisfaction_full']:.4f}  "
            f"gt_sat={_fmt(step['ground_truth_satisfaction_full'])}  "
            f"iou_vs_gt={_fmt(step['iou_vs_ground_truth_board'])}"
        )
        steps.append(step)

    return {
        "arm": arm,
        "director": director,
        "silent_director": is_silent(message_obj),
        "followon_source": source,
        "followon_select": select,
        "horizon_requested": horizon,
        "horizon_reached": len(steps),
        "stop_reason": stop_reason,
        "steps": steps,
        "final_conversation": conversation,
    }


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def mean_sem(values: List[float]):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return mean, sem, n


def pack(values: List[float]) -> Dict:
    mean, sem, n = mean_sem(values)
    return {
        "n": n,
        "mean": None if mean is None else round(mean, 6),
        "sem": None if sem is None else round(sem, 6),
        "n_positive": sum(1 for v in values if v > 0),
        "n_zero": sum(1 for v in values if v == 0),
        "n_negative": sum(1 for v in values if v < 0),
    }


def print_table(title: str, endpoints: Dict[str, Dict], indent: str = "  ") -> None:
    print(f"{indent}{title}")
    print(f"{indent}  {'endpoint':<40}{'n':>5}{'mean':>12}{'sem':>11}{'+/0/-':>12}")
    for name, block in endpoints.items():
        sem = "n/a" if block["sem"] is None else f"{block['sem']:.5f}"
        mean = "n/a" if block["mean"] is None else f"{block['mean']:+.5f}"
        split = f"{block['n_positive']}/{block['n_zero']}/{block['n_negative']}"
        print(f"{indent}  {name:<40}{block['n']:>5}{mean:>12}{sem:>11}{split:>12}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired factual/counterfactual multi-turn Builder rollouts with a local Ollama model."
    )
    ap.add_argument("--divergence-dir", required=True, type=Path)
    ap.add_argument("--structures", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--dataset", default="Abhijnan/craft-benchmark-lean")
    ap.add_argument("--arms", choices=["both", "counterfactual", "factual"], default="both")
    ap.add_argument("--horizon", type=int, default=3,
                    help="Turns to roll out from each divergence point, including the divergence "
                         "turn itself (3 = turns t, t+1, t+2).")
    ap.add_argument("--followon-source", choices=FOLLOWON_SOURCES, default="generate",
                    help="Where follow-on director utterances come from. 'generate' (default): new "
                         "D1-D3 utterances from the DirectorAgent prompt on each arm's own board, a "
                         "continuous causal chain. 'history': replay the logged CRAFT messages.")
    ap.add_argument("--followon-select", choices=FOLLOWON_SELECTS, default="all",
                    help="What the Builder hears at follow-on turns. 'all' (default): every "
                         "non-silent director. 'best': each director alone, keeping the move with "
                         "the highest IoU vs target (oracle selection). history+best ablates "
                         "compounding returns from director selection alone.")
    ap.add_argument("--director-module", default="director_agent_local",
                    help="Module (file name without .py) that defines DirectorAgent.")
    ap.add_argument("--director-model", default=None,
                    help="Ollama model for generated directors (default: same as --model).")
    ap.add_argument("--director-temperature", type=float, default=0.7,
                    help="Matches DirectorAgent's API temperature.")
    ap.add_argument("--director-max-tokens", type=int, default=500)
    ap.add_argument("--conversation-field", default=None,
                    help="CRAFT column with the dialogue so far, used as director conversation "
                         "history. Auto-detected; pass 'none' to start from the divergence turn only.")
    ap.add_argument("--max-history-chars", type=int, default=6000,
                    help="Keep only the last N characters of conversation in director prompts, so "
                         "the prompt stays inside the Ollama context window. 0 = no cap.")
    ap.add_argument("--preview-director-prompt", action="store_true",
                    help="Print the D1 prompt for the first usable turn and exit (no model calls).")
    ap.add_argument("--followon-utterance-field", default=None,
                    help="(history source only) CRAFT column with the historical discussion. "
                         "Auto-detected; check with --inspect-row.")
    ap.add_argument("--inspect-row", action="store_true",
                    help="Print one CRAFT row's columns and the auto-detected fields, then exit.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0, help="Ollama sampling seed; -1 disables.")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--decidability-file", type=Path, default=None)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-silent", action="store_true")
    ap.add_argument("--use-oracle", action="store_true",
                    help="Pass CRAFT oracle_moves at step 0 only (see module docstring).")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if args.horizon < 1:
        sys.exit("--horizon must be >= 1")

    craft_index = build_craft_index(args.dataset)
    if args.inspect_row:
        inspect_craft_row(craft_index)
        return

    sample_row = next(iter(craft_index.values()))
    if not has_director_message_columns(sample_row):
        sys.exit("[data] CRAFT rows need D1_message/D2_message/D3_message "
                 "(turn-t conversation and the history follow-on source use them).")
    if args.horizon > 1:
        print(f"[data] Follow-on turns: source={args.followon_source}, select={args.followon_select}.")

    if args.conversation_field and args.conversation_field.lower() == "none":
        conversation_field = None
    else:
        conversation_field = args.conversation_field or first_present(sample_row, CONVERSATION_KEYS)
    print(f"[data] Director conversation history from CRAFT column: {conversation_field or '(none)'}")

    director_model_name = args.director_model or args.model
    LocalDirectorAgent = None
    views_fn = None
    director_client = None
    if args.followon_source == "generate" and args.horizon > 1:
        director_module = load_director_module(args.director_module)
        LocalDirectorAgent = make_local_director_class(director_module.DirectorAgent)
        views_fn = get_views_fn(director_module)
        director_client = OpenAI(base_url=args.base_url, api_key="ollama", timeout=300.0)
        print(f"[directors] Generating follow-on utterances with '{director_model_name}' "
              f"(DirectorAgent prompt from '{args.director_module}').")
    after_key = first_present(sample_row, BOARD_AFTER_KEYS)
    before_key = first_present(sample_row, BOARD_BEFORE_KEYS)
    if after_key:
        print(f"[data] Ground-truth board after each turn from column '{after_key}'.")
    elif before_key:
        print(f"[data] Ground-truth board after turn t from turn t+1's '{before_key}' "
              f"(last turn of each game will have no ground truth).")
    else:
        print("[warn] No board columns found in CRAFT; ground-truth trajectory metrics will be n/a.")

    run_cf = args.arms in ("both", "counterfactual")
    run_fa = args.arms in ("both", "factual")

    if not args.skip_preflight and not args.preview_director_prompt:
        preflight_ollama(args.base_url, args.model)
        if LocalDirectorAgent is not None and director_model_name != args.model:
            preflight_ollama(args.base_url, director_model_name)

    with open(args.structures, "r", encoding="utf-8") as fh:
        struct_data = json.load(fh)
    target_by_id = {s["id"]: s["structure"] for s in struct_data}
    struct_by_id = {s["id"]: s for s in struct_data}

    if LocalDirectorAgent is not None:
        # Fail now, not 20 turns in, if target views can't be built.
        try:
            compute_target_views(views_fn, struct_data[0])
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[directors] Could not build director target views ({exc}). "
                     f"Check get_director_views in structure_generator_v2.")

    def make_followon(row: Dict, struct_entry: Dict) -> Dict:
        target_views = None
        make_directors = None
        if LocalDirectorAgent is not None:
            target_views = compute_target_views(views_fn, struct_entry)

            def make_directors(r: Dict) -> Dict:
                return {
                    d: LocalDirectorAgent(
                        d, r.get(f"{d}_archetype"), director_client, director_model_name,
                        temperature=args.director_temperature,
                        max_tokens=args.director_max_tokens,
                        seed=None if args.seed < 0 else args.seed,
                        fallback_key=str(r.get("structure_id")),
                    )
                    for d in DIRECTORS
                }
        return {
            "source": args.followon_source,
            "select": args.followon_select,
            "conversation_field": conversation_field,
            "max_history_chars": args.max_history_chars or None,
            "target_views": target_views,
            "make_directors": make_directors,
        }

    turns = load_divergence_turns(args.divergence_dir)
    if not args.no_shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(turns)
        print(f"[data] Shuffled {len(turns)} turns with shuffle-seed={args.shuffle_seed}.")

    if args.decidability_file:
        decidable_keys = load_decidable_keys(args.decidability_file)
        before = len(turns)

        def is_decidable_turn(turn):
            cf_msg = turn.get(f"{turn['predicted']}_message")
            if cf_msg is None:
                return False
            key = (cf_msg["timestamp"], cf_msg["structure"],
                   normalize_director_model(cf_msg["modelCombo"].split("+")[0].strip()))
            return key in decidable_keys

        turns = [t for t in turns if is_decidable_turn(t)]
        print(f"[data] --decidability-file: kept {len(turns)}/{before} turns marked decidable.")

    if args.limit:
        turns = turns[: args.limit]

    agent = LocalBuilderAgent(
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=None if args.seed < 0 else args.seed,
        max_retries=args.max_retries,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_turns_counter: Dict[str, List[Dict]] = defaultdict(list)
    stats: Dict[str, int] = defaultdict(int)
    # step -> endpoint -> values
    deltas_by_step: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # step -> metric -> values (absolute, per arm, vs ground truth)
    gt_by_step: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # turns where both arms reached the full horizon: step0 vs final on the SAME turns
    full_horizon: Dict[str, List[float]] = defaultdict(list)
    deltas_by_director: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    deltas_by_move_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    horizon_reached_counts: Dict[int, int] = defaultdict(int)
    t_start = time.time()

    for count, turn in enumerate(turns, start=1):
        predicted = turn["predicted"]
        historical = turn.get("builderSelected")
        print(f"\nTurn {count}/{len(turns)}  policy={predicted}  history={historical}")

        if historical is None:
            print("  [skip] no builderSelected field — cannot pair a factual arm")
            stats["missing_history"] += 1
            continue
        if predicted == historical:
            print("  [skip] policy agreed with history — not a divergence turn")
            stats["non_divergence"] += 1
            continue

        cf_msg = turn.get(f"{predicted}_message")
        fa_msg = turn.get(f"{historical}_message")
        if cf_msg is None or fa_msg is None:
            print("  [skip] missing message object for one arm")
            stats["missing_message"] += 1
            continue

        cf_silent = is_silent(cf_msg)
        if cf_silent:
            stats["silent_director"] += 1
            print(f"  [silent] {predicted} produced no utterance this turn")
            if args.skip_silent:
                continue
        if is_silent(fa_msg):
            print(f"  [skip] historical director {historical} is silent — log inconsistency")
            stats["silent_history"] += 1
            continue

        turn_index = cf_msg["timestamp"]
        structure_id = cf_msg["structure"]
        director_model = normalize_director_model(cf_msg["modelCombo"].split("+")[0].strip())

        if fa_msg.get("structureBefore") != cf_msg.get("structureBefore"):
            print("  [skip] arms disagree on structureBefore — cannot pair")
            stats["board_mismatch"] += 1
            continue

        row = craft_index.get((turn_index, structure_id, director_model))
        if row is None:
            print(f"  [skip] no CRAFT row for {turn_index}, {structure_id}, {director_model}")
            stats["no_craft_row"] += 1
            continue

        target = target_by_id.get(row["structure_id"])
        if target is None:
            print(f"  [skip] no target structure for id {row['structure_id']}")
            stats["no_target"] += 1
            continue
        target_norm = normalize_structure(target)
        oracle_moves = json.loads(row["oracle_moves"]) if args.use_oracle else None
        director_label = director_label_from_combo(cf_msg["modelCombo"])

        record: Dict = {
            "turn": turn_index,
            "structure_id": row["structure_id"],
            "modelCombo": cf_msg["modelCombo"],
            "director_model_label": director_label,
            "counterfactualDirector": predicted,
            "historicalDirector": historical,
            "silent_director": cf_silent,
            "horizon": args.horizon,
            "followon_source": args.followon_source,
            "followon_select": args.followon_select,
            "director_generation_model": director_model_name if LocalDirectorAgent else None,
            "builder_backend": "ollama",
            "builder_model": args.model,
            "seed": args.seed,
            "shuffle_seed": None if args.no_shuffle else args.shuffle_seed,
        }

        try:
            followon = make_followon(row, struct_by_id[row["structure_id"]])
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] could not build director target views: {exc}")
            stats["no_target_views"] += 1
            continue

        if args.preview_director_prompt:
            if LocalDirectorAgent is None:
                sys.exit("--preview-director-prompt needs --followon-source generate and --horizon > 1.")
            d1 = followon["make_directors"](row)["D1"]
            board0 = parse_board(cf_msg["structureBefore"])
            convo = base_conversation(row, conversation_field)
            convo = "\n".join([c for c in [convo, "Builder: <this arm's turn-t move goes here>"] if c])
            print("\n" + "=" * 76)
            print(f"  D1 PROMPT PREVIEW  (turn {turn_index + 1}, archetype={d1.archetype}, "
                  f"source={d1.archetype_source})")
            print("=" * 76)
            print(d1.build_prompt(board0, followon["target_views"]["D1"],
                                  tail(convo, args.max_history_chars or None)))
            return

        rollout_kwargs = dict(
            row=row, target_norm=target_norm, oracle_moves=oracle_moves,
            craft_index=craft_index, turn_index=turn_index, structure_id=structure_id,
            director_model=director_model, horizon=args.horizon,
            after_key=after_key, followon=followon,
        )

        if run_cf:
            record["counterfactual"] = run_rollout(
                agent, "counterfactual", predicted, cf_msg,
                check_tokens=(count == 1), **rollout_kwargs,
            )
            for s in record["counterfactual"]["steps"]:
                stats[f"cf_action_{s['builder_move']['action']}"] += 1
                stats["cf_parse_failure"] += int(s["parse_failed"])

        if run_fa:
            # Same local Builder, same seed, same prompts, same follow-on
            # dialogue; only the step-0 director differs.
            record["factual"] = run_rollout(
                agent, "factual", historical, fa_msg, **rollout_kwargs,
            )
            for s in record["factual"]["steps"]:
                stats[f"fa_action_{s['builder_move']['action']}"] += 1
                stats["fa_parse_failure"] += int(s["parse_failed"])

        if run_cf and run_fa:
            cf_steps = record["counterfactual"]["steps"]
            fa_steps = record["factual"]["steps"]
            n_common = min(len(cf_steps), len(fa_steps))
            horizon_reached_counts[n_common] += 1

            per_step = []
            for k in range(n_common):
                c, f = cf_steps[k], fa_steps[k]
                d_full = c["satisfaction_full"] - f["satisfaction_full"]
                d_policy = c["partial_views"][predicted] - f["partial_views"][predicted]
                entry = {
                    "step": k,
                    "turn": c["turn"],
                    "full": d_full,
                    "partial_policy_view": d_policy,
                    "iou_vs_ground_truth_board": None,
                }
                deltas_by_step[k]["full_board"].append(d_full)
                deltas_by_step[k]["partial_policy_view"].append(d_policy)
                if not cf_silent:
                    deltas_by_step[k]["full_board_excluding_silent"].append(d_full)
                if c["iou_vs_ground_truth_board"] is not None and f["iou_vs_ground_truth_board"] is not None:
                    d_gt = c["iou_vs_ground_truth_board"] - f["iou_vs_ground_truth_board"]
                    entry["iou_vs_ground_truth_board"] = d_gt
                    deltas_by_step[k]["iou_vs_ground_truth_board"].append(d_gt)
                for arm_name, s in (("counterfactual", c), ("factual", f)):
                    if s["progress_vs_ground_truth"] is not None:
                        gt_by_step[k][f"{arm_name}_progress_vs_ground_truth"].append(s["progress_vs_ground_truth"])
                        gt_by_step[k][f"{arm_name}_iou_vs_ground_truth_board"].append(s["iou_vs_ground_truth_board"])
                per_step.append(entry)
                print(f"    t+{k}: delta_full={d_full:+.4f}  delta_policy_view={d_policy:+.4f}  "
                      f"delta_iou_vs_gt={_fmt(entry['iou_vs_ground_truth_board'])}")

            cf_action = cf_steps[0]["builder_move"]["action"]
            fa_action = fa_steps[0]["builder_move"]["action"]
            move_type_key = f"{cf_action}/{fa_action}"
            reached_full = n_common == args.horizon

            record["move_type"] = move_type_key
            record["delta"] = {
                "steps_compared": n_common,
                "full_horizon": reached_full,
                "progress_before": cf_steps[0]["satisfaction_full_before"],
                "per_step": per_step,
            }

            step0, final = per_step[0], per_step[-1]
            for bucket in (deltas_by_director[director_label], deltas_by_move_type[move_type_key]):
                bucket["step0_full_board"].append(step0["full"])
                bucket["step0_partial_policy_view"].append(step0["partial_policy_view"])
                if reached_full:
                    bucket["final_full_board"].append(final["full"])
                    bucket["final_partial_policy_view"].append(final["partial_policy_view"])
                    if final["iou_vs_ground_truth_board"] is not None:
                        bucket["final_iou_vs_ground_truth_board"].append(final["iou_vs_ground_truth_board"])

            if reached_full:
                full_horizon["step0_full_board"].append(step0["full"])
                full_horizon["final_full_board"].append(final["full"])
                full_horizon["step0_partial_policy_view"].append(step0["partial_policy_view"])
                full_horizon["final_partial_policy_view"].append(final["partial_policy_view"])
                if final["iou_vs_ground_truth_board"] is not None:
                    full_horizon["final_iou_vs_ground_truth_board"].append(final["iou_vs_ground_truth_board"])
                for arm_name, steps in (("counterfactual", cf_steps), ("factual", fa_steps)):
                    p = steps[-1]["progress_vs_ground_truth"]
                    if p is not None:
                        full_horizon[f"final_{arm_name}_progress_vs_ground_truth"].append(p)

        key = f"{cf_msg['modelCombo']}_{row['structure_id']}"
        all_turns_counter[key].append(record)
        stats["scored"] += 1

    # ── Write ──────────────────────────────────────────────────────────────
    for key, rows in all_turns_counter.items():
        out_path = args.out_dir / f"{key}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                rows, fh, indent=2,
                default=lambda obj: obj.to_dict() if hasattr(obj, "to_dict") else str(obj),
            )

    processed = sum(len(v) for v in all_turns_counter.values())
    elapsed = time.time() - t_start
    silent_rate = stats["silent_director"] / max(len(turns), 1)

    summary = {
        "builder_backend": "ollama",
        "builder_model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "arms": args.arms,
        "horizon": args.horizon,
        "followon_source": args.followon_source,
        "followon_select": args.followon_select,
        "director_generation_model": director_model_name if LocalDirectorAgent else None,
        "director_module": args.director_module if LocalDirectorAgent else None,
        "conversation_field": conversation_field,
        "ground_truth_board_source": after_key or (f"next turn's {before_key}" if before_key else None),
        "decidability_file": str(args.decidability_file) if args.decidability_file else None,
        "use_oracle": args.use_oracle,
        "divergence_dir": str(args.divergence_dir),
        "turns_seen": len(turns),
        "turns_scored": processed,
        "games_written": len(all_turns_counter),
        "steps_compared_histogram": dict(sorted(horizon_reached_counts.items())),
        "silent_director_count": stats["silent_director"],
        "silent_director_rate": round(silent_rate, 4),
        "skip_silent": args.skip_silent,
        "parse_failures": {"counterfactual": stats["cf_parse_failure"], "factual": stats["fa_parse_failure"]},
        "action_counts": {k: v for k, v in stats.items() if "_action_" in k},
        "skipped": {
            k: stats[k] for k in (
                "missing_history", "non_divergence", "missing_message",
                "silent_history", "board_mismatch", "no_craft_row", "no_target",
                "no_target_views",
            ) if stats[k]
        },
        "elapsed_sec": round(elapsed, 1),
        "sec_per_turn": round(elapsed / max(processed, 1), 2),
    }

    if run_cf and run_fa:
        summary["paired_delta_by_step"] = {
            f"t+{k}": {name: pack(vals) for name, vals in endpoints.items()}
            for k, endpoints in sorted(deltas_by_step.items())
        }
        summary["ground_truth_by_step"] = {
            f"t+{k}": {name: pack(vals) for name, vals in metrics.items()}
            for k, metrics in sorted(gt_by_step.items())
        }
        summary["full_horizon_turns"] = {name: pack(vals) for name, vals in full_horizon.items()}
        summary["paired_delta_by_director"] = {
            label: {name: pack(vals) for name, vals in endpoints.items()}
            for label, endpoints in sorted(deltas_by_director.items())
        }
        summary["paired_delta_by_move_type"] = {
            label: {name: pack(vals) for name, vals in endpoints.items()}
            for label, endpoints in sorted(deltas_by_move_type.items())
        }

    with open(args.out_dir / f"_run_summary-{args.out_dir.name}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 76)
    print(f"  LOCAL BUILDER ROLLOUT SUMMARY  (horizon={args.horizon}, "
          f"follow-on={args.followon_source}/{args.followon_select}; counterfactual − factual)")
    print("=" * 76)
    for k, v in summary.items():
        if not k.startswith(("paired_delta", "ground_truth_by_step", "full_horizon_turns")):
            print(f"  {k:<28} {v}")
    if "paired_delta_by_step" in summary:
        print("  " + "-" * 72)
        print("  PAIRED DELTA BY STEP")
        for step, endpoints in summary["paired_delta_by_step"].items():
            print_table(step, endpoints, indent="    ")
        print("  " + "-" * 72)
        print("  EACH ARM vs GROUND-TRUTH TRAJECTORY BY STEP")
        print("  (progress_vs_ground_truth > 0 means the arm is closer to the target than history was)")
        for step, metrics in summary["ground_truth_by_step"].items():
            print_table(step, metrics, indent="    ")
        print("  " + "-" * 72)
        print_table("TURNS REACHING FULL HORIZON (step 0 vs final, same turns)",
                    summary["full_horizon_turns"])
        print("  " + "-" * 72)
        print("  BY DIRECTOR MODEL")
        for label, endpoints in summary["paired_delta_by_director"].items():
            print_table(label, endpoints, indent="    ")
        print("  " + "-" * 72)
        print("  BY STEP-0 MOVE TYPE (counterfactual/factual)")
        for label, endpoints in summary["paired_delta_by_move_type"].items():
            print_table(label, endpoints, indent="    ")
    print("=" * 76)
    if silent_rate > 0.10:
        print(
            f"  ⚠ Silent-director rate {silent_rate:.1%} exceeds 10%. Compare 'full_board'\n"
            f"    against 'full_board_excluding_silent' before reading delta as policy quality."
        )
    print(f"  Wrote {len(all_turns_counter)} game files to {args.out_dir}")


if __name__ == "__main__":
    main()