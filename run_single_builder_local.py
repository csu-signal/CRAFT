#!/usr/bin/env python3
"""
run_single_builder_local.py
===========================

Offline / no-API-cost version of `run_single_builder.py`, running BOTH arms of
each divergence turn through the same local Builder.

For every divergence turn the harness runs:

  * counterfactual arm — the utterance of the director the POLICY chose
  * factual arm        — the utterance of the director HISTORY chose
                         (`turn["builderSelected"]`)

Both arms branch from the same pre-move board and go through the same model,
seed, prompt, execution and scoring, so `delta = counterfactual − factual` is
a paired within-turn estimate. Re-running the factual arm is not optional:
the logged historical `R_i` came from GPT-4o-mini, so reusing it would compare
two different Builders AND re-read `y_obs3`, making the experiment circular.

The Builder LLM is served by a local Ollama instance through its
OpenAI-compatible endpoint (http://localhost:11434/v1), so the only things
that change versus the GPT-4o harness are:

  1. `base_url` on the OpenAI client (plus a dummy api_key).
  2. A robustness layer around the response (local 7B/14B models add preamble,
     markdown fences and trailing prose far more often than GPT-4o does).

Everything else -- `create_builder_prompt`, `parse_builder_response`,
`_place_block`, `_remove_block`, `getParticalView`, `calculate_iou_board` --
is imported from `run_single_builder` rather than copied, so the two harnesses
can never drift apart.

Why no structured-output risk
-----------------------------
The handoff flags Ollama's inconsistent support for OpenAI's strict
`json_schema` mode as an open risk.  It does not apply on this path: the CRAFT
Builder prompt asks for a bare text line

    PLACE:block:position:layer[:span_to]:CONFIRM:interpretation
    REMOVE:position:layer[:span_to]:CONFIRM:interpretation
    CLARIFY:question

which needs no structured-output mechanism at all.  `format: json` is likewise
unnecessary and would actively hurt (the parser expects the text form).  The
risk only returns if you port `generate_move_with_tools`, which needs OpenAI
function-calling -- see NOTES at the bottom of this file.

Setup
-----
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen2.5:7b-instruct          # primary recommendation
    # ollama pull qwen2.5:14b-instruct       # fallback if 7B spatial reasoning is weak
    # ollama pull llama3.1:8b-instruct       # licensing / cross-model sanity check

    # IMPORTANT: the builder prompt is long (~2-3k tokens).  Ollama's default
    # context window will silently TRUNCATE it, which looks like a bad Builder
    # rather than a broken harness.  Start the server with a bigger window:
    OLLAMA_CONTEXT_LENGTH=8192 ollama serve

    # verify the OpenAI-compatible endpoint:
    curl http://localhost:11434/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"say OK"}]}'

Usage
-----
    python run_single_builder_local.py \
        --divergence-dir /home/hannah/CRAFT/CRAFT/divergenceData_policy_seed42_kl0.1_closs0.01_nosamp_hpt0.4/validation \
        --structures    /home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json \
        --out-dir       /home/hannah/CRAFT/CRAFT/gittenExperiments/counterfactuals/local_qwen7b_seed42 \
        --model         qwen2.5:7b-instruct \
        --arms both \
        --limit 20                     # smoke test first, then drop --limit

`--arms both` (the default) is the only setting that produces a delta;
`counterfactual` or `factual` alone are for re-running a single side.

Differences from run_single_builder.py that are deliberate fixes
----------------------------------------------------------------
  * The factual arm exists at all — the original only ran the policy's choice.
  * `copy.deepcopy` on the pre-move board. `_place_block`/`_remove_block`
    mutate in place, so the original recorded `structure_before` and
    `structure_after` as the same aliased object, and two arms would have
    contaminated each other.
  * Output files are opened "w", not "a".  The original appends, so a second
    run concatenates two JSON arrays into one file and produces invalid JSON.
  * The HF dataset is indexed ONCE into a dict instead of `ds.filter(...)`
    per turn (that was a full-dataset scan per divergence turn).
  * Silent-director turns are detected and flagged (handoff priority #1): if
    the policy's predicted director said nothing that turn, the counterfactual
    is a guaranteed no-op. They are recorded with `"silent_director": true`,
    reported as a rate, and excluded from a second delta estimate so the two
    can be compared. `--skip-silent` drops them entirely.
  * Raw model text is saved per turn so local-model parse failures can be
    audited instead of silently becoming CLARIFY.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import random

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

SILENT_MARKERS = {"", "no message provided", "none", "null", "n/a"}
MOVE_PREFIXES = ("PLACE:", "REMOVE:", "CLARIFY:")


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
    # Ollama reports 'qwen2.5:7b-instruct'; tolerate an implicit ':latest'.
    if model not in names and f"{model}:latest" not in names:
        sys.exit(
            f"[preflight] Model '{model}' is not pulled. Available: {sorted(names) or '(none)'}\n"
            f"            Run:  ollama pull {model}"
        )
    print(f"[preflight] Ollama reachable at {root}; model '{model}' present.")


def warn_if_prompt_may_be_truncated(prompt: str, assumed_ctx: int = 4096) -> None:
    """
    Rough chars/4 token estimate.  Ollama truncates silently, and a truncated
    builder prompt degrades moves in a way that is easy to misread as
    'the local model is bad at spatial reasoning'.
    """
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
    """
    Pull the PLACE/REMOVE/CLARIFY line out of a local model's response.

    GPT-4o reliably emits the bare line, so the original code takes
    `text.split('\\n')[0]`.  Local models routinely wrap it in ```fences```,
    prepend "Sure, here's my move:", or append a paragraph of justification.
    This scans every line and takes the first genuine move line.
    """
    if not text:
        return None

    cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().strip("[]").strip()
        line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line).strip()      # **bold**
        line = re.sub(r"^(?:[-*>]\s+|\d+[.)]\s+)", "", line)        # list bullets
        line = line.strip("*").strip()
        upper = line.upper()
        for prefix in MOVE_PREFIXES:
            if upper.startswith(prefix):
                return prefix + line[len(prefix):]                  # normalize verb case

    # Second pass: the move line exists but is prefixed inline, e.g.
    # "Final move: PLACE:bs:(0,0):0:CONFIRM:..."
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
# Oracle adherence: did the parsed move match one of the CRAFT-verified
# candidates offered this turn?
# --------------------------------------------------------------------------- #
def _norm_pos_str(s) -> Optional[str]:
    """Normalizes '(0, 0)' / '( 0,0 )' -> '(0,0)', matching oracle.py's _norm_pos
    exactly, so formatting differences never cause a false non-match."""
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
    """Defensive int coercion -- guards against a str-vs-int mismatch between
    the Builder's parsed move and the oracle candidate silently making every
    comparison report False."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def move_matches_candidate(move: Dict, candidate_move: Dict) -> bool:
    """
    True if the Builder's parsed move exactly matches one oracle candidate's
    move fields. block is only compared for 'place' -- oracle.py's own
    comments mark block as "informational only" for 'remove' candidates
    (_validate_move ignores it there), so comparing it for removes would
    produce false non-matches on an otherwise-correct removal.
    """
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


def compute_oracle_adherence(move: Dict, oracle_moves: Optional[List[Dict]]) -> Optional[bool]:
    """
    True/False if oracle_moves is a non-empty candidate list (a real
    constraint was in effect this turn) and the move is place/remove.
    None when there's nothing to adhere TO or nothing meaningful to compare:
      - oracle_moves is None or [] -- no constraint was in effect this turn
        (mirrors generate_move's own `if oracle_moves` fallback to the base,
        non-oracle system prompt for the same empty-list case).
      - the parsed move is 'clarify' -- CRAFT's oracle never enumerates a
        clarify option, so clarifying isn't "selecting the wrong candidate",
        it's a different category of response worth tracking separately
        rather than folding into True/False.
    """
    if not oracle_moves:
        return None
    if move.get("action") == "clarify":
        return None
    return any(move_matches_candidate(move, _candidate_move_dict(cand)) for cand in oracle_moves)
    
    
def _candidate_move_dict(cand: Dict) -> Dict:
    """
    oracle_moves entries may be the raw move dict itself, or a wrapper like
    oracle.py's enumerate_correct_actions() output ({'move': {...}, ...}).
    Handle both rather than assuming one -- an unconfirmed assumption here
    silently produces a 0% adherence rate regardless of the model's actual
    choices, which is exactly the failure mode this is guarding against.
    """
    if "action" in cand:
        return cand
    return cand.get("move", cand)


class LocalBuilderAgent(BuilderAgent):
    """
    BuilderAgent pointed at Ollama's OpenAI-compatible endpoint.

    Prompt construction and response parsing are inherited unchanged, so a
    local run and a GPT-4o run differ only in which model produced the text.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "ollama",          # required by the SDK, ignored by Ollama
        temperature: float = 0.1,
        max_tokens: int = 400,            # >250: local models pad before the move line
        seed: Optional[int] = 0,
        max_retries: int = 2,
        request_timeout: float = 300.0,
    ):
        # NB: deliberately not calling BuilderAgent.__init__ -- it constructs an
        # OpenAI() client bound to api.openai.com.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self.model_name = model_name
        self.oracle_moves = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.last_raw_text: str = ""
        self.last_attempts: int = 0

    # -- the one call that actually differs ---------------------------------
    def _chat(self, messages: List[Dict]) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed       # Ollama honours seed on /v1
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
        """
        Same signature and same return shape as BuilderAgent.generate_move,
        plus a format-correction retry loop for local models.
        """
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
            # Local-model addendum: the single biggest failure mode is prose
            # around the move line, not a wrong move.
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

        except Exception as exc:                                   # noqa: BLE001
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
    """
    Human-readable director model label for grouping/reporting, e.g.
    "Gemma-9B + 4o-mini" -> "Gemma-9B". Unlike normalize_director_model,
    this keeps original casing/punctuation since it's for display and
    aggregation keys, not for matching against the CRAFT dataset.
    """
    return model_combo.split("+")[0].strip()
    
def load_decidable_keys(path: Path) -> set:
    """
    Load a decidability export (see export_decidability.py) into a set of
    (turn_number, structure_id, normalized_director_model) keys marked
    decidable, for filtering divergence turns before running any Ollama calls.
    """
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return {
        (rec["turn_number"], rec["structure_id"], normalize_director_model(rec["director_model"]))
        for rec in records if rec["decidable"]
    }

def build_craft_index(dataset_name: str) -> Dict:
    """
    Index CRAFT once by (turn_number, structure_id, normalized director_model).

    The original does ds.filter(...) inside the turn loop, i.e. a full scan per
    divergence turn. With a local Builder you will run this over far more turns,
    so the scan becomes the bottleneck rather than the LLM.
    """
    ds = load_dataset(dataset_name)
    split = ds["train"]
    index: Dict = {}
    for row in split:
        key = (row["turn_number"], row["structure_id"], normalize_director_model(row["director_model"]))
        index.setdefault(key, row)
    print(f"[data] Indexed {len(index)} CRAFT rows from '{dataset_name}'.")
    return index


def is_silent(message_obj: Dict) -> bool:
    """
    Handoff priority #1: `get_chosen` never lets HISTORY pick a silent
    director, but the policy scores all three arms. A counterfactual on a
    silent director is a guaranteed no-op and will drag the IoU comparison
    negative for reasons unrelated to decision quality.
    """
    if not isinstance(message_obj, dict):
        return True
    utterance = (message_obj.get("utterance") or "").strip().lower()
    return utterance in SILENT_MARKERS


# --------------------------------------------------------------------------- #
# Arm execution
# --------------------------------------------------------------------------- #
def run_arm(
    agent: "LocalBuilderAgent",
    arm: str,
    director: str,
    message_obj: Dict,
    row: Dict,
    target_norm: Dict,
    oracle_moves,
    check_tokens: bool = False,
) -> Dict:
    """
    Run ONE arm (factual or counterfactual) of a divergence turn.

    Both arms must go through this function so they share prompt, model, seed,
    execution and scoring exactly. The only thing that differs between them is
    which director's utterance is fed in.

    Returns a dict; on failure it carries {"error": ...} and no scores.
    """
    utterance = f"{director}: {message_obj['utterance']}"
    # deepcopy: _place_block / _remove_block mutate in place, and the two arms
    # must both start from the same untouched pre-move board. (The original
    # script aliased structure_before to structure_after for this reason.)
    board_before = json.loads(message_obj["structureBefore"].replace("'", '"'))
    board = copy.deepcopy(board_before)
    spans_before = json.loads(row["spans_before"])

    move = agent.generate_move(
        director_discussion=utterance,
        current_state=board,
        available_blocks=AVAILABLE_BLOCKS,
        oracle_moves=oracle_moves,
        check_prompt_tokens=check_tokens,
    )
    raw_text, attempts = agent.last_raw_text, agent.last_attempts
    print(f"    [{arm:<14}] {director}: {move}")

    success = False
    if move["action"] == "place":
        success, board, _ = _place_block(
            board, spans_before, move["position"], move["block"], move["layer"], move["span_to"]
        )
    elif move["action"] == "remove":
        success, board, _ = _remove_block(
            board, spans_before, move["position"], move["layer"], move["span_to"]
        )
        
    oracle_adherent = compute_oracle_adherence(move, oracle_moves)
    print(f"    [{arm:<14}]   oracle: {len(oracle_moves) if oracle_moves else 0} candidate(s), "
          f"adherent={oracle_adherent}")
        
    # Raw progress (whole-board IoU) before this arm's move, for reporting
    # alongside satisfaction_full (progress after the move).
    satisfaction_full_before = calculate_iou_board(normalize_structure(board_before), target_norm)

    # Score every director view, not just this arm's own, so the two arms can
    # be compared on a MATCHED view (handoff experiment 1 asks for whole-board
    # and per-director-view IoU).
    partial_views = {
        d: calculate_iou_board(normalize_structure(getParticalView(d, board)), target_norm)
        for d in ("D1", "D2", "D3")
    }

    return {
        "arm": arm,
        "director": director,
        "utterance": utterance,
        "silent_director": is_silent(message_obj),
        "structure_before": board_before,
        "structure_after": board,
        "satisfaction_full_before": satisfaction_full_before,
        "builder_move": move,
        "move_executed": success,
        "parse_failed": is_parse_failure(move),
        "oracle_moves": oracle_moves,
        "oracle_candidate_count": len(oracle_moves) if oracle_moves else 0,
        "oracle_adherent": oracle_adherent,
        "satisfaction_full": calculate_iou_board(normalize_structure(board), target_norm),
        "satisfaction_partial": partial_views[director],
        "partial_views": partial_views,
        "llm_attempts": attempts,
        "raw_model_text": raw_text,
    }


def mean_sem(values: List[float]):
    """Paired mean and standard error; SEM is None below n=2."""
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return mean, sem, n


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired factual/counterfactual Builder rollouts with a local Ollama model."
    )
    ap.add_argument("--divergence-dir", required=True, type=Path,
                    help="divergenceData/{split} directory written by write_divergence_logs")
    ap.add_argument("--structures", required=True, type=Path,
                    help="structures_dataset_20.json")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--dataset", default="Abhijnan/craft-benchmark-lean")
    ap.add_argument("--arms", choices=["both", "counterfactual", "factual"], default="both",
                    help="Which arm(s) to run. 'both' is the only setting that yields a valid delta.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0, help="Ollama sampling seed; -1 disables.")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="Seed for shuffling turn order before --limit is applied, so a small "
                         "--limit isn't just 'whichever divergence-log files sort first alphabetically'. "
                         "Fixed by default for reproducibility; pass a different value to get a "
                         "different random subsample of the same size.")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="Disable shuffling and use file order as-is (old behavior; mainly useful "
                         "for debugging a specific turn by re-running the full unshuffled set).")
    ap.add_argument("--decidability-file", type=Path, default=None,
                help="Decidability export from export_decidability.py, matched to THIS beta. "
                     "When given, only turns whose predicted-director triple is marked "
                     "decidable are run -- saves LLM calls on turns theory says no model can "
                     "beat chance on, and lets --limit draw from decidable turns directly "
                     "rather than filtering them out after the fact.")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N turns (smoke test).")
    ap.add_argument("--skip-silent", action="store_true",
                    help="Drop turns where the predicted director said nothing (default: keep but flag).")
    ap.add_argument("--use-oracle", action="store_true",
                    help="Pass CRAFT oracle_moves into the prompt (matches the oracle prompt variant).")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    run_cf = args.arms in ("both", "counterfactual")
    run_fa = args.arms in ("both", "factual")

    if not args.skip_preflight:
        preflight_ollama(args.base_url, args.model)

    with open(args.structures, "r", encoding="utf-8") as fh:
        struct_data = json.load(fh)
    target_by_id = {s["id"]: s["structure"] for s in struct_data}

    turns = load_divergence_turns(args.divergence_dir)
    if not args.no_shuffle:
        # load_divergence_turns reads files in sorted() order, and divergence
        # logs are often grouped one-file-per-game, which in turn tends to
        # group by director model. Without shuffling, a small --limit is a
        # biased sample (whichever director models' files sort first
        # alphabetically), not a random subsample -- this is what produced
        # the frozen-at-fixed-n groups (DeepSeek-Lite stuck at n=9, Gemma-9B
        # stuck at n=29) across the 20/50/100-turn checkpoints.
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(turns)
        print(f"[data] Shuffled {len(turns)} turns with shuffle-seed={args.shuffle_seed}.")
        
    if args.decidability_file:
        decidable_keys = load_decidable_keys(args.decidability_file)
        before = len(turns)

        def is_decidable_turn(turn):
            predicted = turn["predicted"]
            cf_msg = turn.get(f"{predicted}_message")
            if cf_msg is None:
                return False  # silent-director turn -- no message to key on, exclude
            key = (cf_msg["timestamp"], cf_msg["structure"],
                   normalize_director_model(cf_msg["modelCombo"].split("+")[0].strip()))
            return key in decidable_keys

        turns = [t for t in turns if is_decidable_turn(t)]
        print(f"[data] --decidability-file: kept {len(turns)}/{before} turns marked decidable.")

    
    if args.limit:
        turns = turns[: args.limit]

    craft_index = build_craft_index(args.dataset)

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
    deltas_full: List[float] = []
    deltas_partial_own: List[float] = []
    deltas_partial_cf_view: List[float] = []
    deltas_full_nonsilent: List[float] = []
    deltas_by_director: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    deltas_by_move_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
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
            # Not a divergence; a paired delta would be identically zero and
            # would dilute the estimate. write_divergence_logs should not emit
            # these, but guard anyway.
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
            # get_chosen should never let history pick a silent director; if it
            # happens the log is suspect, so drop the pair rather than score it.
            print(f"  [skip] historical director {historical} is silent — log inconsistency")
            stats["silent_history"] += 1
            continue

        turn_index = cf_msg["timestamp"]
        structure_id = cf_msg["structure"]
        director_model = normalize_director_model(cf_msg["modelCombo"].split("+")[0].strip())

        if fa_msg.get("structureBefore") != cf_msg.get("structureBefore"):
            # Both arms must branch from the SAME board or the comparison is
            # not a counterfactual at all.
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
            "builder_backend": "ollama",
            "builder_model": args.model,
            "seed": args.seed,
            "shuffle_seed": None if args.no_shuffle else args.shuffle_seed,
        }

        if run_cf:
            record["counterfactual"] = run_arm(
                agent, "counterfactual", predicted, cf_msg, row, target_norm,
                oracle_moves, check_tokens=(count == 1),
            )
            stats[f"cf_action_{record['counterfactual']['builder_move']['action']}"] += 1
            stats["cf_parse_failure"] += int(record["counterfactual"]["parse_failed"])

        if run_fa:
            # Factual arm: SAME local Builder, same seed, same prompt — only the
            # director's utterance changes. Re-running it is what makes the
            # comparison valid; reusing the logged historical R_i (produced by
            # GPT-4o-mini) would be circular and would also compare two
            # different Builders.
            record["factual"] = run_arm(
                agent, "factual", historical, fa_msg, row, target_norm, oracle_moves,
            )
            stats[f"fa_action_{record['factual']['builder_move']['action']}"] += 1
            stats["fa_parse_failure"] += int(record["factual"]["parse_failed"])

        if run_cf and run_fa:
            cf, fa = record["counterfactual"], record["factual"]
            d_full = cf["satisfaction_full"] - fa["satisfaction_full"]
            d_own = cf["satisfaction_partial"] - fa["satisfaction_partial"]
            # Matched view: both arms scored on the view of the director the
            # POLICY chose. This is the per-director-view endpoint from the
            # handoff and is the cleaner of the two partial comparisons,
            # because d_own changes the view along with the arm.
            d_cf_view = cf["partial_views"][predicted] - fa["partial_views"][predicted]
            record["delta"] = {
                "full": d_full,
                "partial_own_view": d_own,
                "partial_policy_view": d_cf_view,
                "raw_progress": {
                    # board_before is validated identical for both arms above,
                    # so either arm's satisfaction_full_before works here.
                    "before": cf["satisfaction_full_before"],
                    "after_counterfactual": cf["satisfaction_full"],
                    "after_factual": fa["satisfaction_full"],
                },
            }
            
            deltas_full.append(d_full)
            deltas_partial_own.append(d_own)
            deltas_partial_cf_view.append(d_cf_view)
            if not cf_silent:
                deltas_full_nonsilent.append(d_full)
            
            deltas_by_director[director_label]["full"].append(d_full)
            deltas_by_director[director_label]["partial_own_view"].append(d_own)
            deltas_by_director[director_label]["partial_policy_view"].append(d_cf_view)
            if not cf_silent:
                deltas_by_director[director_label]["full_excluding_silent"].append(d_full)
                                
            cf_action = cf["builder_move"]["action"]
            fa_action = fa["builder_move"]["action"]
            move_type_key = f"{cf_action}/{fa_action}"   # e.g. "place/place", "remove/place", "clarify/place"

            deltas_by_move_type[move_type_key]["full"].append(d_full)
            deltas_by_move_type[move_type_key]["partial_own_view"].append(d_own)
            deltas_by_move_type[move_type_key]["partial_policy_view"].append(d_cf_view)
            if not cf_silent:
                deltas_by_move_type[move_type_key]["full_excluding_silent"].append(d_full)
                
            record["move_type"] = move_type_key
                
            print(f"    delta_full={d_full:+.4f}  delta_policy_view={d_cf_view:+.4f}")
            print(
                f"    progress: before={cf['satisfaction_full_before']:.4f}  "
                f"after_cf={cf['satisfaction_full']:.4f}  after_fa={fa['satisfaction_full']:.4f}"
            )
            avg_full = sum(deltas_full) / len(deltas_full)
            avg_policy_view = sum(deltas_partial_cf_view) / len(deltas_partial_cf_view)
            print(f"    running avg (n={len(deltas_full)}): delta_full={avg_full:+.4f}  "
                  f"delta_policy_view={avg_policy_view:+.4f}")

        key = f"{cf_msg['modelCombo']}_{row['structure_id']}"
        all_turns_counter[key].append(record)
        stats["scored"] += 1

    # ── Write ──────────────────────────────────────────────────────────────
    for key, rows in all_turns_counter.items():
        out_path = args.out_dir / f"{key}.json"
        with open(out_path, "w", encoding="utf-8") as fh:   # "w": the original "a" produced invalid JSON on re-runs
            json.dump(
                rows, fh, indent=2,
                default=lambda obj: obj.to_dict() if hasattr(obj, "to_dict") else str(obj),
            )

    processed = sum(len(v) for v in all_turns_counter.values())
    elapsed = time.time() - t_start
    silent_rate = stats["silent_director"] / max(len(turns), 1)

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

    summary = {
        "builder_backend": "ollama",
        "builder_model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "arms": args.arms,
        "decidability_file": str(args.decidability_file) if args.decidability_file else None,
        "use_oracle": args.use_oracle,
        "divergence_dir": str(args.divergence_dir),
        "turns_seen": len(turns),
        "turns_scored": processed,
        "games_written": len(all_turns_counter),
        "silent_director_count": stats["silent_director"],
        "silent_director_rate": round(silent_rate, 4),
        "skip_silent": args.skip_silent,
        "parse_failures": {"counterfactual": stats["cf_parse_failure"], "factual": stats["fa_parse_failure"]},
        "action_counts": {k: v for k, v in stats.items() if "_action_" in k},
        "skipped": {
            k: stats[k] for k in (
                "missing_history", "non_divergence", "missing_message",
                "silent_history", "board_mismatch", "no_craft_row", "no_target",
            ) if stats[k]
        },
        "elapsed_sec": round(elapsed, 1),
        "sec_per_turn": round(elapsed / max(processed, 1), 2),
    }
    
    if run_cf and run_fa:
        summary["paired_delta"] = {
            "full_board": pack(deltas_full),
            "full_board_excluding_silent": pack(deltas_full_nonsilent),
            "partial_own_view": pack(deltas_partial_own),
            "partial_policy_view": pack(deltas_partial_cf_view),
        }
        
        summary["paired_delta_by_director"] = {
            label: {
                "full_board": pack(vals["full"]),
                "full_board_excluding_silent": pack(vals.get("full_excluding_silent", [])),
                "partial_own_view": pack(vals["partial_own_view"]),
                "partial_policy_view": pack(vals["partial_policy_view"]),
            }
            for label, vals in sorted(deltas_by_director.items())
        }
        
        summary["paired_delta_by_move_type"] = {
            label: {
                "full_board": pack(vals["full"]),
                "full_board_excluding_silent": pack(vals.get("full_excluding_silent", [])),
                "partial_own_view": pack(vals["partial_own_view"]),
                "partial_policy_view": pack(vals["partial_policy_view"]),
            }
            for label, vals in sorted(deltas_by_move_type.items())
        }

    with open(args.out_dir / f"_run_summary-{args.out_dir.name}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 68)
    print("  LOCAL BUILDER RUN SUMMARY  (counterfactual − factual, same Builder)")
    if "paired_delta_by_director" in summary:
        print("  " + "-" * 64)
        print("  BY DIRECTOR MODEL")
        for label, endpoints in summary["paired_delta_by_director"].items():
            print(f"  {label}")
            print(f"    {'endpoint':<30}{'n':>5}{'mean':>12}{'sem':>11}{'+/0/-':>12}")
            for name, block in endpoints.items():
                sem = "n/a" if block["sem"] is None else f"{block['sem']:.5f}"
                mean = "n/a" if block["mean"] is None else f"{block['mean']:+.5f}"
                split = f"{block['n_positive']}/{block['n_zero']}/{block['n_negative']}"
                print(f"    {name:<30}{block['n']:>5}{mean:>12}{sem:>11}{split:>12}")
    if "paired_delta_by_move_type" in summary:
        print("  " + "-" * 64)
        print("  BY MOVE TYPE (counterfactual/factual)")
        for label, endpoints in summary["paired_delta_by_move_type"].items():
            print(f"  {label}")
            print(f"    {'endpoint':<30}{'n':>5}{'mean':>12}{'sem':>11}{'+/0/-':>12}")
            for name, block in endpoints.items():
                sem = "n/a" if block["sem"] is None else f"{block['sem']:.5f}"
                mean = "n/a" if block["mean"] is None else f"{block['mean']:+.5f}"
                split = f"{block['n_positive']}/{block['n_zero']}/{block['n_negative']}"
                print(f"    {name:<30}{block['n']:>5}{mean:>12}{sem:>11}{split:>12}")
    print("=" * 68)
    for k, v in summary.items():
        if k != "paired_delta":
            print(f"  {k:<28} {v}")
    if "paired_delta" in summary:
        print("  " + "-" * 64)
        print(f"  {'endpoint':<32}{'n':>5}{'mean':>12}{'sem':>11}{'+/0/-':>12}")
        for name, block in summary["paired_delta"].items():
            sem = "n/a" if block["sem"] is None else f"{block['sem']:.5f}"
            mean = "n/a" if block["mean"] is None else f"{block['mean']:+.5f}"
            split = f"{block['n_positive']}/{block['n_zero']}/{block['n_negative']}"
            print(f"  {name:<32}{block['n']:>5}{mean:>12}{sem:>11}{split:>12}")
    print("=" * 68)
    if silent_rate > 0.10:
        print(
            f"  ⚠ Silent-director rate {silent_rate:.1%} exceeds the 10% threshold from the\n"
            f"    handoff. Treat this as the leading explanation for any negative\n"
            f"    full-board delta BEFORE interpreting it as policy quality. Compare\n"
            f"    'full_board' against 'full_board_excluding_silent' above."
        )
    print(f"  Wrote {len(all_turns_counter)} game files to {args.out_dir}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# NOTES
# --------------------------------------------------------------------------- #
# 1. Both arms are re-run through the SAME local Builder, same seed, same
#    prompt; only the director utterance differs. This is what makes the delta
#    interpretable. Absolute IoU will not match the GPT-4o-mini historical
#    baseline (a weaker Builder builds worse boards regardless of who it
#    listened to), and reusing the logged historical R_i instead of re-running
#    would be circular — it would re-read y_obs3 and "confirm" delta by
#    construction.
#
# 2. Three delta endpoints are reported:
#      full_board          — whole-board IoU. Expected to be the weakest signal
#                            (estimated effect ~0.0015 IoU, likely underpowered
#                            at the available number of divergence turns).
#      partial_policy_view — both arms scored on the view of the director the
#                            POLICY chose. Matched comparison; this is the
#                            ~0.005-effect endpoint most likely to show signal.
#      partial_own_view    — each arm scored on its own director's view. The
#                            view changes with the arm, so a difference here
#                            mixes "better board" with "different window onto
#                            the board". Reported for completeness; prefer
#                            partial_policy_view.
#
# 3. Silent-director turns are counted, flagged, and excluded from
#    'full_board_excluding_silent'. If the two full-board figures diverge
#    materially, the student's negative chart is explained by silent picks
#    rather than by policy quality (handoff priority #1).
#
# 4. Tool mode (`generate_move_with_tools`) is NOT ported. It relies on OpenAI
#    function-calling; Ollama's /v1 supports tools for some models (qwen2.5
#    does) but multi-call budgeting and tool_call_id round-tripping behave
#    inconsistently. If you need it, subclass and override `_chat`, and test
#    the simulate_move schema against the target model before trusting a run.
#
# 5. Cost: 'both' doubles the LLM calls per divergence turn. At ~850 pooled
#    divergence turns that is ~1700 local generations — budget accordingly, and
#    smoke-test with --limit 20 first.
