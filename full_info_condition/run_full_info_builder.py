#!/usr/bin/env python3
"""
run_full_info_builder.py
========================

CRAFT without information separation: ONE Builder receives all three wall
views (D1 left, D2 far, D3 right) and must build a structure consistent with
all of them, within the usual 20-turn budget, starting from an empty board.
No Directors, no dialogue.

Backends (no third-party packages required; all HTTP is urllib):
  ollama           local, native /api/chat with num_ctx set per request
                   (avoids Ollama's silent prompt truncation at the default ctx)
  openai           any OpenAI-compatible /v1/chat/completions (OpenAI, vLLM, ...)
  gemini           Gemini via its OpenAI-compatible endpoint
  anthropic        Anthropic Messages API
  scripted-oracle  LLM-free, follows the true-target oracle  (harness check / upper bound)
  scripted-views   LLM-free, perfect reasoning from views only (information ceiling)

Quick start
-----------
  # 1. harness check, no model needed (seconds):
  python run_full_info_builder.py --structures structures_dataset_20.json \
      --out-dir runs/scripted_views --backend scripted-views

  # 2. local smoke test:
  ollama pull qwen2.5:7b-instruct
  python run_full_info_builder.py --structures structures_dataset_20.json \
      --out-dir runs/qwen7b_smoke --backend ollama --model qwen2.5:7b-instruct --limit 2

  # 3. API run (3 runs x 20 structures, as in the paper):
  export OPENAI_API_KEY=...
  python run_full_info_builder.py --structures structures_dataset_20.json \
      --out-dir runs/gpt4omini --backend openai --model gpt-4o-mini --runs 3

  # 4. Director mode (CRAFT game; every Director sees all 3 views). Off unless --directors is given:
  python run_full_info_builder.py --structures structures_dataset_20.json \
      --out-dir runs/dirs_all_qwen7b --backend ollama --model qwen2.5:7b-instruct --directors --limit 2
  #    control with standard information separation:  --director-views own
  #    CRAFT-style split models:  --backend openai --model gpt-4o-mini \
  #                               --director-backend ollama --director-model qwen2.5:7b-instruct

  # Builder assistance (mutually exclusive; default is neither):
  #    --oracle-in-prompt   up to 5 verified progress moves in the prompt (CRAFT main results)
  #    --sim-tool           CRAFT's simulate_move tool, --max-simulations calls per turn

See README_full_info.md for the design decisions and output schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# When run as a script this module is __main__; register it under its import name so the lazy
# imports in craft_directors / craft_sim_tool reuse THIS module (same classes, same BackendError)
# instead of loading a second copy.
if __name__ == "__main__":
    sys.modules.setdefault("run_full_info_builder", sys.modules[__name__])

from craft_full_info_env import (
    BLOCKS, CELLS, DIRECTORS, INTERIOR_CELLS, WALLS,
    Board, OraclePolicy, ViewsPolicy, classify_turn, compute_metrics,
    enumerate_oracle_moves, load_structures, min_moves, move_matches, norm_pos,
    project_all_views, sample_oracle_moves, start_board, target_views, views_ceiling,
)

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a Builder in a LEGO construction task. You work alone and are given all three "
    "wall views of the target structure. Respond in the specified PLACE/REMOVE/DONE format."
)
SINGLE_LINE_ADDENDUM = (
    " Output EXACTLY ONE line and nothing else: no preamble, no markdown, no code fences, "
    "no explanation after the line."
)
REASONING_ADDENDUM = (
    " Think step by step first. Then put your move on the LAST line, on its own, in the "
    "required format."
)

VIEW_GUIDE = """HOW TO READ THE VIEWS
- In each view, row_0 / row_1 / row_2 are LAYERS (vertical stack depth), NOT grid rows:
  row_0 = layer 0 (bottom), row_1 = layer 1 (middle), row_2 = layer 2 (top).
- Within a layer, the three entries are listed LEFT to RIGHT from that viewer's seat:
    D1 (left wall):  (0,0), (1,0), (2,0)
    D2 (far wall):   (0,0), (0,1), (0,2)
    D3 (right wall): (0,2), (1,2), (2,2)
- color=none means that cell is empty at that layer.
- size 2 = a large block whose BOTH cells lie on this wall; two adjacent size-2 entries of the
  same colour are ONE large block.
- size 1 = a small block, OR a large block whose other half is NOT on this wall. The same large
  block can look size 2 from one wall and size 1 from another.
- Corner (0,0) appears in both D1 and D2; corner (0,2) appears in both D2 and D3. A block there
  must agree with both views.
- Cells (1,1) and (2,1) appear in NO view. Nothing placed there is visible, and all three views
  can be satisfied without placing anything there."""

RULES = """BOARD AND PHYSICS
The coordinate grid from above:
  (0,0) (0,1) (0,2)   <- far / back row
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)   <- near / front row
- Each cell holds a stack of at most 3 blocks. Layer 0 is the bottom.
- Block codes: colour g/b/r/y/o (green/blue/red/yellow/orange) + size s (small) or l (large).
  Available: {blocks}
- A small block occupies one cell. A large block occupies TWO orthogonally adjacent cells on the
  SAME layer (sideways or forward/back, never vertical).
- PLACE only on top of a stack: the layer MUST equal the number of blocks currently at that cell
  (empty cell -> layer 0; cell with 2 blocks -> layer 2).
- A large block can be placed only if both cells currently have the SAME number of blocks.
  Give both endpoints: position and span_to.
- REMOVE only the top block of a stack (layer = number of blocks - 1). Removing a large block
  removes both halves and requires the correct span_to.
- One move per turn. A failed move still uses up the turn."""

OUTPUT_FORMAT = """OUTPUT FORMAT - choose ONE:
1. Place small block:  PLACE:block_code:position:layer:CONFIRM:reason
   Example: PLACE:bs:(0,0):0:CONFIRM:blue small at bottom of corner (0,0), seen by D1 and D2
2. Place large block:  PLACE:block_code:position:layer:span_to:CONFIRM:reason
   Example: PLACE:gl:(0,0):0:(1,0):CONFIRM:green large across D1's left and middle bottom cells
3. Remove small block: REMOVE:position:layer:CONFIRM:reason
4. Remove large block: REMOVE:position:layer:span_to:CONFIRM:reason
   NOTE: REMOVE never includes a block code.
{done_line}"""

DONE_LINE = ("5. Finish:            DONE:reason\n"
             "   Use DONE only when the board already matches all three views. It ends the game; "
             "the board is frozen and scored as is.")


def _annotated_views(views: Dict) -> str:
    """Same information as the JSON views, with global coordinates attached."""
    lines = []
    names = {"D1": "left wall", "D2": "far wall", "D3": "right wall"}
    for d in DIRECTORS:
        lines.append(f"{d} ({names[d]}), cells left to right: {', '.join(WALLS[d])}")
        for k in (2, 1, 0):
            cells = []
            row = views[d][f"row_{k}"]
            partners = {}
            i = 0
            while i < 2:   # pair adjacent size-2 entries of the same colour (one domino)
                if (int(row[i].get("size", 1)) == 2 and int(row[i + 1].get("size", 1)) == 2
                        and row[i]["color"] == row[i + 1]["color"]):
                    partners[i], partners[i + 1] = WALLS[d][i + 1], WALLS[d][i]
                    i += 2
                else:
                    i += 1
            for i, (c, e) in enumerate(zip(WALLS[d], row)):
                if e["color"] == "none":
                    cells.append(f"{c}=empty")
                elif i in partners:
                    cells.append(f"{c}={e['color']} large (with {partners[i]})")
                else:
                    cells.append(f"{c}={e['color']} small-or-large")
            lines.append(f"  layer {k}: " + " | ".join(cells))
    return "\n".join(lines)


def _json_views(views: Dict) -> str:
    """Compact JSON, one layer per line (valid JSON; about half the tokens of indent=1)."""
    lines = ["{"]
    for di, d in enumerate(DIRECTORS):
        rows = [f'    "row_{k}": {json.dumps(views[d][f"row_{k}"])}' for k in range(3)]
        lines.append(f'  "{d}": {{\n' + ",\n".join(rows) + "\n  }" + ("," if di < 2 else ""))
    lines.append("}")
    return "\n".join(lines)


def format_move(m: Dict) -> str:
    a = m.get("action")
    if a == "place":
        s = f"PLACE {m.get('block')} at {m.get('position')} layer {m.get('layer')}"
        return s + (f" spanning to {m['span_to']}" if m.get("span_to") else "")
    if a == "remove":
        s = f"REMOVE from {m.get('position')} layer {m.get('layer')}"
        return s + (f" spanning to {m['span_to']}" if m.get("span_to") else "")
    return a.upper() if a else "?"


def build_prompt(views: Dict, board: Board, history: List[str], turn: int, n_turns: int,
                 cfg: argparse.Namespace, shown_candidates: Optional[List[Dict]]) -> str:
    if cfg.view_format == "annotated":
        views_block = "TARGET VIEWS (what the finished structure must look like from each wall):\n" + _annotated_views(views)
    else:
        views_block = ("TARGET VIEWS (what the finished structure must look like from each wall):\n"
                       + _json_views(views))

    parts = [
        "You are a Builder in a LEGO construction task.",
        "There are NO Directors in this game. Instead you are given all three private wall views "
        "directly: D1's view of the left wall, D2's view of the far wall and D3's view of the right "
        "wall. Your job is to build, on the board, a single structure that is consistent with ALL "
        "THREE views.",
        "",
        RULES.format(blocks=", ".join(BLOCKS)),
        "",
        VIEW_GUIDE,
        "",
        views_block,
        "",
        "CURRENT BOARD STATE (each list is a stack, bottom block first):",
        json.dumps(board.stacks),
    ]
    if cfg.show_current_views:
        parts += ["", "CURRENT BOARD AS SEEN FROM EACH WALL (same format as the target views):",
                  _json_views(project_all_views(board))]
    if shown_candidates:
        parts += ["", "CANDIDATE MOVES (verified physically valid and making progress this turn):",
                  "\n".join("  " + format_move(c) for c in shown_candidates),
                  "Choose one of these candidates."]
    if cfg.history_window != 0:
        hist = history if cfg.history_window < 0 else history[-cfg.history_window:]
        parts += ["", "YOUR PREVIOUS TURNS:", "\n".join(hist) if hist else "(none - this is the first turn)"]
    parts += [
        "",
        f"TURN {turn} of {n_turns}. Turns remaining including this one: {n_turns - turn + 1}.",
        "",
        OUTPUT_FORMAT.format(done_line=DONE_LINE if cfg.allow_done else ""),
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Response extraction and parsing
# --------------------------------------------------------------------------- #
_POS = r"\(\s*\d\s*,\s*\d\s*\)"
_PLACE_RE = re.compile(
    rf"^PLACE\s*:\s*([A-Za-z]{{2}})\s*:\s*({_POS})\s*:\s*(\d+)\s*(?::\s*({_POS}))?\s*(?::\s*CONFIRM\s*:?(.*))?$",
    re.IGNORECASE)
_REMOVE_RE = re.compile(
    rf"^REMOVE\s*:\s*(?:[A-Za-z]{{2}}\s*:\s*)?({_POS})\s*:\s*(\d+)\s*(?::\s*({_POS}))?\s*(?::\s*CONFIRM\s*:?(.*))?$",
    re.IGNORECASE)
_DONE_RE = re.compile(r"^DONE\b\s*:?(.*)$", re.IGNORECASE)
_CLARIFY_RE = re.compile(r"^CLARIFY\s*:(.*)$", re.IGNORECASE)
_PREFIXES = ("PLACE:", "REMOVE:", "DONE", "CLARIFY:")


def _clean_line(raw: str) -> str:
    line = raw.strip().strip("`").strip().strip("[]").strip()
    line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line).strip()
    line = re.sub(r"^(?:[-*>]\s+|\d+[.)]\s+)", "", line)
    line = re.sub(r"^(?:final\s+)?(?:move|answer)\s*[:=-]\s*", "", line, flags=re.IGNORECASE)
    return line.strip("*").strip()


def extract_move_line(text: str, last: bool) -> Optional[str]:
    """
    Find the move line in a model response. Text before the last </think> is
    ignored. `last=False` takes the first move line (single-line mode);
    `last=True` takes the final one (reasoning mode).
    """
    if not text:
        return None
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"```[a-zA-Z]*", "\n", text).replace("```", "\n")
    found = []
    for raw in text.splitlines():
        line = _clean_line(raw)
        if line.upper().startswith(_PREFIXES):
            found.append(line)
            continue
        m = re.search(r"\b(PLACE|REMOVE)\s*:", line, flags=re.IGNORECASE)
        if m:
            found.append(line[m.start():].strip().rstrip("*`"))
    if not found:
        return None
    return found[-1] if last else found[0]


def parse_move(line: Optional[str]) -> Dict:
    if not line:
        return {"action": "parse_error", "error": "no move line found"}
    m = _PLACE_RE.match(line)
    if m:
        block = m.group(1).lower()
        span = norm_pos(m.group(4)) if m.group(4) else None
        move = {"action": "place", "block": block, "position": norm_pos(m.group(2)),
                "layer": int(m.group(3)), "span_to": span, "confirmation": (m.group(5) or "").strip()}
        if span and not block.endswith("l"):
            move["note"] = "span_to ignored for small block"
            move["span_to"] = None
        return move
    m = _REMOVE_RE.match(line)
    if m:
        return {"action": "remove", "position": norm_pos(m.group(1)), "layer": int(m.group(2)),
                "span_to": norm_pos(m.group(3)) if m.group(3) else None,
                "confirmation": (m.group(4) or "").strip()}
    m = _DONE_RE.match(line)
    if m:
        return {"action": "done", "reason": m.group(1).strip()}
    m = _CLARIFY_RE.match(line)
    if m:
        return {"action": "clarify", "clarification": m.group(1).strip()}
    return {"action": "parse_error", "error": f"unparseable move line: {line[:200]}"}


# --------------------------------------------------------------------------- #
# HTTP backends
# --------------------------------------------------------------------------- #
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class BackendError(RuntimeError):
    def __init__(self, msg, status=None, body=""):
        super().__init__(msg)
        self.status, self.body = status, body


HTTP_MAX_ATTEMPTS = 6        # per request, with exponential backoff 2,4,8,16,32 s (--http-retries)


class BackendDown(RuntimeError):
    """The model server kept failing; the run is stopped so it does not burn minutes per call."""


def is_generation_abort(exc: BaseException) -> bool:
    """Server killed one generation (Ollama: 'token repeat limit reached'). Re-sending the SAME request fails the
    same way, so callers retry with a different seed instead of waiting out the HTTP backoff."""
    return (isinstance(exc, BackendError) and exc.status == 500
            and "repeat limit" in (exc.body or str(exc)).lower())


def turn_backend_failed(move: Dict, dir_records: Optional[List[Dict]] = None) -> bool:
    if move.get("action") == "parse_error" and str(move.get("error", "")).startswith("backend error"):
        return True
    return bool(dir_records) and all(r.get("error") for r in dir_records)


def _post_json(url: str, payload: Dict, headers: Dict, timeout: float, max_attempts: Optional[int] = None) -> Dict:
    max_attempts = max_attempts or HTTP_MAX_ATTEMPTS
    data = json.dumps(payload).encode()
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            if exc.code == 500 and "repeat limit" in body.lower():
                raise BackendError(f"HTTP {exc.code}: {body[:500]}", exc.code, body) from None   # reseed, don't wait
            if exc.code in RETRY_STATUS and attempt < max_attempts:
                wait = float(exc.headers.get("retry-after") or delay)
                print(f"    [http {exc.code}] {body.strip()[:300]!r}; retrying in {wait:.0f}s "
                      f"(attempt {attempt}/{max_attempts})")
                time.sleep(min(wait, 120))
                delay = min(delay * 2, 60)
                continue
            raise BackendError(f"HTTP {exc.code}: {body[:500]}", exc.code, body) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < max_attempts:
                print(f"    [network] {exc}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise BackendError(f"network error: {exc}") from None
    raise BackendError("exhausted retries")


class ChatBackend:
    """chat(messages) -> (text, usage dict)"""
    name = "base"

    def chat(self, messages: List[Dict], seed: Optional[int]) -> (str, Dict):
        raise NotImplementedError


class OllamaBackend(ChatBackend):
    name = "ollama"

    def __init__(self, model, base_url, temperature, max_tokens, num_ctx, timeout):
        self.model, self.root = model, base_url.rstrip("/").removesuffix("/v1")
        self.temperature, self.max_tokens, self.num_ctx, self.timeout = temperature, max_tokens, num_ctx, timeout
        self.warned_ctx = False

    def preflight(self):
        try:
            with urllib.request.urlopen(f"{self.root}/api/tags", timeout=10) as resp:
                tags = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as exc:
            sys.exit(f"[preflight] Cannot reach Ollama at {self.root} ({exc}). Start it with: ollama serve")
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            sys.exit(f"[preflight] Model '{self.model}' not pulled. Available: {sorted(names) or '(none)'}\n"
                     f"            Run: ollama pull {self.model}")
        print(f"[preflight] Ollama at {self.root}; model '{self.model}' present; num_ctx={self.num_ctx}.")

    def chat(self, messages, seed):
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.max_tokens}
        if seed is not None:
            opts["seed"] = seed
        out = _post_json(f"{self.root}/api/chat",
                         {"model": self.model, "messages": messages, "stream": False, "options": opts},
                         {}, self.timeout)
        # eval_count counts every generated token, including a thinking model's reasoning
        usage = {"prompt_tokens": out.get("prompt_eval_count"), "completion_tokens": out.get("eval_count"),
                 "reasoning_tokens": None, "total_tokens": None, "done_reason": out.get("done_reason")}
        pt = usage["prompt_tokens"] or 0
        if pt >= 0.95 * self.num_ctx and not self.warned_ctx:
            print(f"  [warn] prompt used {pt}/{self.num_ctx} ctx tokens; it may be truncated. Raise --num-ctx.")
            self.warned_ctx = True
        return (out.get("message") or {}).get("content", "") or "", usage


class OpenAICompatBackend(ChatBackend):
    name = "openai"

    def __init__(self, model, base_url, api_key, temperature, max_tokens, timeout, send_seed=True,
                 reasoning_effort=None):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.tok_key = "max_tokens"
        self.send_temperature, self.send_seed = True, send_seed
        self.reasoning_effort = reasoning_effort          # None = do not send

    def base_payload(self, messages, seed) -> Dict:
        payload = {"model": self.model, "messages": messages, self.tok_key: self.max_tokens}
        if self.send_temperature:
            payload["temperature"] = self.temperature
        if self.send_seed and seed is not None:
            payload["seed"] = seed
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def adapt(self, exc: "BackendError") -> bool:
        """React to a 400 naming a parameter this model rejects. True = retry the request."""
        if exc.status != 400:
            return False
        body = (exc.body or "").lower()
        if "function tools" in body and "reasoning_effort" in body:
            raise BackendError(
                "This model's Chat Completions endpoint rejects function tools unless reasoning_effort is 'none' "
                "(it defaults to a higher effort). Pass --reasoning-effort none (and --director-reasoning-effort "
                "none for Directors), or use --sim-tool-mode text. Server said: " + (exc.body or "")[:300],
                400, exc.body) from None
        if "max_completion_tokens" in body and self.tok_key == "max_tokens":
            self.tok_key = "max_completion_tokens"
        elif "reasoning_effort" in body and self.reasoning_effort is not None:
            print(f"    [param] model rejected reasoning_effort={self.reasoning_effort!r}; dropping it")
            self.reasoning_effort = None
        elif "temperature" in body and self.send_temperature:
            self.send_temperature = False
        elif "seed" in body and self.send_seed:
            self.send_seed = False
        else:
            return False
        print(f"    [param] adapted request: tokens={self.tok_key} temperature={self.send_temperature} "
              f"seed={self.send_seed} reasoning_effort={self.reasoning_effort}")
        return True

    def chat(self, messages, seed):
        for _ in range(5):   # adapt to per-model parameter restrictions, then retry
            try:
                out = _post_json(self.url, self.base_payload(messages, seed), self.headers, self.timeout)
                break
            except BackendError as exc:
                if not self.adapt(exc):
                    raise
        else:
            raise BackendError("could not find accepted request parameters")
        choice = (out.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        u = out.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        return text, {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                      "reasoning_tokens": details.get("reasoning_tokens"),   # OpenAI o-series / gpt-5
                      "total_tokens": u.get("total_tokens"),                 # used to detect hidden thinking
                      "finish_reason": choice.get("finish_reason")}


class AnthropicBackend(ChatBackend):
    name = "anthropic"

    def __init__(self, model, base_url, api_key, temperature, max_tokens, timeout):
        self.model, self.url = model, base_url.rstrip("/") + "/v1/messages"
        self.headers = {"x-api-key": api_key or "", "anthropic-version": "2023-06-01"}
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.send_temperature = True

    def chat(self, messages, seed):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        for _ in range(2):
            payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": msgs}
            if self.send_temperature:
                payload["temperature"] = self.temperature
            try:
                out = _post_json(self.url, payload, self.headers, self.timeout)
                break
            except BackendError as exc:
                if exc.status == 400 and "temperature" in (exc.body or "").lower() and self.send_temperature:
                    self.send_temperature = False
                    print("    [param] dropping temperature for this model")
                    continue
                raise
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        u = out.get("usage") or {}
        inp = sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return text, {"prompt_tokens": inp, "completion_tokens": u.get("output_tokens"),  # includes thinking
                      "reasoning_tokens": None, "total_tokens": None,
                      "finish_reason": out.get("stop_reason")}


# --------------------------------------------------------------------------- #
# LLM builder policy
# --------------------------------------------------------------------------- #
_FORMAT_LINES = {
    "place": ["PLACE:block_code:position:layer:CONFIRM:reason",
              "PLACE:block_code:position:layer:span_to:CONFIRM:reason"],
    "remove": ["REMOVE:position:layer:CONFIRM:reason", "REMOVE:position:layer:span_to:CONFIRM:reason"],
    "done": ["DONE:reason"],
    "clarify": ["CLARIFY:your specific question"],
}


def chat_for_move(backend: ChatBackend, system: str, prompt: str, cfg: argparse.Namespace,
                  allowed: set, seed: Optional[int]) -> Dict:
    """One Builder decision: call the model, extract/parse the move, re-prompt on bad format."""
    system = system + (REASONING_ADDENDUM if cfg.reasoning else SINGLE_LINE_ADDENDUM)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    order = [a for a in ("place", "remove", "done", "clarify") if a in allowed]
    raws, usages = [], []
    move = {"action": "parse_error", "error": "no response"}
    for attempt in range(cfg.max_retries + 1):
        try:
            text, usage = backend.chat(messages, None if seed is None else seed + attempt)
        except BackendError as exc:
            if is_generation_abort(exc) and attempt < cfg.max_retries:
                print(f"    [generation aborted by server: {(exc.body or '')[:120]!r}] retrying with a new seed")
                continue
            raws.append(f"<backend error: {exc}>")
            move = {"action": "parse_error", "error": f"backend error: {exc}"}
            break
        raws.append(text)
        usages.append(usage)
        move = parse_move(extract_move_line(text, last=cfg.reasoning))
        if move["action"] not in allowed and move["action"] != "parse_error":
            move = {"action": "parse_error", "error": f"action {move['action'].upper()} not allowed here"}
        if move["action"] != "parse_error":
            break
        if attempt < cfg.max_retries:
            print(f"    [retry {attempt + 1}] unparseable output, re-prompting for format")
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                    "That was not in the required format. Reply with ONE line only, starting with "
                    + ", ".join(a.upper() + ":" for a in order) + " using exactly:\n"
                    + "\n".join(l for a in order for l in _FORMAT_LINES[a]) + "\nNo other text."},
            ]
    return {"move": move, "raw": raws, "usage": usages, "prompt": prompt, "attempts": len(raws)}


def decide_move(backend: ChatBackend, system: str, prompt: str, cfg: argparse.Namespace, allowed: set,
                seed: Optional[int], sim_ctx: Optional[Dict] = None, mode: str = "single") -> Dict:
    """Builder decision: plain (chat_for_move) or with the simulate_move tool (--sim-tool).
    In tool mode the system message is CRAFT's tool-mode message, as in generate_move_with_tools."""
    if cfg.sim_tool:
        from craft_sim_tool import run_tool_loop
        return run_tool_loop(backend, prompt, cfg, allowed, seed, sim_ctx["board"], sim_ctx["target"],
                             sim_ctx["views"], mode)
    return chat_for_move(backend, system, prompt, cfg, allowed, seed)


class LLMPolicy:
    def __init__(self, backend: ChatBackend, cfg: argparse.Namespace):
        self.backend, self.cfg = backend, cfg
        self.name = f"{backend.name}:{cfg.model}"

    def decide(self, ctx: Dict) -> Dict:
        cfg = self.cfg
        prompt = build_prompt(ctx["views"], ctx["board"], ctx["history"], ctx["turn"], cfg.turns, cfg,
                              ctx.get("shown_candidates"))
        allowed = {"place", "remove"} | ({"done"} if cfg.allow_done else set())
        return decide_move(self.backend, SYSTEM_PROMPT, prompt, cfg, allowed, ctx.get("seed"),
                           {"board": ctx["board"], "target": ctx["target"], "views": ctx["views"]}, "single")


# --------------------------------------------------------------------------- #
# Game loop
# --------------------------------------------------------------------------- #
def _seed_for(structure_id: str, run: int, turn: int, base: int) -> int:
    h = hashlib.sha256(f"{structure_id}|{run}|{turn}|{base}".encode()).hexdigest()
    return int(h[:8], 16)


def _history_line(turn: int, move: Dict, res) -> str:
    if move["action"] == "parse_error":
        return f"turn {turn}: no valid move produced - turn wasted"
    status = "OK" if res and res.ok else f"FAILED ({res.error})" if res else "?"
    return f"turn {turn}: {format_move(move)} -> {status}"


def build_turn_record(t, move, res, cands, shown, board_before, board, metrics, decision, cfg) -> Dict:
    """Per-turn log entry shared by the single-Builder and Director game loops."""
    acting = move["action"] in ("place", "remove")
    rec = {
        "turn": t,
        "action": move["action"],
        "move": move,
        "executed": bool(res and res.ok),
        "error": res.error if res and not res.ok else move.get("error"),
        "error_type": res.error_type if res and not res.ok else None,
        "oracle_available": len(cands),
        "oracle_adherent": any(move_matches(move, c) for c in cands) if cands and acting else None,
        "oracle_shown": shown,
        "shown_adherent": any(move_matches(move, c) for c in shown) if shown and acting else None,
        "taxonomy": classify_turn(move, res, cands),
        "board_before": board_before.stacks,
        "board_after": board.to_json(),
        "metrics": metrics,
    }
    if "raw" in decision:
        rec["raw_model_text"] = decision["raw"]
        rec["llm_attempts"] = decision["attempts"]
        rec["usage"] = decision["usage"]
        if cfg.save_prompts or t == 1:
            rec["prompt"] = decision["prompt"]
        rec["prompt_sha1"] = hashlib.sha1(decision["prompt"].encode()).hexdigest()[:12]
    if "simulations" in decision:
        from craft_sim_tool import sim_turn_fields
        rec.update(sim_turn_fields(decision, move))
    return rec


def play_game(entry: Dict, structure_index: int, policy, cfg: argparse.Namespace, run: int, previousData, checkpoint=None) -> Dict:
    partType = "empty"
    filepath = f'{previousData}/dpip_structure_{structure_index + 1:03d}_{run}.json'
    if not os.path.exists(filepath):
        filepath = f'{previousData}/craft_structure_{structure_index + 1:03d}_{run}.json'
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        partType = data['games'][0]['partialCompletionCategory']

    sid = entry["id"]
    target = Board.from_structure(entry)
    views = target_views(entry, cfg.views_source)
    board = start_board(entry, partType)
    history: List[str] = []
    turns: List[Dict] = []
    done_at = None
    fail_streak = 0
    t0 = time.time()

    for t in range(1, cfg.turns + 1):
        if done_at is not None:
            turns.append({"turn": t, "action": "frozen", "metrics": turns[-1]["metrics"]})
            continue
        cands = enumerate_oracle_moves(board, target)
        shown = sample_oracle_moves(cands, cfg.oracle_n, f"{sid}:{t}") if cfg.oracle_in_prompt else None
        ctx = {"board": board, "target": target, "views": views, "history": history, "turn": t,
               "seed": None if cfg.seed < 0 else _seed_for(sid, run, t, cfg.seed),
               "shown_candidates": shown}
        board_before = board.copy()
        decision = policy.decide(ctx)
        move = decision["move"]

        res = None
        if move["action"] in ("place", "remove"):
            res = board.apply(move)
        elif move["action"] == "done":
            done_at = t

        metrics = compute_metrics(board, target, views)
        rec = build_turn_record(t, move, res, cands, shown, board_before, board, metrics, decision, cfg)
        turns.append(rec)
        if checkpoint:
            checkpoint({"structure_id": sid, "run": run, "partial": True, "policy": policy.name,
                        "config": config_dict(cfg), "turns": turns})
        fail_streak = fail_streak + 1 if turn_backend_failed(move) else 0
        limit = getattr(cfg, "max_backend_fail_turns", 0)
        if limit and fail_streak >= limit:
            raise BackendDown(f"{sid} run{run}: {fail_streak} consecutive turns with backend errors "
                              f"(last: {str(move.get('error'))[:300]})")
        if move["action"] != "done":
            history.append(_history_line(t, move, res))

        m = metrics
        print(f"  [{sid} run{run} t{t:02d}] {format_move(move):<48} "
              f"{'ok ' if rec['executed'] else ('   ' if move['action'] == 'done' else 'ERR')} "
              f"OP={m['progress']:.3f} CP={m['completion']:.3f} views={m['view_match']:.3f}"
              + (f"  ({rec['error']})" if rec["error"] and cfg.verbose else ""))

    final = turns[-1]["metrics"]
    game_usage = token_totals([u for t in turns for u in (t.get("usage") or [])])
    if game_usage["calls"]:
        print(f"  [tokens {sid} run{run}] input={game_usage['input_tokens']:,}  "
              f"output={game_usage['output_total_tokens']:,} "
              f"(reasoning={game_usage['reasoning_tokens']:,})  calls={game_usage['calls']}")
    return {
        "structure_id": sid,
        "complexity": entry.get("complexity"),
        "run": run,
        "policy": policy.name,
        "config": config_dict(cfg),
        "target": target.to_json(),
        "target_views": views,
        "min_moves_true_target": min_moves(target),
        "views_ceiling": views_ceiling(entry, views),
        "done_at": done_at,
        "final_board": board.to_json(),
        "final_metrics": final,
        "turns": turns,
        "elapsed_sec": round(time.time() - t0, 1),
        "token_usage": game_usage,
    }


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def token_totals(usages: List[Dict]) -> Dict:
    """
    Sum per-call usage. output_total_tokens = every generated token (visible text + reasoning).
    Hidden reasoning is taken from reasoning_tokens when the API reports it; otherwise from
    total_tokens - prompt_tokens - completion_tokens (Gemini's OpenAI endpoint reports it that way).
    """
    tot = {"calls": 0, "calls_missing_usage": 0, "input_tokens": 0,
           "output_visible_tokens": 0, "reasoning_tokens": 0, "output_total_tokens": 0}
    for u in usages:
        tot["calls"] += 1
        inp, out = u.get("prompt_tokens"), u.get("completion_tokens")
        if inp is None and out is None:
            tot["calls_missing_usage"] += 1
            continue
        inp, out = inp or 0, out or 0
        total = u.get("total_tokens")
        out_all = max(out, total - inp) if total else out
        reasoning = u.get("reasoning_tokens")
        if reasoning is None:
            reasoning = out_all - out
        tot["input_tokens"] += inp
        tot["output_total_tokens"] += out_all
        tot["reasoning_tokens"] += reasoning
        tot["output_visible_tokens"] += out_all - reasoning
    return tot


def _sim_stats(all_turns: List[Dict]) -> Optional[Dict]:
    turns = [t for t in all_turns if "sim_calls" in t]
    if not turns:
        return None
    sims = [s for t in turns for s in t["simulations"]]
    acting = [t for t in turns if t["action"] in ("place", "remove")]
    return {
        "turns": len(turns),
        "turns_with_simulation": sum(1 for t in turns if t["sim_calls"]),
        "sim_calls_per_turn": _ms([float(t["sim_calls"]) for t in turns]),
        "sim_ok_rate": _ms([float(s["result"]["ok"]) for s in sims]),
        "final_was_simulated_ok": _ms([float(t["final_was_simulated_ok"]) for t in acting
                                       if t.get("final_was_simulated_ok") is not None]),
        "final_failed_in_sim": sum(1 for t in acting if t.get("final_failed_in_sim")),
        "forced_final": dict(Counter(t["sim_forced_final"] for t in turns if t.get("sim_forced_final"))),
        "nudges": sum(t.get("sim_nudges", 0) for t in turns),
        "required_unmet_turns": sum(1 for t in turns if t.get("sim_required_unmet")),
    }


def _director_stats(all_turns: List[Dict]) -> Optional[Dict]:
    recs = [(t, r) for t in all_turns for r in t.get("directors", [])]
    if not recs:
        return None
    return {
        "speakers_per_turn": _ms([float(len(t.get("speakers", []))) for t in all_turns]),
        "messages": len(recs),
        "silent_messages": sum(r["silent"] for _, r in recs),
        "director_errors": sum(bool(r.get("error")) for _, r in recs),
        "tag_fragments_stripped": sum(bool(r.get("tag_fragment_stripped")) for _, r in recs),
        "truncated_messages": sum(bool(r.get("truncated")) for _, r in recs),
        "spoke_by_director": dict(Counter(r["director"] for _, r in recs)),
        "turns_with_no_instruction": sum(1 for t in all_turns
                                         if t.get("directors") is not None and
                                         all(r["silent"] for r in t["directors"])),
        "archetypes": dict(Counter(r["archetype"] for _, r in recs)),
    }


FINAL_KEYS = ["progress", "completion", "iou", "position_accuracy",
              "visible_progress", "visible_completion", "view_match"]


def _ms(vals: List[float]) -> Dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sem": None}
    sem = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4), "sem": None if sem is None else round(sem, 4)}


def summarize(games: List[Dict]) -> Dict:
    all_turns = [t for g in games for t in g["turns"] if t["action"] != "frozen"]
    by_cplx = defaultdict(list)
    for g in games:
        by_cplx[g.get("complexity") or "?"].append(g)
    adh = [t["oracle_adherent"] for t in all_turns if t.get("oracle_adherent") is not None]
    n_turns = games[0]["config"]["turns"] if games else 0
    return {
        "n_games": len(games),
        "policy": games[0]["policy"] if games else None,
        "final": {k: _ms([g["final_metrics"][k] for g in games]) for k in FINAL_KEYS},
        "views_exact_rate": _ms([float(g["final_metrics"]["views_exact"]) for g in games]),
        "views_ceiling": {k: _ms([g["views_ceiling"][k] for g in games]) for k in FINAL_KEYS},
        "final_by_complexity": {c: {k: _ms([g["final_metrics"][k] for g in gs]) for k in ("progress", "view_match")}
                                for c, gs in sorted(by_cplx.items())},
        "turnwise_mean": {k: [round(statistics.fmean(g["turns"][i]["metrics"][k] for g in games), 4)
                              for i in range(n_turns)]
                          for k in ("progress", "completion", "view_match")} if games else {},
        "action_counts": dict(Counter(t["action"] for t in all_turns)),
        "executed_rate": _ms([float(t["executed"]) for t in all_turns if t["action"] in ("place", "remove")]),
        "error_types": dict(Counter(t["error_type"] for t in all_turns if t.get("error_type"))),
        "taxonomy": dict(Counter(t["taxonomy"] for t in all_turns)),
        "oracle_adherence": _ms([float(a) for a in adh]),
        "done_turns": [g["done_at"] for g in games],
        "tokens": token_totals([u for t in all_turns for u in (t.get("usage") or [])]),
        "tokens_directors": token_totals([u for t in all_turns for u in t.get("director_usage", [])]),
        "director_stats": _director_stats(all_turns),
        "sim_stats": _sim_stats(all_turns),
    }


def print_summary(s: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"  FULL-INFORMATION BUILDER  |  {s['policy']}  |  games={s['n_games']}")
    print("=" * 72)
    print(f"  {'metric (turn 20)':<24}{'mean':>9}{'sem':>9}   {'views-only ceiling':>18}")
    for k in FINAL_KEYS:
        a, c = s["final"][k], s["views_ceiling"][k]
        fmt = lambda b: ("n/a" if b["mean"] is None else f"{b['mean']:.3f}")
        sem = "n/a" if a["sem"] is None else f"{a['sem']:.3f}"
        print(f"  {k:<24}{fmt(a):>9}{sem:>9}   {fmt(c):>18}")
    ve = s["views_exact_rate"]["mean"]
    print(f"  {'all 3 views exact':<24}{(0 if ve is None else ve):>9.3f}")
    print("  " + "-" * 68)
    print(f"  actions: {s['action_counts']}")
    print(f"  executed rate (place/remove): {s['executed_rate']['mean']}")
    print(f"  engine errors: {s['error_types']}")
    print(f"  oracle adherence (vs TRUE target): {s['oracle_adherence']['mean']}")
    print(f"  taxonomy: {s['taxonomy']}")
    ss = s.get("sim_stats")
    if ss:
        print(f"  simulate_move: {ss['turns_with_simulation']}/{ss['turns']} turns used it, "
              f"calls/turn={ss['sim_calls_per_turn']['mean']}, sim ok rate={ss['sim_ok_rate']['mean']}, "
              f"final move = an ok simulation: {ss['final_was_simulated_ok']['mean']}, "
              f"submitted a move that failed in sim: {ss['final_failed_in_sim']}, forced finals={ss['forced_final']}, "
              f"nudges={ss['nudges']}, turns where --sim-require was still unmet={ss['required_unmet_turns']}")
    ds = s.get("director_stats")
    if ds:
        print(f"  directors: {ds['messages']} messages ({ds['silent_messages']} silent), "
              f"speakers/turn={ds['speakers_per_turn']['mean']}, by director={ds['spoke_by_director']}, "
              f"malformed-tag messages cleaned={ds['tag_fragments_stripped']}, "
              f"truncated at max tokens (dropped)={ds['truncated_messages']}")
    tkd = s.get("tokens_directors") or {"calls": 0}
    if tkd["calls"]:
        print(f"  TOKENS (directors)  input={tkd['input_tokens']:,}  output={tkd['output_total_tokens']:,} "
              f"(visible={tkd['output_visible_tokens']:,}, reasoning={tkd['reasoning_tokens']:,})  "
              f"calls={tkd['calls']}")
    tk = s["tokens"]
    if tk["calls"]:
        print(f"  TOKENS (builder)  input={tk['input_tokens']:,}  output={tk['output_total_tokens']:,} "
              f"(visible={tk['output_visible_tokens']:,}, reasoning={tk['reasoning_tokens']:,})  "
              f"calls={tk['calls']}" + (f"  [usage missing on {tk['calls_missing_usage']} calls]"
                                        if tk["calls_missing_usage"] else ""))
    print("=" * 72)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
CONFIG_KEYS = ["backend", "model", "turns", "temperature", "max_tokens", "seed", "reasoning", "view_format",
               "show_current_views", "history_window", "oracle_in_prompt", "oracle_n", "views_source",
               "allow_done", "max_retries", "num_ctx",
               "directors", "director_views", "director_backend", "director_model", "director_temperature",
               "director_max_tokens", "sim_tool", "max_simulations", "sim_score", "sim_require", "sim_tool_mode",
               "reasoning_effort", "director_reasoning_effort"]


def config_dict(cfg) -> Dict:
    return {k: getattr(cfg, k) for k in CONFIG_KEYS}


def make_backend(kind, model, base_url, api_key_env, temperature, max_tokens, cfg,
                 reasoning_effort=None) -> ChatBackend:
    if reasoning_effort is not None and kind not in ("openai", "gemini"):
        print(f"[warn] reasoning effort is only sent for the openai/gemini backends; ignored for '{kind}'")
    if not model:
        sys.exit(f"a model name is required for backend '{kind}'")
    if kind == "ollama":
        be = OllamaBackend(model, base_url or "http://localhost:11434", temperature, max_tokens,
                           cfg.num_ctx, cfg.timeout)
        if not cfg.skip_preflight:
            be.preflight()
        return be
    if kind in ("openai", "gemini"):
        default = {"openai": "https://api.openai.com/v1",
                   "gemini": "https://generativelanguage.googleapis.com/v1beta/openai"}[kind]
        env = api_key_env or {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[kind]
        key = os.environ.get(env)
        if not key and not base_url:
            sys.exit(f"Set {env} (or pass an api-key-env option)")
        be = OpenAICompatBackend(model, base_url or default, key, temperature, max_tokens, cfg.timeout,
                                 send_seed=kind == "openai", reasoning_effort=reasoning_effort)
        be.name = kind
        return be
    if kind == "anthropic":
        key = (os.environ.get(api_key_env) if api_key_env else None) \
            or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        if not key and not base_url:
            sys.exit("Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY)")
        return AnthropicBackend(model, base_url or "https://api.anthropic.com", key, temperature, max_tokens,
                                cfg.timeout)
    sys.exit(f"unknown backend {kind}")


def make_builder_backend(cfg) -> ChatBackend:
    return make_backend(cfg.backend, cfg.model, cfg.base_url, cfg.api_key_env, cfg.temperature,
                        cfg.max_tokens, cfg, cfg.reasoning_effort)


def make_director_backend(cfg) -> ChatBackend:
    return make_backend(cfg.director_backend, cfg.director_model, cfg.director_base_url,
                        cfg.director_api_key_env, cfg.director_temperature, cfg.director_max_tokens, cfg,
                        cfg.director_reasoning_effort)


def make_policy(cfg):
    if cfg.backend == "scripted-oracle":
        return OraclePolicy()
    if cfg.backend == "scripted-views":
        return ViewsPolicy()
    return LLMPolicy(make_builder_backend(cfg), cfg)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CRAFT with no information separation: one Builder, all three views.")
    ap.add_argument("--structures", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "openai", "gemini", "anthropic", "scripted-oracle", "scripted-views"])
    ap.add_argument("--model", default=None, help="e.g. qwen2.5:7b-instruct, gpt-4o-mini, claude-...")
    ap.add_argument("--base-url", default=None, help="override endpoint (e.g. a vLLM server with --backend openai)")
    ap.add_argument("--api-key-env", default=None, help="name of env var holding the API key")
    ap.add_argument("--structure-ids", default=None, help="comma-separated ids, e.g. structure_001,structure_004")
    ap.add_argument("--limit", type=int, default=None, help="first N structures only")
    ap.add_argument("--runs", type=int, default=1, help="independent runs per structure (paper: 3)")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.1, help="CRAFT builder default 0.1")
    ap.add_argument("--reasoning-effort", default=None, choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    help="Builder reasoning effort for openai/gemini backends (not sent if omitted). GPT-5.4-and-later "
                         "models reject function tools on Chat Completions unless this is 'none' (use "
                         "--sim-tool-mode text to combine tools with reasoning). Hidden reasoning "
                         "tokens count against --max-tokens.")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 400, or 3000 with --reasoning")
    ap.add_argument("--seed", type=int, default=0, help="base sampling seed; -1 disables")
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window per request")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=2, help="format-correction retries per turn (free)")
    ap.add_argument("--reasoning", action="store_true", help="allow free reasoning; move on the last line")
    ap.add_argument("--view-format", choices=["json", "annotated"], default="json",
                    help="json = the Directors' view format; annotated = same info with coordinates")
    ap.add_argument("--show-current-views", action="store_true",
                    help="also show the current board projected onto each wall (ablation)")
    ap.add_argument("--history-window", type=int, default=-1, help="-1 all past turns, 0 none, N last N")
    assist = ap.add_mutually_exclusive_group()   # Builder assistance: oracle OR simulation tool OR neither
    assist.add_argument("--oracle-in-prompt", action="store_true",
                        help="show up to --oracle-n verified moves (NOTE: leaks true-target interior info)")
    assist.add_argument("--sim-tool", action="store_true",
                        help="give the Builder CRAFT's simulate_move tool instead of oracle candidates")
    ap.add_argument("--max-simulations", type=int, default=3, help="simulate_move calls per turn (CRAFT: 3)")
    ap.add_argument("--sim-require", type=int, default=0, metavar="N",
                    help="require at least N simulate_move calls before the Builder's final answer (default 0 = "
                         "optional, as in CRAFT). Forces tool_choice where the API supports it, else re-prompts "
                         "up to 2 times per turn. Small local models usually need 1.")
    ap.add_argument("--sim-tool-mode", choices=["native", "text"], default="native",
                    help="native = the backend's tool-calling API (CRAFT). text = the tool is described in the "
                         "prompt and <tool_call> text is parsed by the harness; use it when a local model returns "
                         "empty replies or ignores native tools")
    ap.add_argument("--sim-score", choices=["target", "views"], default="target",
                    help="target = progress vs the true structure (CRAFT; leaks target info); "
                         "views = progress = view consistency")
    ap.add_argument("--oracle-n", type=int, default=5)
    ap.add_argument("--views-source", choices=["recompute", "dataset"], default="recompute",
                    help="dataset views store size=1 everywhere; recompute follows paper App. B.4")
    ap.add_argument("--no-done", dest="allow_done", action="store_false", help="disable the DONE action")
    ap.add_argument("--save-prompts", action="store_true", help="store every prompt (default: turn 1 only)")
    # ---- Director mode ------------------------------------------------------
    dg = ap.add_argument_group("director mode (CRAFT game with Directors)")
    dg.add_argument("--directors", action="store_true",
                    help="play CRAFT with 3 Directors instructing the Builder (off: single full-info Builder)")
    dg.add_argument("--director-views", choices=["all", "own"], default="all",
                    help="all = every Director sees all 3 views (no separation); own = standard CRAFT control")
    dg.add_argument("--director-backend", default=None,
                    choices=["ollama", "openai", "gemini", "anthropic"], help="default: same as --backend")
    dg.add_argument("--director-model", default=None, help="default: same as --model")
    dg.add_argument("--director-base-url", default=None, help="default: same as --base-url if same backend")
    dg.add_argument("--director-api-key-env", default=None)
    dg.add_argument("--director-temperature", type=float, default=0.7, help="CRAFT Director default 0.7")
    dg.add_argument("--director-reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    help="default: same as --reasoning-effort when Directors use the same backend as the Builder")
    dg.add_argument("--director-max-tokens", type=int, default=512,
                    help="paper: 512 open-weight, 2000 GPT, 3000 Claude/Gemini")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--http-retries", type=int, default=6, metavar="N",
                    help="attempts per model request on 5xx/429/network errors (backoff 2,4,8,16,32 s); "
                         "1 = fail fast")
    ap.add_argument("--max-backend-fail-turns", type=int, default=3, metavar="N",
                    help="stop the run after N consecutive turns whose model calls all failed (0 = never stop). "
                         "A partial game log is kept.")
    ap.add_argument("--previousDataSettings", type=str, default="/home/hannah/CRAFT/CRAFT/previousRunData",
                    help="Path to previous data folder if replicating settings")
    cfg = ap.parse_args(argv)
    global HTTP_MAX_ATTEMPTS
    PREVIOUS_DATA_PATH = cfg.previousDataSettings
    HTTP_MAX_ATTEMPTS = max(1, cfg.http_retries)
    if cfg.max_tokens is None:
        cfg.max_tokens = 3000 if cfg.reasoning else 400
    if cfg.sim_tool and cfg.max_simulations < 1:
        sys.exit("--max-simulations must be >= 1")
    if cfg.sim_tool and not 0 <= cfg.sim_require <= cfg.max_simulations:
        sys.exit("--sim-require must be between 0 and --max-simulations")
    if not cfg.sim_tool:
        if cfg.sim_require:
            sys.exit("--sim-require needs --sim-tool")
        cfg.max_simulations = cfg.sim_score = cfg.sim_require = cfg.sim_tool_mode = None
    if cfg.backend.startswith("scripted"):
        if cfg.sim_tool:
            sys.exit("--sim-tool needs an LLM Builder backend (not scripted-*)")
        if cfg.directors:
            sys.exit("--directors needs an LLM Builder backend (not scripted-*)")
        cfg.model = cfg.backend
    if cfg.directors:
        cfg.allow_done = False            # CRAFT has no DONE; the Builder may CLARIFY instead
        cfg.director_backend = cfg.director_backend or cfg.backend
        cfg.director_model = cfg.director_model or cfg.model
        if cfg.director_base_url is None and cfg.director_backend == cfg.backend:
            cfg.director_base_url = cfg.base_url
        if cfg.director_api_key_env is None and cfg.director_backend == cfg.backend:
            cfg.director_api_key_env = cfg.api_key_env
        if cfg.director_reasoning_effort is None and cfg.director_backend == cfg.backend:
            cfg.director_reasoning_effort = cfg.reasoning_effort
    else:
        cfg.director_views = cfg.director_backend = cfg.director_model = None
        cfg.director_temperature = cfg.director_max_tokens = cfg.director_reasoning_effort = None

    structures = load_structures(cfg.structures)
    for i, s_ in enumerate(structures):          # index in the full dataset: seeds Director archetypes
        s_["_index"] = i
    if cfg.structure_ids:
        wanted = [s.strip() for s in cfg.structure_ids.split(",")]
        structures = [s for s in structures if s["id"] in wanted]
    if cfg.limit:
        structures = structures[: cfg.limit]
    if not structures:
        sys.exit("no structures selected")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg.out_dir / "_config.json"
    if cfg_path.exists() and not cfg.overwrite:
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), v) for k, v in config_dict(cfg).items()
                if (k in old or v not in (None, False)) and old.get(k) != v}
        if diff:
            sys.exit(f"[config] {cfg.out_dir} holds a run with a different config: {diff}\n"
                     f"         Use a new --out-dir, or --overwrite.")
    cfg_path.write_text(json.dumps(config_dict(cfg), indent=2))

    if cfg.directors:
        from craft_directors import play_game_directors
        builder_be = make_builder_backend(cfg)
        director_be = builder_be if (cfg.director_backend, cfg.director_model, cfg.director_base_url,
                                     cfg.director_temperature, cfg.director_max_tokens,
                                     cfg.director_reasoning_effort) == \
            (cfg.backend, cfg.model, cfg.base_url, cfg.temperature, cfg.max_tokens,
             cfg.reasoning_effort) else make_director_backend(cfg)
        label = (f"directors[{cfg.director_views}] {cfg.director_backend}:{cfg.director_model} "
                 f"-> builder {cfg.backend}:{cfg.model}")
        tag = f"dirs-{cfg.director_views}_{cfg.director_model}+{cfg.model}"
    else:
        policy = make_policy(cfg)
        label, tag = policy.name, cfg.model
    # One cheap call before any game: native tool calling can be refused for a given model + reasoning effort
    # (e.g. GPT-5.4+ on Chat Completions with effort != none). Failing here costs a few hundred tokens instead of a
    # wasted game's Director calls.
    if cfg.sim_tool and cfg.sim_tool_mode == "native" and cfg.backend in ("openai", "gemini") \
            and not cfg.skip_preflight:
        from craft_sim_tool import probe_tool_calling
        builder_obj = builder_be if cfg.directors else policy.backend
        builder_obj.tool_mode = "native"
        try:
            res = probe_tool_calling(builder_obj, trials=1)
        except BackendError as exc:
            sys.exit(f"[preflight] The Builder cannot use native tool calling with these settings:\n  {exc}")
        print("[preflight] native tool calling accepted by the Builder model"
              + ("" if res["structured"] else " (but the test prompt got no tool call back; check `[SIM summary]`)"))
    safe_model = re.sub(r"[^A-Za-z0-9.+_-]+", "_", tag)
    games = []
    for run in range(cfg.runs):
        for entry in structures:
            path = cfg.out_dir / f"{safe_model}_{entry['id']}_run{run}.json"
            if path.exists() and not cfg.overwrite:
                print(f"[skip] {path.name} exists")
                games.append(json.loads(path.read_text()))
                continue
            print(f"\n=== {entry['id']} ({entry.get('complexity')}) run {run} | {label} ===")
            partial = path.with_name(path.stem + ".partial.json")     # rewritten after every turn
            ckpt = lambda g_, p_=partial: p_.write_text(json.dumps(g_, indent=1))     # noqa: E731
            try:
                if cfg.directors:
                    g = play_game_directors(entry, entry["_index"], builder_be, director_be, cfg, run, PREVIOUS_DATA_PATH,
                                            checkpoint=ckpt)
                else:
                    g = play_game(entry, entry["_index"], policy, cfg, run, PREVIOUS_DATA_PATH, checkpoint=ckpt)
            except BackendDown as exc:
                print(f"\n[ABORT] {exc}")
                print(f"  Partial game log: {partial}")
                print("  The model server is failing. Check its log (Ollama on macOS: ~/.ollama/logs/server.log) and "
                      "`ollama ps`. Re-run the same command to resume: finished games are skipped.")
                sys.exit(1)
            path.write_text(json.dumps(g, indent=1))
            if partial.exists():
                partial.unlink()
            games.append(g)

    summary = summarize(games)
    (cfg.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"  Wrote {len(games)} game files + _summary.json to {cfg.out_dir}")


if __name__ == "__main__":
    main()
