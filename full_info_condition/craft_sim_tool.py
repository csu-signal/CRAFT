#!/usr/bin/env python3
"""
craft_sim_tool.py
=================

Move-simulation tool for the Builder (--sim-tool), an alternative to oracle
candidates (--oracle-in-prompt). The two are mutually exclusive; neither is the
default.

Ported from the CRAFT repo:
  * simulate_move           <- agents/builder_tools.simulate_move
                               (re-targeted from EnhancedGameState.execute_move to this
                               harness's Board; same return keys and hint wording)
  * tool schema + workflow  <- builder_agent.generate_move_with_tools /
                               create_builder_prompt_with_tools
  * budget handling         <- generate_move_with_tools: up to N simulate_move calls per
                               turn; if a round would exceed the budget it is discarded
                               and a final answer is forced; when the budget is used up a
                               final answer is forced with tools disabled.
Simulations are free: they do not use game turns.

Return fields of simulate_move (keys as in CRAFT):
  ok, error, hint                         as CRAFT
  blocks_correct / blocks_total           --sim-score target: layer-exact correct blocks / target blocks
                                          --sim-score views:  matching view cells / 81
  overall_progress                        target: CRAFT overall progress (IoU+CP+PA)/3; views: view_match
  correctness.structurePlacement          target: move is a verified forward-progress move toward the
                                          true target (same test as the oracle); views: = sidePlacement
  correctness.sidePlacement               the move makes the affected wall cells match the target views
                                          (place) or removes a block that mismatches them (remove);
                                          False for moves touching only (1,1)/(2,1)
CRAFT's EnhancedGameState is not available here, so these two flags follow the
definitions above rather than CRAFT's internal code.

Tool calling is native for every backend (OpenAI-compatible incl. Gemini, Ollama
/api/chat, Anthropic). For local models that print the call as text instead of
emitting a structured call (e.g. Qwen's <tool_call>{...}</tool_call>), the text is
parsed as a call.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from craft_full_info_env import (
    WALLS, Board, compute_metrics, enumerate_oracle_moves, move_matches, norm_pos, project_view,
)

TOOL_NAME = "simulate_move"
TOOL_ROUND_ADDENDUM = (" Use the simulate_move tool through the tool-calling interface, not by writing the call "
                       "as text. Only your final answer is plain text: one PLACE/REMOVE/CLARIFY line.")
MAX_EMPTY_RETRIES = 2       # re-sends of a request whose reply came back completely empty (new seed each time)
MAX_SIM_NUDGES = 2          # consecutive re-prompts without a tool call under --sim-require before giving up
TOOL_DESCRIPTION = ("Dry-run a proposed move against the environment. "
                    "Returns {ok, error, metrics, correctness, resulting_structure}. "
                    "Does NOT mutate game state.")
# CRAFT used "span_to": {"type": ["string", "null"]}; a plain string (optional) is used here
# because some providers' OpenAI-compatible endpoints reject union types.
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "move": {
            "type": "object",
            "description": ("Proposed move with keys: action (place|remove|clarify), "
                            "block (e.g. 'gs'), position (e.g. '(0,0)'), "
                            "layer (int), span_to (e.g. '(1,0)'; omit for small blocks)."),
            "properties": {
                "action": {"type": "string", "enum": ["place", "remove", "clarify"]},
                "block": {"type": "string"},
                "position": {"type": "string"},
                "layer": {"type": "integer"},
                "span_to": {"type": "string"},
            },
            "required": ["action", "position", "layer"],
        }
    },
    "required": ["move"],
}
OPENAI_TOOLS = [{"type": "function",
                 "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS}}]
ANTHROPIC_TOOLS = [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_PARAMETERS}]


def tool_system_message(max_simulations: int) -> str:
    """CRAFT generate_move_with_tools system message (verbatim)."""
    return (f"You are a Builder agent. You may call simulate_move up to {max_simulations} times to dry-run moves. "
            "After simulations, output ONE final move in the PLACE/REMOVE/CLARIFY text format. "
            "No JSON, no extra commentary.")


def tool_addendum(max_simulations: int, mode: str) -> str:
    """mode 'directors': CRAFT's thin_addendum verbatim. mode 'single': same rules, no Directors."""
    if mode == "directors":
        return f"""
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
    return f"""
---

TOOL MODE — simulate_move available ({max_simulations} calls max this turn; simulations do not use up turns):

WORKFLOW:
1. Simulate the move(s) you are considering before committing to one.
2. Pick the result with greatest value for "progress".
3. Submit that exact move as your FINAL answer. DO NOT INVENT NEW MOVE AFTER SIMULATING.
4. If a sim fails (ok=False) → fix ONLY the field the hint specifies, retry once.
5. NEVER submit a move that returned ok=False.
6. NEVER submit a remove move where simulate shows structurePlacement=False.
"""


# --------------------------------------------------------------------------- #
# simulate_move
# --------------------------------------------------------------------------- #
def _coerce_move(raw) -> Dict:
    move = dict(raw.get("move", raw)) if isinstance(raw, dict) else {}
    move.setdefault("span_to", None)
    move.setdefault("confirmation", "simulation")
    move["action"] = str(move.get("action", "")).lower().strip()
    if move.get("block") is not None:
        move["block"] = str(move["block"]).lower().strip()
    move["position"] = norm_pos(move.get("position")) or move.get("position")
    move["span_to"] = norm_pos(move.get("span_to")) if move.get("span_to") not in (None, "", "null", "None") else None
    try:
        move["layer"] = int(move.get("layer"))
    except (TypeError, ValueError):
        pass
    return move


def _hint(error: str, error_type: Optional[str]) -> str:
    hint = f"FAILED: {error}."
    m = (re.search(r"spans to (\(\d,\d\))", error) or re.search(r"other half is at (\(\d,\d\))", error))
    if m:
        hint += f" → Use span_to={m.group(1)} instead."
    elif error_type == "layer":
        hint += " → Count stack height from board state for correct layer."
    elif error_type == "empty":
        hint += " → Cell is empty, cannot remove."
    elif error_type == "span" and "neighbour" in error:
        hint += " → span_to must be directly adjacent (not diagonal)."
    hint += " Fix ONLY the field causing this error, keep everything else."
    return hint


def _wall_slots(cells: List[str]) -> List[Tuple[str, int]]:
    return [(d, WALLS[d].index(c)) for c in cells for d in WALLS if c in WALLS[d]]


def _side_placement(before: Board, after: Board, move: Dict, target_views: Dict) -> bool:
    cells = [move["position"]] + ([move["span_to"]] if move.get("span_to") else [])
    slots = _wall_slots(cells)
    if not slots:
        return False
    k = move["layer"]
    board = after if move["action"] == "place" else before
    for d, i in slots:
        cur = project_view(board, d)[f"row_{k}"][i]
        tgt = target_views[d][f"row_{k}"][i]
        same = cur["color"] == tgt["color"] and int(cur["size"]) == int(tgt["size"])
        if move["action"] == "place" and not same:
            return False
        if move["action"] == "remove" and same:
            return False     # removing a block that already matches the view is not a correction
    return True


def simulate_move(board: Board, target: Board, target_views: Dict, raw_move, score: str = "target") -> Dict:
    move = _coerce_move(raw_move)
    if move["action"] not in ("place", "remove"):
        return {"ok": False, "error": f"cannot simulate action '{move['action']}'",
                "hint": "Only place and remove moves can be simulated."}
    sim = board.copy()
    try:
        res = sim.apply(move)
    except Exception as exc:                                          # noqa: BLE001
        return {"ok": False, "error": str(exc), "hint": f"Exception: {exc}. Fix move format."}
    if not res.ok:
        return {"ok": False, "error": res.error, "hint": _hint(res.error, res.error_type)}

    m = compute_metrics(sim, target, target_views)
    side = _side_placement(board, sim, move, target_views)
    if score == "views":
        matched = round(m["view_match"] * 81)
        return {"ok": True, "blocks_correct": matched, "blocks_total": 81, "overall_progress": m["view_match"],
                "correctness": {"structurePlacement": side, "sidePlacement": side}}
    total = sum(len(s) for s in target.stacks.values())
    structure = any(move_matches(move, c) for c in enumerate_oracle_moves(board, target))
    return {"ok": True, "blocks_correct": round(m["completion"] * total), "blocks_total": total,
            "overall_progress": m["progress"],
            "correctness": {"structurePlacement": structure, "sidePlacement": side}}


# --------------------------------------------------------------------------- #
# Tool-capable chat per backend.  Conversation is kept in OpenAI format:
#   {"role": "assistant", "content": str|None, "tool_calls": [{"id", "type", "function": {"name", "arguments"}}]}
#   {"role": "tool", "tool_call_id": id, "name": name, "content": str}
# --------------------------------------------------------------------------- #
_MOVE_LINE_RE = re.compile(r"^\s*[-*>\[`]*\s*(PLACE|REMOVE|CLARIFY)\s*:", re.IGNORECASE | re.MULTILINE)
_SIM_LINE_RE = re.compile(r"^\s*[-*>\[`]*\s*SIMULATE(?:_MOVE)?\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SIM_CALL_RE = re.compile(r"simulate_move\s*\(", re.IGNORECASE)


def _json_objects(text: str) -> List[Dict]:
    """All top-level {...} objects in text that parse as JSON (string-aware brace matching)."""
    objs, i = [], 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < len(text):
            c = text[j]
            if in_str:
                esc = (c == "\\" and not esc)
                if c == '"' and not esc:
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0 and j < len(text):
            try:
                obj = json.loads(text[i:j + 1])
                if isinstance(obj, dict):
                    objs.append(obj)
                i = j + 1
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return objs


def _as_args(obj) -> Optional[Dict]:
    """Normalize a decoded call payload to {'move': {...}}."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("move"), dict):
        return {"move": obj["move"]}
    if "action" in obj:
        return {"move": obj}
    return None


def _python_style_calls(text: str) -> List[Dict]:
    """simulate_move(action="place", block="ys", position="(0,0)", layer=0) / simulate_move({...})."""
    import ast
    calls = []
    for m in _SIM_CALL_RE.finditer(text):
        depth, j, in_str, quote = 1, m.end(), False, ""
        while j < len(text) and depth:
            c = text[j]
            if in_str:
                if c == quote and text[j - 1] != "\\":
                    in_str = False
            elif c in "\"'":
                in_str, quote = True, c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth:
            continue
        inner = text[m.end():j - 1]
        try:
            node = ast.parse(f"f({inner})", mode="eval").body
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
            args = _as_args(node.args and ast.literal_eval(node.args[0]) or kw)
        except (SyntaxError, ValueError):
            args = _as_args((_json_objects(inner) or [None])[0])
        if args:
            calls.append(args)
    return calls


def _text_tool_calls(text: str) -> List[Dict]:
    """
    Recover calls a local model wrote as text instead of emitting a structured call. Accepted forms:
      <tool_call>{"name": "simulate_move", "arguments": {...}}</tool_call>   (Qwen native text form)
      {"name": "simulate_move", "arguments"|"parameters"|"args": {...}}      (also inside ```json fences)
      {"move": {...}}  /  {"action": "place", ...}  (bare payload; only if the text has no PLACE:/REMOVE: line)
      simulate_move(action="place", block="ys", position="(0,0)", layer=0)  /  simulate_move({...})
      SIMULATE:PLACE:ys:(0,0):0:CONFIRM:x                                    (move-line format)
    """
    text = text or ""
    found: List[Dict] = []
    for obj in _json_objects(text):
        if obj.get("name") == TOOL_NAME or obj.get("function") == TOOL_NAME:
            payload = obj.get("arguments", obj.get("parameters", obj.get("args", obj.get("input"))))
            args = _as_args(payload)
            if args:
                found.append(args)
        elif isinstance(obj.get("function"), dict) and obj["function"].get("name") == TOOL_NAME:
            args = _as_args(obj["function"].get("arguments"))
            if args:
                found.append(args)
    if not found:
        found = _python_style_calls(text)
    if not found:
        from run_full_info_builder import parse_move
        for m in _SIM_LINE_RE.finditer(text):
            mv = parse_move(m.group(1).strip().strip("`"))
            if mv["action"] in ("place", "remove"):
                found.append({"move": {k: v for k, v in mv.items() if k != "confirmation" and v is not None}})
    if not found and not _MOVE_LINE_RE.search(text):          # bare payload: only when no final-move line exists
        for obj in _json_objects(text):
            args = _as_args(obj)
            if args:
                found.append(args)
    return [{"id": f"call_txt_{i}", "name": TOOL_NAME, "arguments": a} for i, a in enumerate(found)]


def _openai_style_assistant(text: str, calls: List[Dict]) -> Dict:
    msg = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                             for c in calls]
    return msg


def _tool_chat_native(backend, messages: List[Dict], use_tools: bool, seed: Optional[int], force: bool = False):
    """-> (text, calls [{id, name, arguments: dict}], assistant_message, usage)"""
    from run_full_info_builder import (AnthropicBackend, BackendError, OllamaBackend, OpenAICompatBackend,
                                       _post_json)
    if isinstance(backend, OllamaBackend):
        conv = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                conv.append({"role": "assistant", "content": m.get("content") or "",
                             "tool_calls": [{"function": {"name": tc["function"]["name"],
                                                          "arguments": json.loads(tc["function"]["arguments"])}}
                                            for tc in m["tool_calls"]]})
            elif m["role"] == "tool":
                conv.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", TOOL_NAME)})
            else:
                conv.append({"role": m["role"], "content": m.get("content") or ""})
        opts = {"temperature": backend.temperature, "num_ctx": backend.num_ctx, "num_predict": backend.max_tokens}
        if seed is not None:
            opts["seed"] = seed
        payload = {"model": backend.model, "messages": conv, "stream": False, "options": opts}
        if use_tools:
            payload["tools"] = OPENAI_TOOLS
        out = _post_json(f"{backend.root}/api/chat", payload, {}, backend.timeout)
        msg = out.get("message") or {}
        text = msg.get("content") or ""
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            calls.append({"id": f"call_{i}", "name": fn.get("name"),
                          "arguments": args if isinstance(args, dict) else json.loads(args)})
        if use_tools and not calls:
            calls = _text_tool_calls(text)
        usage = {"prompt_tokens": out.get("prompt_eval_count"), "completion_tokens": out.get("eval_count"),
                 "reasoning_tokens": None, "total_tokens": None, "done_reason": out.get("done_reason")}
        return text, calls, _openai_style_assistant(text, calls), usage

    if isinstance(backend, OpenAICompatBackend):
        # tool results carry a "name" key (used by Ollama/Anthropic conversions); OpenAI's schema has none
        msgs = [m if m.get("role") != "tool" or backend.name == "gemini"
                else {k: v for k, v in m.items() if k != "name"} for m in messages]
        for _ in range(6):
            payload = backend.base_payload(msgs, seed)
            if use_tools:
                payload["tools"], payload["tool_choice"] = OPENAI_TOOLS, ("required" if force else "auto")
            try:
                out = _post_json(backend.url, payload, backend.headers, backend.timeout)
                break
            except BackendError as exc:
                if exc.status == 400 and force and "tool_choice" in (exc.body or "").lower():
                    force = False          # endpoint rejects tool_choice=required; the loop re-prompts instead
                    continue
                if not backend.adapt(exc):
                    raise
        else:
            raise BackendError("could not find accepted request parameters")
        choice = (out.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": fn.get("arguments")}
            calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
        if use_tools and not calls:
            calls = _text_tool_calls(text)
        u = out.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                 "reasoning_tokens": details.get("reasoning_tokens"), "total_tokens": u.get("total_tokens"),
                 "done_reason": choice.get("finish_reason")}
        return text, calls, _openai_style_assistant(text, calls), usage

    if isinstance(backend, AnthropicBackend):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        conv: List[Dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": json.loads(tc["function"]["arguments"])})
                conv.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no text)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list) \
                        and conv[-1]["content"] and conv[-1]["content"][0].get("type") == "tool_result":
                    conv[-1]["content"].append(block)
                else:
                    conv.append({"role": "user", "content": [block]})
            else:
                conv.append({"role": "user", "content": m["content"]})
        # merge consecutive user turns (tool results followed by an instruction)
        merged: List[Dict] = []
        for m in conv:
            if merged and merged[-1]["role"] == m["role"] == "user":
                prev = merged[-1]["content"]
                prev = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
                cur = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
                merged[-1]["content"] = prev + cur
            else:
                merged.append(m)
        payload = {"model": backend.model, "max_tokens": backend.max_tokens, "system": system, "messages": merged,
                   "tools": ANTHROPIC_TOOLS,
                   # history may contain tool_use blocks, so tools stay declared; final answers disable them
                   "tool_choice": ({"type": "any"} if force else {"type": "auto"}) if use_tools else {"type": "none"}}
        if backend.send_temperature:
            payload["temperature"] = backend.temperature
        out = _post_json(backend.url, payload, backend.headers, backend.timeout)
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        calls = [{"id": b["id"], "name": b["name"], "arguments": b.get("input") or {}}
                 for b in out.get("content", []) if b.get("type") == "tool_use"] if use_tools else []
        u = out.get("usage") or {}
        inp = sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        usage = {"prompt_tokens": inp, "completion_tokens": u.get("output_tokens"),
                 "reasoning_tokens": None, "total_tokens": None}
        return text, calls, _openai_style_assistant(text, calls), usage

    raise TypeError(f"backend {type(backend).__name__} does not support tool calls")


# --------------------------------------------------------------------------- #
# Text tool mode (--sim-tool-mode text): the tool is described in the prompt and calls are read from the model's
# text, so the backend's own tool-call parsing is never involved. Needed for models/servers whose native tool
# parsing drops or garbles replies (an empty reply with tokens generated is the usual sign).
# --------------------------------------------------------------------------- #
TOOL_TEXT_PROTOCOL = """

TOOL PROTOCOL: you have one tool, simulate_move:
""" + json.dumps({"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS}) + """

To dry-run a move, reply with ONLY a tool call in exactly this form, and nothing else:
<tool_call>
{"name": "simulate_move", "arguments": {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0, "span_to": "(1,0)"}}}
</tool_call>
Leave out "block" for a remove, and leave out "span_to" for a small block. The result comes back inside
<tool_response></tool_response> tags. After simulating, reply with your final answer: ONE PLACE/REMOVE/CLARIFY line
and no tool call."""


def _to_text_protocol(messages: List[Dict]) -> List[Dict]:
    """OpenAI-style history (assistant tool_calls / tool results) -> plain text turns."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            body = m.get("content") or ""
            for tc in m.get("tool_calls") or []:
                call = {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                body += ("\n" if body else "") + "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"
            out.append({"role": "assistant", "content": body or "(no reply)"})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def tool_chat(backend, messages: List[Dict], use_tools: bool, seed: Optional[int], force: bool = False):
    """-> (text, calls, assistant_message, usage). backend.tool_mode: 'native' (default) or 'text'."""
    if getattr(backend, "tool_mode", "native") != "text":
        return _tool_chat_native(backend, messages, use_tools, seed, force)
    text, _, _, usage = _tool_chat_native(backend, _to_text_protocol(messages), False, seed, False)
    calls = _text_tool_calls(text) if use_tools else []
    return text, calls, {"role": "assistant", "content": text or None}, usage


# --------------------------------------------------------------------------- #
# Tool loop (port of generate_move_with_tools) + final-format retries
# --------------------------------------------------------------------------- #
def run_tool_loop(backend, prompt: str, cfg, allowed: set, seed: Optional[int],
                  board: Board, target: Board, views: Dict, mode: str) -> Dict:
    from run_full_info_builder import (BackendError, REASONING_ADDENDUM, SINGLE_LINE_ADDENDUM, _FORMAT_LINES,
                                       extract_move_line, is_generation_abort, parse_move)
    n = cfg.max_simulations
    # CRAFT's tool-mode system message, verbatim. The harness's usual "output EXACTLY ONE line, no JSON" addendum
    # is NOT added: it contradicts calling a tool and made small models skip simulate_move. The one-line
    # format is requested explicitly when the final answer is asked for.
    system = tool_system_message(n) + (REASONING_ADDENDUM if cfg.reasoning else TOOL_ROUND_ADDENDUM)
    tool_mode = getattr(cfg, "sim_tool_mode", None) or "native"
    if backend is not None:
        backend.tool_mode = tool_mode
    if tool_mode == "text":
        system += TOOL_TEXT_PROTOCOL
    full_prompt = prompt + tool_addendum(n, mode)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": full_prompt}]
    raws, usages, sims = [], [], []
    used, final_text, forced, nudges, misses = 0, None, None, 0, 0   # misses = consecutive replies without a call
    require = getattr(cfg, "sim_require", 0) or 0

    def call(use_tools, force=False):
        for k in range(MAX_EMPTY_RETRIES + 1):
            try:
                text, calls, amsg, usage = tool_chat(backend, messages, use_tools,
                                                     None if seed is None else seed + k, force)
            except BackendError as exc:
                if is_generation_abort(exc) and k < MAX_EMPTY_RETRIES:
                    print(f"    [SIM generation aborted by server: {(exc.body or '')[:120]!r}] "
                          f"retrying with a new seed ({k + 1}/{MAX_EMPTY_RETRIES})")
                    continue
                raise
            raws.append(text if not calls else (text + "\n" if text else "") + json.dumps(
                [{"name": c["name"], "arguments": c["arguments"]} for c in calls]))
            usages.append(usage)
            if (text or "").strip() or calls:
                break
            u = usage or {}
            print(f"    [SIM empty reply] completion_tokens={u.get('completion_tokens')} "
                  f"done_reason={u.get('done_reason')}"
                  + ("  <- hit the token cap before any visible text (reasoning tokens count against "
                     "--max-tokens: raise it or lower --reasoning-effort)" if u.get("done_reason") == "length" else
                     "  <- tokens were generated but nothing came back (tool-call parsing swallowed the reply?)"
                     if (u.get("completion_tokens") or 0) > 3 else "")
                  + (f"; retrying ({k + 1}/{MAX_EMPTY_RETRIES})" if k < MAX_EMPTY_RETRIES else ""))
        return text, calls, amsg

    try:
        while used < n:
            need = used < require
            text, calls, amsg = call(True, force=need)
            messages.append(amsg)
            if not calls:
                if need and misses < MAX_SIM_NUDGES:        # --sim-require: tool use is mandatory
                    misses += 1
                    nudges += 1
                    if amsg.get("content") is None:
                        amsg["content"] = "(no reply)"
                    print(f"    [SIM nudge {misses}/{MAX_SIM_NUDGES}] model answered without calling simulate_move "
                          f"({used}/{require} required calls so far); re-prompting")
                    print(f"      model said: {(text or '').strip()[:300]!r}")
                    messages.append({"role": "user", "content": (
                        "You have not called simulate_move yet. You MUST call simulate_move on your intended "
                        "move before giving a final answer. Call the simulate_move tool now." if used == 0 else
                        f"You MUST call simulate_move {require} times before your final answer; so far {used}. "
                        "Call the simulate_move tool again now: if the last simulation failed, simulate a "
                        "corrected move; otherwise simulate your intended move again or an alternative.")})
                    continue
                final_text = text                           # final answer without (more) simulation
                break
            misses = 0                                      # a tool call resets the consecutive-miss counter
            if used + len(calls) > n:                       # round would exceed the budget
                messages.pop()
                messages.append({"role": "user", "content": (
                    f"You have used {used}/{n} simulations. No more simulate_move calls allowed. "
                    "Output your FINAL move now in PLACE/REMOVE/CLARIFY text format.")})
                final_text, _, amsg = call(False)
                messages.append(amsg)
                forced = "over_budget_round"
                break
            used += len(calls)
            for c in calls:
                if c.get("name") != TOOL_NAME:
                    result = {"ok": False, "error": f"unknown tool {c.get('name')}", "hint": "Use simulate_move."}
                else:
                    result = simulate_move(board, target, views, c.get("arguments") or {}, cfg.sim_score)
                    if not result["ok"]:                     # CRAFT re-wraps the hint here
                        m = re.search(r"span_to=(\(\d,\d\))", result.get("hint", ""))
                        partner = f" The correct span_to is {m.group(1)}." if m else ""
                        result["hint"] = (f"Move failed: {result['error']}.{partner} "
                                          "Fix this specific move and try again. "
                                          "Do NOT submit a move that already failed in simulation.")
                sims.append({"move": _coerce_move(c.get("arguments") or {}), "result": result})
                ok = "ok" if result["ok"] else "FAIL"
                print(f"    [SIM {ok}] {sims[-1]['move'].get('action')} {sims[-1]['move'].get('block', '')} "
                      f"{sims[-1]['move'].get('position')} L{sims[-1]['move'].get('layer')} "
                      f"progress={result.get('overall_progress', '-')}")
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": TOOL_NAME,
                                 "content": json.dumps(result)})
            if used >= n:
                messages.append({"role": "user", "content": (
                    f"Simulation budget exhausted ({n}/{n}). Output your FINAL move now in "
                    "PLACE/REMOVE/CLARIFY text format. Do NOT call simulate_move again.")})
                final_text, _, amsg = call(False)
                messages.append(amsg)
                forced = "budget_exhausted"
                break
    except BackendError as exc:
        print(f"    [SIM backend error] {str(exc)[:300]}")
        raws.append(f"<backend error: {exc}>")
        return {"move": {"action": "parse_error", "error": f"backend error: {exc}"}, "raw": raws, "usage": usages,
                "prompt": full_prompt, "attempts": len(raws), "simulations": sims, "sim_calls": used,
                "sim_forced_final": forced, "sim_nudges": nudges, "sim_require": require}

    print(f"    [SIM summary] {used} simulation call(s) this turn"
          + (f", {nudges} nudge(s)" if nudges else "")
          + (f", forced final: {forced}" if forced else "")
          + ("  <- model answered without using the tool" if used == 0 else ""))
    order = [a for a in ("place", "remove", "done", "clarify") if a in allowed]
    move = {"action": "parse_error", "error": "no final answer"}
    for attempt in range(cfg.max_retries + 1):
        move = parse_move(extract_move_line(final_text or "", last=cfg.reasoning))
        if move["action"] not in allowed and move["action"] != "parse_error":
            move = {"action": "parse_error", "error": f"action {move['action'].upper()} not allowed here"}
        if move["action"] != "parse_error" or attempt == cfg.max_retries:
            break
        print(f"    [retry {attempt + 1}] unparseable final answer, re-prompting for format")
        print(f"      model said: {(final_text or '').strip()[:300]!r}")
        messages.append({"role": "user", "content":
                         "That was not in the required format. Reply with ONE line only, starting with "
                         + ", ".join(a.upper() + ":" for a in order) + " using exactly:\n"
                         + "\n".join(l for a in order for l in _FORMAT_LINES[a]) + "\nNo other text."})
        try:
            final_text, _, amsg = call(False)
        except BackendError as exc:
            raws.append(f"<backend error: {exc}>")
            break
        messages.append(amsg)

    return {"move": move, "raw": raws, "usage": usages, "prompt": full_prompt, "attempts": len(raws),
            "simulations": sims, "sim_calls": used, "sim_forced_final": forced, "sim_nudges": nudges,
            "sim_require": require}


def sim_turn_fields(decision: Dict, move: Dict) -> Dict:
    """Per-turn diagnostics: did the Builder submit what it simulated?"""
    sims = decision.get("simulations")
    if sims is None:
        return {}
    acting = move.get("action") in ("place", "remove")
    matched = [s for s in sims if acting and move_matches(move, s["move"])]
    return {
        "simulations": sims,
        "sim_calls": decision.get("sim_calls", 0),
        "sim_forced_final": decision.get("sim_forced_final"),
        "sim_nudges": decision.get("sim_nudges", 0),
        "sim_required_unmet": decision.get("sim_calls", 0) < (decision.get("sim_require") or 0),
        "final_was_simulated_ok": any(s["result"]["ok"] for s in matched) if acting and sims else None,
        "final_failed_in_sim": (bool(matched) and all(not s["result"]["ok"] for s in matched)) if acting else None,
    }


# --------------------------------------------------------------------------- #
# Probe: does this model do tool calling at all, independent of the game?
#   python craft_sim_tool.py --backend ollama --model qwen2.5:14b-instruct
# --------------------------------------------------------------------------- #
def probe_tool_calling(backend, trials: int = 3) -> Dict:
    sys_msg = "You are a Builder agent. Use the simulate_move tool when asked."
    if getattr(backend, "tool_mode", "native") == "text":
        sys_msg += TOOL_TEXT_PROTOCOL
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": ("The board is empty. Before answering, dry-run placing a small yellow block "
                                         "(code ys) at (0,0) on layer 0 by calling simulate_move, then wait for the "
                                         "result.")}]
    out = {"structured": 0, "text_form": 0, "none": 0, "replies": []}
    for _ in range(trials):
        text, calls, _, _ = tool_chat(backend, msgs, True, None)
        structured = bool(calls) and not any(str(c["id"]).startswith("call_txt_") for c in calls)
        if not (text or "").strip() and not calls:
            out.setdefault("empty", 0)
            out["empty"] += 1
        if structured:
            out["structured"] += 1
        elif calls:
            out["text_form"] += 1
        else:
            out["none"] += 1
        out["replies"].append({"text": (text or "")[:300], "calls": [c["arguments"] for c in calls]})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Probe a model's tool-calling support")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai", "gemini", "anthropic"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tool-mode", choices=["native", "text"], default="native")
    ap.add_argument("--reasoning-effort", default=None, choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    a = ap.parse_args()
    from run_full_info_builder import make_backend
    a.skip_preflight = False
    be = make_backend(a.backend, a.model, a.base_url, a.api_key_env, 0.1, 400, a, a.reasoning_effort)
    be.tool_mode = a.tool_mode
    r = probe_tool_calling(be, a.trials)
    print(json.dumps(r, indent=2))
    n = a.trials
    if r["structured"] == n:
        print(f"\nOK: {a.model} returned a structured tool call in {n}/{n} trials.")
    elif r["structured"] or r["text_form"]:
        print(f"\nPARTIAL: structured={r['structured']}, call written as text={r['text_form']}, none={r['none']} of {n}. "
              "Text-form calls are recovered by the harness; 'none' means the model ignored the tool.")
    else:
        print(f"\nNO TOOL CALLS in {n} trials. Check that this model/template supports tools "
              "(for Ollama: `ollama show <model>` should list 'tools' under Capabilities), or use another model.")
