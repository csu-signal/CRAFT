#!/usr/bin/env python3
"""
Self-tests for the full-information CRAFT harness. No model or network needed.

    python test_full_info.py path/to/structures_dataset_20.json
    # or: STRUCTURES=path/to/structures_dataset_20.json pytest -q test_full_info.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from craft_full_info_env import (
    DIRECTORS, Board, OraclePolicy, ViewsPolicy, check_structure, compute_metrics,
    load_structures, min_moves, target_views,
)
from run_full_info_builder import extract_move_line, main as run_main, parse_move

STRUCTURES = os.environ.get("STRUCTURES") or (sys.argv[1] if len(sys.argv) > 1 else "structures_dataset_20.json")


def _data():
    return load_structures(STRUCTURES)


# ---- CRAFT reference IoU, copied verbatim (logic) from run_single_builder.py
def craft_calculate_iou_board(current, target):
    intersection = union = 0
    for coord in current.keys():
        a, b = set(current[coord]), set(target[coord])
        intersection += len(a & b)
        union += len(a | b)
    return intersection / union if union > 0 else 0.0


def test_dataset_consistency():
    for e in _data():
        assert not check_structure(e), (e["id"], check_structure(e))


def test_recomputed_views_match_stored_colours():
    sizes = set()
    for e in _data():
        rv, dv = target_views(e, "recompute"), e["director_views"]
        for d in DIRECTORS:
            for k in range(3):
                for a, b in zip(rv[d][f"row_{k}"], dv[d][f"row_{k}"]):
                    assert a["color"] == b["color"], (e["id"], d, k)
                    sizes.add(b["size"])
    print(f"  stored view sizes present in dataset: {sorted(sizes)}")


def test_physics():
    b = Board()
    r = b.place("gs", "(0,1)", 1)
    assert not r.ok and r.error_type == "layer"
    r = b.place("gl", "(0,0)", 0)
    assert not r.ok and r.error_type == "span"
    r = b.place("gl", "(0,0)", 0, "(1,1)")
    assert not r.ok and r.error_type == "span"                       # diagonal
    assert b.place("ys", "(0,0)", 0).ok
    r = b.place("gl", "(0,0)", 1, "(0,1)")
    assert not r.ok and r.error_type == "span"                       # unequal heights
    assert b.place("bs", "(0,1)", 0).ok
    assert b.place("gl", "( 0 , 0 )", 1, "(0,1)").ok                 # tolerant position parsing
    assert b.spans[1] == [("(0,0)", "(0,1)")]
    assert b.place("rs", "(0,0)", 2).ok
    r = b.place("rs", "(0,0)", 3)
    assert not r.ok and r.error_type == "height"
    r = b.remove("(0,0)", 1, "(0,1)")
    assert not r.ok and r.error_type == "layer"                      # not the top
    assert b.remove("(0,0)", 2).ok
    r = b.remove("(0,1)", 1)
    assert not r.ok and r.error_type == "span"                       # large needs span_to
    r = b.remove("(0,1)", 1, "(1,1)")
    assert not r.ok and r.error_type == "span"                       # wrong partner
    assert b.remove("(0,1)", 1, "(0,0)").ok
    assert b.stacks["(0,0)"] == ["ys"] and b.stacks["(0,1)"] == ["bs"] and b.spans[1] == []
    r = b.remove("(2,2)", 0)
    assert not r.ok and r.error_type == "empty"


def test_parser():
    cases = {
        "PLACE:gs:(0,0):0:CONFIRM:ok": ("place", "gs", "(0,0)", 0, None),
        "Sure!\n```\nPLACE:gl:(0,0):0:(1,0):CONFIRM:x\n```": ("place", "gl", "(0,0)", 0, "(1,0)"),
        "**REMOVE:(2,2):0:(2,1):CONFIRM:x**": ("remove", None, "(2,2)", 0, "(2,1)"),
        "REMOVE:bl:(0,0):1:CONFIRM:x": ("remove", None, "(0,0)", 1, None),
        "Final move: PLACE:os:( 1, 2 ):2:CONFIRM:reason: with colon": ("place", "os", "(1,2)", 2, None),
        "place:GS:(0,0):0": ("place", "gs", "(0,0)", 0, None),
    }
    for text, (a, blk, pos, layer, span) in cases.items():
        m = parse_move(extract_move_line(text, last=False))
        assert m["action"] == a and m["position"] == pos and m["layer"] == layer and m["span_to"] == span, (text, m)
        if blk:
            assert m["block"] == blk
    assert parse_move(extract_move_line("DONE: all views match", False))["action"] == "done"
    assert parse_move(extract_move_line("I think corners first.", False))["action"] == "parse_error"
    # reasoning mode takes the LAST move line and ignores <think> content
    txt = "<think>PLACE:rs:(0,0):0:CONFIRM:no</think>\nMaybe PLACE:gs:(0,0):0 ...\nPLACE:bs:(0,0):0:CONFIRM:yes"
    assert parse_move(extract_move_line(txt, last=True))["block"] == "bs"


def _play(policy, e, cap=40):
    tgt, views = Board.from_structure(e), target_views(e)
    b, t = Board(), 0
    while t < cap:
        m = policy.decide({"board": b, "target": tgt, "views": views})["move"]
        if m["action"] == "done":
            break
        assert b.apply(m).ok, m
        t += 1
    return b, t, compute_metrics(b, tgt, views)


def test_oracle_completes_in_min_moves():
    for e in _data():
        b, t, m = _play(OraclePolicy(), e)
        assert m["completion"] == 1.0 and m["views_exact"], e["id"]
        assert t == min_moves(Board.from_structure(e)) <= 20, (e["id"], t)


def test_views_policy_satisfies_views():
    for e in _data():
        _, _, m = _play(ViewsPolicy(), e)
        assert m["views_exact"], e["id"]


def test_iou_parity_with_craft():
    for e in _data():
        tgt = Board.from_structure(e)
        b, _, m = _play(ViewsPolicy(), e)
        assert abs(m["iou"] - craft_calculate_iou_board(b.stacks, tgt.stacks)) < 1e-12


def test_cli_scripted():
    with tempfile.TemporaryDirectory() as tmp:
        for be in ("scripted-oracle", "scripted-views"):
            out = Path(tmp) / be
            run_main(["--structures", STRUCTURES, "--out-dir", str(out), "--backend", be, "--limit", "3"])
            s = json.loads((out / "_summary.json").read_text())
            assert s["n_games"] == 3 and s["final"]["view_match"]["mean"] == 1.0
            assert len(s["turnwise_mean"]["progress"]) == 20


class _FakeBackend:
    """Returns canned responses in order; records calls."""
    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, seed):
        self.calls.append(messages)
        return self.replies.pop(0), {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


def test_chat_for_move_retry_and_allowed_actions():
    import argparse
    from run_full_info_builder import chat_for_move
    cfg = argparse.Namespace(reasoning=False, max_retries=2)
    be = _FakeBackend(["no idea", "CLARIFY:which one?"])
    d = chat_for_move(be, "sys", "prompt", cfg, {"place", "remove", "clarify"}, None)
    assert d["move"]["action"] == "clarify" and d["attempts"] == 2 and len(be.calls[1]) == 4
    be = _FakeBackend(["CLARIFY:which one?", "DONE:x", "PLACE:gs:(0,0):0:CONFIRM:ok"])
    d = chat_for_move(be, "sys", "prompt", cfg, {"place", "remove"}, None)   # clarify/done not allowed
    assert d["move"]["action"] == "place" and d["attempts"] == 3


def test_director_prompts_and_parsing():
    from craft_directors import craft_builder_prompt, director_prompt, parse_director_response
    e = _data()[0]
    views, b = target_views(e), Board()
    p_all = director_prompt("D3", "skeptical", "all", views, b, "Turn 1 (now):")
    p_own = director_prompt("D3", "skeptical", "own", views, b, "Turn 1 (now):")
    blob = p_all.split("from each wall)\n    ", 1)[1].split("\n", 1)[0]
    assert json.loads(blob) == views                                   # all three views, valid JSON
    blob = p_own.split("from your side)\n    ", 1)[1].split("\n", 1)[0]
    assert json.loads(blob) == views["D3"] and "ALL THREE" not in p_own
    bp = craft_builder_prompt("D1: put a small blue in my bottom left", b.stacks, "all")
    assert "D1: put a small blue in my bottom left" in bp and "NO information separation" in bp
    assert "NO information separation" not in craft_builder_prompt("x", b.stacks, "own")
    assert parse_director_response("<think>a</think><message>hi</message>")["public_message"] == "hi"
    assert parse_director_response("<think>a</think>\nplace it")["public_message"] == "place it"
    assert parse_director_response("<message>open only")["public_message"] == "open only"
    assert parse_director_response("")["public_message"] == "No message provided"


def test_simulate_move():
    from craft_sim_tool import simulate_move
    e = _data()[0]
    tgt, views, b = Board.from_structure(e), target_views(e), Board()
    r = simulate_move(b, tgt, views, {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}})
    assert r["ok"] and r["correctness"]["structurePlacement"] and r["blocks_correct"] == 1
    r = simulate_move(b, tgt, views, {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 1}})
    assert not r["ok"] and "stack height" in r["hint"]
    assert b.stacks["(0,0)"] == []                                     # never mutates the real board
    r = simulate_move(b, tgt, views, {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}},
                      score="views")
    assert r["ok"] and r["blocks_total"] == 81


def test_tool_loop_budget_and_modes():
    import argparse
    import craft_sim_tool
    e = _data()[0]
    tgt, views = Board.from_structure(e), target_views(e)
    good = {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}}
    script = [  # round 1: 2 calls (used=2); round 2: 2 calls would exceed 3 -> forced final
        ("", [{"id": "a", "name": "simulate_move", "arguments": good},
              {"id": "b", "name": "simulate_move", "arguments": {"move": dict(good["move"], layer=2)}}]),
        ("", [{"id": "c", "name": "simulate_move", "arguments": good}] * 2),
        ("PLACE:ys:(0,0):0:CONFIRM:simulated", []),
    ]
    seen = []

    def fake_tool_chat(backend, messages, use_tools, seed, force=False):
        seen.append((use_tools, messages[1]["content"]))
        text, calls = script.pop(0)
        return text, calls, craft_sim_tool._openai_style_assistant(text, calls), {"prompt_tokens": 1}

    orig = craft_sim_tool.tool_chat
    craft_sim_tool.tool_chat = fake_tool_chat
    try:
        cfg = argparse.Namespace(max_simulations=3, sim_score="target", reasoning=False, max_retries=2)
        d = craft_sim_tool.run_tool_loop(None, "PROMPT", cfg, {"place", "remove", "clarify"}, None,
                                         Board(), tgt, views, "directors")
    finally:
        craft_sim_tool.tool_chat = orig
    assert d["move"]["action"] == "place" and d["sim_calls"] == 2 and d["sim_forced_final"] == "over_budget_round"
    assert [s["result"]["ok"] for s in d["simulations"]] == [True, False]
    assert seen[-1][0] is False                                        # final answer requested without tools
    assert "Simulate each director's instruction" in seen[0][1]        # CRAFT addendum in director mode
    fields = craft_sim_tool.sim_turn_fields(d, d["move"])
    assert fields["final_was_simulated_ok"] is True and fields["final_failed_in_sim"] is False


def _run_loop_with_script(script, **cfg_over):
    import argparse
    import craft_sim_tool
    e = _data()[0]
    tgt, views = Board.from_structure(e), target_views(e)
    seen = []

    def fake_tool_chat(backend, messages, use_tools, seed, force=False):
        seen.append({"use_tools": use_tools, "force": force, "n_msgs": len(messages), "last": messages[-1]})
        text, calls = script.pop(0)
        return text, calls, craft_sim_tool._openai_style_assistant(text, calls), {"prompt_tokens": 1}

    orig = craft_sim_tool.tool_chat
    craft_sim_tool.tool_chat = fake_tool_chat
    try:
        cfg = argparse.Namespace(max_simulations=3, sim_score="target", reasoning=False, max_retries=2,
                                 sim_require=cfg_over.pop("sim_require", 1), **cfg_over)
        d = craft_sim_tool.run_tool_loop(None, "PROMPT", cfg, {"place", "remove", "clarify"}, None,
                                         Board(), tgt, views, "single")
    finally:
        craft_sim_tool.tool_chat = orig
    return d, seen


def test_sim_require_nudges_until_tool_is_used():
    good = {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}}
    call = {"id": "a", "name": "simulate_move", "arguments": good}
    d, seen = _run_loop_with_script([
        ("PLACE:ys:(0,0):0:CONFIRM:x", []),          # ignores the tool -> nudge 1
        ("PLACE:ys:(0,0):0:CONFIRM:y", []),          # still no tool call -> nudge 2
        ("", [call]),                                # finally calls the tool
        ("PLACE:ys:(0,0):0:CONFIRM:simulated", []),  # final answer
    ])
    assert d["sim_calls"] == 1 and d["sim_nudges"] == 2 and d["move"]["action"] == "place"
    assert all(x["force"] for x in seen[:3])        # tool_choice forced while the requirement is unmet
    assert "MUST call simulate_move" in seen[1]["last"]["content"] if seen[1]["last"]["role"] == "user" else True
    assert seen[3]["force"] is False or seen[3]["use_tools"] is True   # requirement met: no longer forced
    from craft_sim_tool import sim_turn_fields
    f = sim_turn_fields(d, d["move"])
    assert f["sim_nudges"] == 2 and f["sim_required_unmet"] is False


def test_sim_require_counts_consecutive_misses_only():
    """One call per prompt (what Qwen-7B does) must still be able to reach --sim-require 3."""
    good = {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}}
    call = {"id": "a", "name": "simulate_move", "arguments": good}
    text = ("PLACE:ys:(0,0):0:CONFIRM:x", [])
    d, _ = _run_loop_with_script([text, ("", [call]), text, ("", [call]), text, ("", [call]),
                                  ("PLACE:ys:(0,0):0:CONFIRM:final", [])], sim_require=3)
    assert d["sim_calls"] == 3 and d["sim_nudges"] == 3 and d["sim_forced_final"] == "budget_exhausted"
    from craft_sim_tool import sim_turn_fields
    assert sim_turn_fields(d, d["move"])["sim_required_unmet"] is False


def test_sim_require_gives_up_after_max_nudges():
    d, _ = _run_loop_with_script([("PLACE:ys:(0,0):0:CONFIRM:x", [])] * 3)
    assert d["sim_calls"] == 0 and d["sim_nudges"] == 2 and d["move"]["action"] == "place"
    from craft_sim_tool import sim_turn_fields
    assert sim_turn_fields(d, d["move"])["sim_required_unmet"] is True


def test_tool_choice_payloads():
    import run_full_info_builder as rb
    import craft_sim_tool
    sent = []

    def fake_post(url, payload, headers, timeout, max_attempts=6):
        sent.append(payload)
        if payload.get("tool_choice") == "required" and len(sent) == 2:        # 2nd test: endpoint rejects it
            raise rb.BackendError("HTTP 400", 400, '{"error": "invalid tool_choice value"}')
        if "/v1/messages" in url:
            return {"content": [{"type": "text", "text": "ok"}], "usage": {}}
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    orig = rb._post_json
    rb._post_json = fake_post
    try:
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        oa = rb.OpenAICompatBackend("m", "http://x/v1", "k", 0.1, 100, 5)
        craft_sim_tool.tool_chat(oa, msgs, True, None, force=True)
        assert sent[-1]["tool_choice"] == "required"
        craft_sim_tool.tool_chat(oa, msgs, True, None, force=True)             # rejected, retried with auto
        assert sent[-1]["tool_choice"] == "auto"
        craft_sim_tool.tool_chat(oa, msgs, True, None, force=False)
        assert sent[-1]["tool_choice"] == "auto"
        an = rb.AnthropicBackend("m", "http://x", "k", 0.1, 100, 5)
        craft_sim_tool.tool_chat(an, msgs, True, None, force=True)
        assert sent[-1]["tool_choice"] == {"type": "any"}
        craft_sim_tool.tool_chat(an, msgs, False, None, force=True)
        assert sent[-1]["tool_choice"] == {"type": "none"}
    finally:
        rb._post_json = orig


def test_sim_require_flag_validation():
    for argv in (["--sim-require", "1"], ["--sim-tool", "--sim-require", "9"]):
        try:
            run_main(["--structures", STRUCTURES, "--out-dir", "/tmp/_y", "--backend", "ollama", "--model", "m",
                      "--skip-preflight"] + argv)
        except SystemExit as exc:
            assert exc.code not in (0, None)
        else:
            raise AssertionError(argv)


def test_director_tag_fragment_cleanup():
    from craft_directors import parse_director_response, strip_tag_fragments
    for raw in ("<think>a</think>\n\n=message>\n    Let's go", "<think>a</think>\n:message>\n  Let's go",
                "message>Let's go", "<message>Let's go</message", "<think>a</think>\n[message]\nLet's go"):
        out = parse_director_response(raw)
        assert out["public_message"] == "Let's go" and out["tag_fragment_stripped"], (raw, out)
    ok = parse_director_response("<think>a</think><message>Message the builder about it</message>")
    assert ok["public_message"] == "Message the builder about it" and not ok["tag_fragment_stripped"]
    assert parse_director_response("<think>a</think>\n=message>\n")["public_message"] == "No message provided"
    assert strip_tag_fragments("Put a message > here") == "Put a message > here"


def test_text_tool_call_recovery():
    from craft_sim_tool import _text_tool_calls
    mv = {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}
    want = [{"move": mv}]
    pos = {
        "qwen tag": '<tool_call>\n{"name": "simulate_move", "arguments": {"move": ' + json.dumps(mv) + '}}\n</tool_call>',
        "fenced": '```json\n{"name": "simulate_move", "arguments": {"move": ' + json.dumps(mv) + '}}\n```',
        "args as string": '{"name": "simulate_move", "arguments": ' + json.dumps(json.dumps({"move": mv})) + '}',
        "parameters key, no move wrapper": '{"name": "simulate_move", "parameters": ' + json.dumps(mv) + '}',
        "bare move wrapper": 'Let me check first.\n{"move": ' + json.dumps(mv) + '}',
        "bare action object": json.dumps(mv),
        "python kwargs": 'simulate_move(action="place", block="ys", position="(0,0)", layer=0)',
        "python dict": 'simulate_move({"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}})',
        "SIMULATE line": "SIMULATE:PLACE:ys:(0,0):0:CONFIRM:checking",
        "nested function key": '{"function": {"name": "simulate_move", "arguments": ' + json.dumps({"move": mv}) + '}}',
    }
    for name, text in pos.items():
        got = _text_tool_calls(text)
        assert [c["arguments"] for c in got] == want, (name, got)
    two = _text_tool_calls('<tool_call>{"name":"simulate_move","arguments":{"move":' + json.dumps(mv) + '}}</tool_call>'
                           '<tool_call>{"name":"simulate_move","arguments":{"move":' + json.dumps(dict(mv, layer=1)) + '}}</tool_call>')
    assert len(two) == 2 and two[1]["arguments"]["move"]["layer"] == 1
    for name, text in {"plain move": "PLACE:ys:(0,0):0:CONFIRM:x", "prose": "I would use simulate_move first.",
                       "bare action beside a move line": json.dumps(mv) + "\nPLACE:ys:(0,0):0:CONFIRM:x",
                       "empty": "", "other tool": '{"name": "search", "arguments": {"q": 1}}'}.items():
        assert _text_tool_calls(text) == [], (name, _text_tool_calls(text))


def test_tool_mode_system_prompt_has_no_single_line_rule():
    import argparse
    import craft_sim_tool
    seen = []

    def fake(backend, messages, use_tools, seed, force=False):
        seen.append(messages[0]["content"])
        return "PLACE:ys:(0,0):0:CONFIRM:x", [], craft_sim_tool._openai_style_assistant("PLACE:ys:(0,0):0:CONFIRM:x", []), {}

    e = _data()[0]
    orig, craft_sim_tool.tool_chat = craft_sim_tool.tool_chat, fake
    try:
        cfg = argparse.Namespace(max_simulations=3, sim_score="target", reasoning=False, max_retries=2, sim_require=0)
        craft_sim_tool.run_tool_loop(None, "P", cfg, {"place", "remove", "clarify"}, None, Board(),
                                     Board.from_structure(e), target_views(e), "single")
    finally:
        craft_sim_tool.tool_chat = orig
    assert "EXACTLY ONE line" not in seen[0] and "tool-calling interface" in seen[0]
    assert "You may call simulate_move up to 3 times" in seen[0]          # CRAFT's message is still the base


def test_empty_replies_are_retried_not_counted_as_misses():
    good = {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}}
    call = {"id": "a", "name": "simulate_move", "arguments": good}
    d, _ = _run_loop_with_script([("", []), ("", []), ("", [call]),                  # 2 empties, then a call
                                  ("PLACE:ys:(0,0):0:CONFIRM:x", [])])
    assert d["sim_calls"] == 1 and d["sim_nudges"] == 0 and len(d["usage"]) == 4 and d["move"]["action"] == "place"
    d, _ = _run_loop_with_script([("", [])] * 3 + [("PLACE:ys:(0,0):0:CONFIRM:x", [])], sim_require=0)
    assert d["move"]["action"] == "place" and len(d["usage"]) == 4                  # gives up retrying after 2 retries


def test_text_tool_mode_roundtrip():
    """Backend that returns EMPTY replies whenever native tools are sent (the Ollama failure seen in practice)."""
    import argparse
    import run_full_info_builder as rb
    import craft_sim_tool
    payloads = []

    def fake_post(url, payload, headers, timeout, max_attempts=6):
        payloads.append(payload)
        if payload.get("tools"):
            return {"message": {"content": ""}, "prompt_eval_count": 50, "eval_count": 40, "done_reason": "stop"}
        last = payload["messages"][-1]["content"]
        if "<tool_response>" in last:
            return {"message": {"content": "PLACE:ys:(0,0):0:CONFIRM:simulated"}, "prompt_eval_count": 60, "eval_count": 8}
        call = {"name": "simulate_move", "arguments": {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}}}
        return {"message": {"content": "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"}, "prompt_eval_count": 50,
                "eval_count": 30}

    e = _data()[0]
    orig, rb._post_json = rb._post_json, fake_post
    try:
        be = rb.OllamaBackend("m", "http://x", 0.1, 400, 8192, 5)
        cfg = argparse.Namespace(max_simulations=3, sim_score="target", reasoning=False, max_retries=2,
                                 sim_require=1, sim_tool_mode="text")
        d = craft_sim_tool.run_tool_loop(be, "PROMPT", cfg, {"place", "remove", "clarify"}, None, Board(),
                                         Board.from_structure(e), target_views(e), "single")
        # same game in native mode fails exactly the way the log did: every reply is empty
        cfg2 = argparse.Namespace(**dict(vars(cfg), sim_tool_mode="native"))
        d2 = craft_sim_tool.run_tool_loop(be, "PROMPT", cfg2, {"place", "remove", "clarify"}, None, Board(),
                                          Board.from_structure(e), target_views(e), "single")
    finally:
        rb._post_json = orig
    assert d["sim_calls"] == 1 and d["move"]["action"] == "place" and d["sim_required_unmet"] if "sim_required_unmet" in d else True
    assert d["sim_calls"] == 1 and d["simulations"][0]["result"]["ok"]
    text_payloads = [p for p in payloads if "tools" not in p]
    assert text_payloads and "TOOL PROTOCOL" in text_payloads[0]["messages"][0]["content"]
    assert any("<tool_response>" in m["content"] for m in text_payloads[1]["messages"] if m["role"] == "user")
    assert any(m["role"] == "assistant" and "<tool_call>" in m["content"] for m in text_payloads[1]["messages"])
    assert d2["sim_calls"] == 0 and d2["move"]["action"] == "parse_error"           # native mode: swallowed replies


def _serve(handler_factory):
    import threading
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _handler(responder):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            d = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(d)

        def do_GET(self):
            self._send(200, {"models": [{"name": "stub:latest"}]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            code, obj = responder()
            self._send(code, obj)
    return H


def test_retry_message_shows_server_error_text():
    import contextlib
    import io
    import run_full_info_builder as rb
    calls = []

    def responder():
        calls.append(1)
        return (500, {"error": "llama runner process has terminated: exit status 2"}) if len(calls) == 1 \
            else (200, {"ok": True})

    srv, url = _serve(_handler(responder))
    orig_sleep, rb.time.sleep = rb.time.sleep, lambda *_: None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out = rb._post_json(url + "/api/chat", {}, {}, 5)
    finally:
        rb.time.sleep = orig_sleep
        srv.shutdown()
    assert out == {"ok": True}
    assert "llama runner process has terminated" in buf.getvalue() and "attempt 1/6" in buf.getvalue()


def test_run_stops_and_keeps_partial_log_when_backend_keeps_failing():
    for extra in ([], ["--directors"]):
        srv, url = _serve(_handler(lambda: (500, {"error": "llama runner process has terminated"})))
        with tempfile.TemporaryDirectory() as tmp:
            try:
                run_main(["--structures", STRUCTURES, "--out-dir", tmp, "--backend", "ollama", "--model", "stub",
                          "--base-url", url, "--skip-preflight", "--limit", "1", "--http-retries", "1",
                          "--timeout", "5"] + extra)
            except SystemExit as exc:
                assert exc.code == 1, exc.code
            else:
                raise AssertionError("run did not stop")
            finally:
                srv.shutdown()
            partials = list(Path(tmp).glob("*.partial.json"))
            assert len(partials) == 1 and not list(Path(tmp).glob("*_run0.json"))   # partial only, no finished game
            g = json.loads(partials[0].read_text())
            assert g["partial"] and len(g["turns"]) == 3, len(g["turns"])      # stopped after 3 failed turns


def test_partial_log_removed_after_a_finished_game():
    with tempfile.TemporaryDirectory() as tmp:
        run_main(["--structures", STRUCTURES, "--out-dir", tmp, "--backend", "scripted-views", "--limit", "1"])
        assert not list(Path(tmp).glob("*.partial.json")) and list(Path(tmp).glob("*_run0.json"))


def _body_handler(responder):
    """Like _handler, but the responder sees (path, request_body)."""
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self._reply(200, {"models": [{"name": "stub:latest"}]})

        def _reply(self, code, obj):
            d = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(d)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code, obj = responder(self.path, body)
            self._reply(code, obj)
    return H


def _gpt_like(payloads, move_line="PLACE:ys:(0,0):0:CONFIRM:x"):
    """Chat Completions stub following GPT-5.x rules: no max_tokens, temperature only with effort 'none',
    function tools only with effort 'none', strict tool-message schema; one tool call, then the final line."""
    def responder(path, body):
        payloads.append(body)
        err = lambda msg, param: (400, {"error": {"message": msg, "param": param}})            # noqa: E731
        if "max_tokens" in body:
            return err("Unsupported parameter: 'max_tokens' is not supported with this model. Use "
                       "'max_completion_tokens' instead.", "max_tokens")
        if body.get("tools") and body.get("reasoning_effort") != "none":
            return err("Function tools with reasoning_effort are not supported for gpt-5.6-luna in "
                       "/v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort "
                       "to 'none'.", "reasoning_effort")
        if body.get("temperature", 1) != 1 and body.get("reasoning_effort") != "none":
            return err("Unsupported value: 'temperature' does not support 0.1 with this model.", "temperature")
        for m in body["messages"]:
            if m["role"] == "tool" and "name" in m:
                return err("Additional properties are not allowed ('name' was unexpected)", "messages")
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        if body.get("tools") and not any(m["role"] == "tool" for m in body["messages"]):
            call = {"id": "call_1", "type": "function", "function": {"name": "simulate_move", "arguments": json.dumps(
                {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}})}}
            return 200, {"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}],
                         "usage": usage}
        return 200, {"choices": [{"message": {"content": move_line}, "finish_reason": "stop"}], "usage": usage}
    return responder


def test_gpt_style_endpoint_needs_effort_none_for_tools():
    import argparse
    import run_full_info_builder as rb
    import craft_sim_tool
    e = _data()[0]
    tgt, views = Board.from_structure(e), target_views(e)
    cfg = argparse.Namespace(max_simulations=3, sim_score="target", reasoning=False, max_retries=2, sim_require=1,
                             sim_tool_mode="native")
    for effort in (None, "none"):
        payloads = []
        srv, url = _serve(_body_handler(_gpt_like(payloads)))
        try:
            be = rb.OpenAICompatBackend("gpt-5.6-luna", url + "/v1", "k", 0.1, 400, 5, reasoning_effort=effort)
            d = craft_sim_tool.run_tool_loop(be, "PROMPT", cfg, {"place", "remove", "clarify"}, None, Board(), tgt,
                                             views, "single")
        finally:
            srv.shutdown()
        if effort is None:    # clear, actionable error instead of a silent failure
            assert d["move"]["action"] == "parse_error" and "--reasoning-effort none" in d["move"]["error"], d["move"]
        else:
            assert d["move"]["action"] == "place" and d["sim_calls"] == 1 and d["simulations"][0]["result"]["ok"]
            assert all(p.get("reasoning_effort") == "none" for p in payloads)
            assert any("max_completion_tokens" in p for p in payloads) and not any("max_tokens" in p for p in payloads[-2:])
            assert all("name" not in m for p in payloads for m in p["messages"] if m["role"] == "tool")
            assert [u.get("done_reason") for u in d["usage"]][-1] == "stop"


def test_reasoning_effort_flags_reach_builder_and_directors():
    payloads = []

    def responder(path, body):
        payloads.append(body)
        system = body["messages"][0]["content"]
        text = ("<think>t</think><message>Place a small yellow block in my bottom left.</message>"
                if "You are Director" in system else "PLACE:ys:(0,0):0:CONFIRM:x")
        return 200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    srv, url = _serve(_body_handler(responder))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run_main(["--structures", STRUCTURES, "--out-dir", tmp, "--backend", "openai", "--model", "m",
                      "--base-url", url + "/v1", "--limit", "1", "--turns", "2", "--directors",
                      "--reasoning-effort", "none"])
        finally:
            srv.shutdown()
        cfg = json.loads((Path(tmp) / "_config.json").read_text())
    assert cfg["reasoning_effort"] == "none" and cfg["director_reasoning_effort"] == "none"
    assert payloads and all(p.get("reasoning_effort") == "none" for p in payloads)
    assert any("You are Director" in p["messages"][0]["content"] for p in payloads)       # Directors got it too


def test_generation_abort_is_reseeded_not_waited_out():
    """Ollama 'token repeat limit' 500s are deterministic per request: retry with a new seed, no backoff."""
    import run_full_info_builder as rb
    seen_seeds, sleeps = [], []

    def responder(path, body):
        seed = body["options"]["seed"]
        seen_seeds.append(seed)
        if seed % 2 == 0:
            return 500, {"error": "prediction aborted, token repeat limit reached"}
        system = body["messages"][0]["content"]
        text = ("<think>t</think><message>Put a small yellow block in my bottom left.</message>"
                if "You are Director" in system else "CLARIFY:which one?")
        return 200, {"message": {"content": text}, "prompt_eval_count": 10, "eval_count": 5}

    srv, url = _serve(_body_handler(responder))
    orig_sleep, rb.time.sleep = rb.time.sleep, lambda x: sleeps.append(x)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run_main(["--structures", STRUCTURES, "--out-dir", tmp, "--backend", "ollama", "--model", "stub",
                      "--base-url", url, "--skip-preflight", "--limit", "1", "--turns", "3", "--directors"])
        finally:
            rb.time.sleep = orig_sleep
            srv.shutdown()
        g = json.loads(next(Path(tmp).glob("*_run0.json")).read_text())
    assert any(x % 2 == 0 for x in seen_seeds) and not sleeps, sleeps                   # aborts happened, no waiting
    assert len(g["turns"]) == 3 and not any(r["error"] for t in g["turns"] for r in t["directors"])
    assert all(t["action"] == "clarify" for t in g["turns"])                              # builder answered every turn


def test_truncated_director_replies_are_dropped_not_forwarded():
    """A Director reply cut off by the token cap must not reach the Builder as its 'instruction'."""
    builder_prompts = []
    for api in ("openai", "ollama"):
        def responder(path, body):
            system = body["messages"][0]["content"]
            if "You are Director" in system:
                cut = "<think>Let me think. D1 bottom layer wants y at (0,0), blue large across (1,0) impossible"
                if api == "openai":
                    return 200, {"choices": [{"message": {"content": cut}, "finish_reason": "length"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 512}}
                return 200, {"message": {"content": cut}, "done_reason": "length", "prompt_eval_count": 10,
                             "eval_count": 512}
            builder_prompts.append(body["messages"][1]["content"])
            text = "CLARIFY:what next?"
            if api == "openai":
                return 200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            return 200, {"message": {"content": text}, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5}

        srv, url = _serve(_body_handler(responder))
        with tempfile.TemporaryDirectory() as tmp:
            try:
                run_main(["--structures", STRUCTURES, "--out-dir", tmp, "--backend", api, "--model", "stub",
                          "--base-url", url + ("/v1" if api == "openai" else ""), "--skip-preflight", "--limit", "1",
                          "--turns", "2", "--directors"])
            finally:
                srv.shutdown()
            g = json.loads(next(Path(tmp).glob("*_run0.json")).read_text())
            summary = json.loads((Path(tmp) / "_summary.json").read_text())
        recs = [r for t in g["turns"] for r in t["directors"]]
        assert recs and all(r["truncated"] and r["silent"] for r in recs), api
        assert all("no director gave an instruction" in t["discussion"] for t in g["turns"])
        assert summary["director_stats"]["truncated_messages"] == len(recs)
    assert builder_prompts and not any("blue large across" in p for p in builder_prompts)   # nothing leaked


def test_complete_message_is_kept_even_if_finish_reason_is_length():
    from craft_directors import parse_director_response
    import re
    text = "<think>a</think><message>Put a small yellow block in my bottom left.</message>"
    assert re.search(r"<message>.*?</message>", text, re.DOTALL | re.IGNORECASE)    # the guard's condition
    assert parse_director_response(text)["public_message"].startswith("Put a small yellow")


def test_preflight_stops_before_any_director_call_when_tools_are_refused():
    payloads = []
    srv, url = _serve(_body_handler(_gpt_like(payloads)))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run_main(["--structures", STRUCTURES, "--out-dir", tmp, "--backend", "openai", "--model", "gpt-5.4-mini",
                      "--base-url", url + "/v1", "--limit", "1", "--directors", "--sim-tool",
                      "--reasoning-effort", "medium"])
        except SystemExit as exc:
            assert isinstance(exc.code, str) and "[preflight]" in exc.code and "--sim-tool-mode text" in exc.code, exc.code
        else:
            raise AssertionError("preflight did not stop the run")
        finally:
            srv.shutdown()
    assert payloads and not any("You are Director" in p["messages"][0]["content"] for p in payloads)  # no Director spend
    assert len(payloads) <= 4                                                                         # just the probe


def test_gpt_style_text_mode_works_with_medium_reasoning():
    import argparse
    import run_full_info_builder as rb
    import craft_sim_tool
    payloads = []

    def responder(path, body):
        payloads.append(body)
        if body.get("tools"):
            return 400, {"error": {"message": "Function tools with reasoning_effort are not supported ...",
                                   "param": "reasoning_effort"}}
        if "max_tokens" in body:
            return 400, {"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}}
        if body.get("temperature", 1) != 1 and body.get("reasoning_effort") != "none":
            return 400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.1", "param": "temperature"}}
        usage = {"prompt_tokens": 100, "completion_tokens": 300, "total_tokens": 400,
                 "completion_tokens_details": {"reasoning_tokens": 250}}
        if "<tool_response>" in body["messages"][-1]["content"]:
            text = "PLACE:ys:(0,0):0:CONFIRM:simulated"
        else:
            call = {"name": "simulate_move", "arguments": {"move": {"action": "place", "block": "ys", "position": "(0,0)", "layer": 0}}}
            text = "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"
        return 200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": usage}

    e = _data()[0]
    srv, url = _serve(_body_handler(responder))
    try:
        be = rb.OpenAICompatBackend("gpt-5.4-mini", url + "/v1", "k", 0.1, 4000, 5, reasoning_effort="medium")
        cfg = argparse.Namespace(max_simulations=3, sim_score="target", reasoning=False, max_retries=2, sim_require=1,
                                 sim_tool_mode="text")
        d = craft_sim_tool.run_tool_loop(be, "PROMPT", cfg, {"place", "remove", "clarify"}, None, Board(),
                                         Board.from_structure(e), target_views(e), "single")
    finally:
        srv.shutdown()
    assert d["move"]["action"] == "place" and d["sim_calls"] == 1 and d["simulations"][0]["result"]["ok"]
    assert all("tools" not in p and p.get("reasoning_effort") == "medium" for p in payloads if "max_tokens" not in p)
    from run_full_info_builder import token_totals
    assert token_totals(d["usage"])["reasoning_tokens"] == 500                                      # 2 calls x 250 reported


def test_oracle_and_sim_tool_are_exclusive():
    try:
        run_main(["--structures", STRUCTURES, "--out-dir", "/tmp/_x", "--oracle-in-prompt", "--sim-tool"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse accepted both flags")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} tests passed")
