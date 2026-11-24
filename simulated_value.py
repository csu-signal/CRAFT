"""
Simulated "value if followed" for every director message, and a check of whom the
recorded Builder was really following.

PART 1  (recordings, converted by craft_recordings_frames.py)
   For each recorded turn, the Builder's actual move is read from the board change and
   compared with every director's located request and with the target:
     followed a request, correct      instruction-following that worked
     followed a request, wrong        instruction-following that failed
     correct, requested by nobody     correct move not traceable to any instruction
     wrong, requested by nobody       other
   (placements and removals separately; turns with no change or a complex change counted).

PART 2  (labels)
   For every director message with a located request, the request is SIMULATED on the
   recorded board -- the environment is deterministic -- giving the one-step value of
   following that director:
     place  +1 the block is the target's block at that cell and layer (next free layer)
            -1 placeable but not the target's block
             0 not placeable there (the harness would reject it)
     remove +1 removes a wrong block from the top of a stack, -1 a correct one, 0 invalid
   Two versions: v_exec (located in the stated frame: where the Builder puts it -- what
   following the director yields in a harness with the D2/D3 mismatch) and v_meant (the
   cell the director meant -- what following yields once the harness is fixed). The
   message's PRIMARY request (its first located request) is used. Labels are written to
   --out for every recorded director-turn (blank when no request was located).

PART 3  (validation on live games where one director was heard: Random, Policy, MM6 ...)
   Does the simulated value of the heard message predict the actual outcome? Progress per
   turn by v_exec, and Builder fidelity (the Builder's move equals the simulated move).

    python simulated_value.py --recordings craft_recordings_games.json \
        --live haiku_games.html [--frame-check frame_check.py] [--out simulated_labels.csv]
"""
import argparse
import contextlib
import csv
import io
import json
import os
import re
import runpy
import sys
from collections import Counter, defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--recordings", required=True, help="output of craft_recordings_frames.py")
ap.add_argument("--live", default=None, help="games-viewer HTML/JSON of live games (for validation)")
ap.add_argument("--frame-check", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_check.py"))
ap.add_argument("--out", default="simulated_labels.csv")
ap.add_argument("--reversed-dirs", default="D2,D3", help="directors whose target list was reversed (recordings and pre-fix live games)")
args = ap.parse_args()
DIRS = ("D1", "D2", "D3")

_argv = sys.argv
sys.argv = [args.frame_check, "--games", args.recordings]
with contextlib.redirect_stdout(io.StringIO()):
    FC = runpy.run_path(args.frame_check)
sys.argv = _argv
extract, STATED = FC["extract"], FC["STATED"]
REV = {d.strip() for d in args.reversed_dirs.split(",") if d.strip()}
MEANT = {d: (STATED[d][::-1] if d in REV else STATED[d]) for d in DIRS}


def load_games(path):
    if path.endswith(".json"):
        return json.load(open(path))
    html = open(path, encoding="utf-8", errors="replace").read()
    script = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S)[0]
    i = script.find("const DATA = ") + len("const DATA = ")
    return json.JSONDecoder().raw_decode(script[i:])[0]


def primary_request(text, d):
    """first located place/remove request: (intent, colour, size, pos, layer, cell_abs)."""
    for m in extract(text, d):
        if m["intent"] in ("request", "remove") and m["pos"] is not None and m["layer"] is not None:
            return m
    return None


def simulate(m, frame, board, target):
    """one-step value of executing request m located through frame."""
    cell = m.get("cell_abs") or frame[m["pos"]]
    L = m["layer"]
    stack, tgt = board.get(cell, []), target.get(cell) or []
    if m["intent"] == "remove":
        if not stack or L != len(stack) - 1 or stack[-1][0] != m["color"]:
            return 0, cell, L
        correct_there = L < len(tgt) and tgt[L] == stack[-1]
        return (-1 if correct_there else 1), cell, L
    if L != len(stack):
        return 0, cell, L                                         # not the next free layer: rejected
    want = tgt[L] if L < len(tgt) else None
    ok = want is not None and want[0] == m["color"] and (m["size"] is None or want[1] == m["size"])
    lower_ok = all(k < len(tgt) and stack[k] == tgt[k] for k in range(L))
    return (1 if ok and lower_ok else -1), cell, L
    
    
def simulate_fixed(m, d, board, target):
    """v_meant_fixed (hand-off 3.2): the board -- placeability, the stack, the block on
    top -- at the STATED-frame cell, where the Builder puts or removes the block; the
    target at the MEANT cell, the view the director read its colours from."""
    if m.get("cell_abs"):
        return simulate(m, STATED[d], board, target)[0]
    placed, meant = STATED[d][m["pos"]], MEANT[d][m["pos"]]
    return simulate(m, STATED[d], board, dict(target, **{placed: target.get(meant) or []}))[0]


def board_move(before, after):
    """('place'|'remove'|'none'|'complex', [(cell, layer, block)])"""
    cells = set(before) | set(after)
    grown = [c for c in cells if len(after.get(c, [])) == len(before.get(c, [])) + 1 and after[c][:-1] == before.get(c, [])]
    shrunk = [c for c in cells if len(after.get(c, [])) == len(before.get(c, [])) - 1 and after.get(c, []) == before[c][:-1]]
    changed = [c for c in cells if after.get(c, []) != before.get(c, [])]
    if not changed:
        return "none", []
    if grown and len(grown) == len(changed) and len(grown) <= 2:
        return "place", [(c, len(after[c]) - 1, after[c][-1]) for c in grown]
    if shrunk and len(shrunk) == len(changed) and len(shrunk) <= 2:
        return "remove", [(c, len(before[c]) - 1, before[c][-1]) for c in shrunk]
    return "complex", []


# ====================================================================== PART 1 + 2: recordings
REC = load_games(args.recordings)
cats = Counter()
by_dir_follow = Counter()
labels_out = []
for g in REC["games"]:
    x = g["columns"][next(iter(g["columns"]))]
    target = g["target"]
    before = x.get("start_board") or {}
    for t in x["turns"]:
        after = t.get("board_after") or before
        reqs = {}
        for d in DIRS:
            text = ((t.get("messages") or {}).get(d) or {}).get("text") or ""
            m = primary_request(text, d) if text.strip() else None
            v_exec = v_meant = v_fixed = None
            if m:
                v_exec, cell, L = simulate(m, STATED[d], before, target)
                v_meant = simulate(m, MEANT[d], before, target)[0] if not m.get("cell_abs") else v_exec
                v_fixed = simulate_fixed(m, d, before, target)
                reqs[d] = (m, cell, L)
            labels_out.append({"model_label": g.get("model_label", ""), "structure_id": g["structure_id"],
                               "turn": t["turn"], "director": d, "intent": m["intent"] if m else "",
                               "v_exec": "" if v_exec is None else v_exec, "v_meant": "" if v_meant is None else v_meant,
                               "v_meant_fixed": "" if v_fixed is None else v_fixed})
        kind, moves = board_move(before, after)
        if kind in ("none", "complex"):
            cats[kind] += 1
        else:
            cell, L, block = moves[0]
            matched = [d for d, (m, c, l) in reqs.items()
                       if ((m["intent"] == "request") == (kind == "place")) and c in [mv[0] for mv in moves] and l == L
                       and m["color"] == block[0] and (m["size"] is None or m["size"] == block[1])]
            tgt = target.get(cell) or []
            if kind == "place":
                correct = L < len(tgt) and tgt[L] == block
            else:
                correct = not (L < len(tgt) and tgt[L] == block)      # removing a wrong block is the correct removal
            cats[(kind, "followed a request" if matched else "requested by nobody", "correct" if correct else "wrong")] += 1
            for d in matched:
                by_dir_follow[(kind, d, correct)] += 1
        before = after

print("=" * 100 + "\nPART 1. RECORDINGS: WHOM DID THE RECORDED BUILDER FOLLOW?\n" + "=" * 100)
tot = sum(cats.values())
print(f"   turns: {tot}   no change: {cats['none']}   complex change: {cats['complex']}")
for kind in ("place", "remove"):
    n_k = sum(v for k, v in cats.items() if isinstance(k, tuple) and k[0] == kind)
    if not n_k:
        continue
    print(f"\n   {'PLACEMENTS' if kind == 'place' else 'REMOVALS'} ({n_k}):")
    for src in ("followed a request", "requested by nobody"):
        for ok in ("correct", "wrong"):
            v = cats[(kind, src, ok)]
            print(f"      {src:<22}{ok:<9}{v:>6}  ({100 * v / n_k:4.1f}%)")
    print("      matched requests by director (correct / wrong): " + "   ".join(
        f"{d} {by_dir_follow[(kind, d, True)]}/{by_dir_follow[(kind, d, False)]}" for d in DIRS))

print("\n" + "=" * 100 + "\nPART 2. SIMULATED VALUE-IF-FOLLOWED LABELS (recordings)\n" + "=" * 100)
print(f"   {'':<4}{'messages':>10}{'labelled':>10}" + "".join(f"{f'{k}=+1/0/-1':>22}" for k in ("v_exec", "v_meant")))
for d in DIRS:
    L_ = [r for r in labels_out if r["director"] == d]
    lab = [r for r in L_ if r["v_exec"] != ""]
    dist = lambda k: "/".join(f"{100 * np.mean([r[k] == v for r in lab]):.0f}%" for v in (1, 0, -1)) if lab else "-"
    print(f"   {d:<4}{len(L_):>10}{len(lab):>10}{dist('v_exec'):>22}{dist('v_meant'):>22}")
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(labels_out[0]))
    w.writeheader(); w.writerows(labels_out)
print(f"   labels written to {args.out} ({len(labels_out)} director-turns; blank where no request was located)")

# ====================================================================== PART 3: validation
if args.live:
    LIVE = load_games(args.live)
    print("\n" + "=" * 100 + "\nPART 3. VALIDATION ON LIVE GAMES WHERE ONE DIRECTOR WAS HEARD\n" + "=" * 100)
    for c in LIVE["columns"]:
        rows = []
        fid = Counter()
        for g in LIVE["games"]:
            x = g["columns"].get(c)
            if not x:
                continue
            board = x.get("start_board") or {}
            prev = (x.get("start_craft") or {}).get("op") or 0.0
            for t in x["turns"]:
                op = t["craft"]["op"] if t.get("craft") else prev
                heard = t.get("heard") or []
                if len(heard) == 1:
                    d = heard[0]
                    text = ((t.get("messages") or {}).get(d) or {}).get("text") or ""
                    m = primary_request(text, d) if text.strip() else None
                    v = None
                    if m:
                        v, cell, L = simulate(m, STATED[d], board, g["target"])
                        mv = t.get("move") or {}
                        if mv.get("position"):
                            fid["checked"] += 1
                            fid["same cell+layer"] += mv.get("position") == cell and (mv.get("layer") or 0) == L
                    rows.append((v, op - prev, bool(t.get("executed"))))
                prev = op
                board = t.get("board_after") or board
        if len(rows) < 50:
            continue
        print(f"\n   [{c}]  {len(rows)} single-director turns" + (
            f"   Builder move at the simulated cell and layer: {100 * fid['same cell+layer'] / max(fid['checked'], 1):.0f}% of {fid['checked']}" if fid["checked"] else ""))
        for v, lab in ((1, "v_exec = +1 (correct if followed)"), (0, "v_exec =  0 (not placeable)"),
                       (-1, "v_exec = -1 (wrong if followed)"), (None, "no located request")):
            sel = [r for r in rows if r[0] == v]
            if sel:
                d_ = np.array([r[1] for r in sel])
                se = d_.std(ddof=1) / np.sqrt(len(d_)) if len(d_) > 1 else float("nan")
                print(f"      {lab:<36} n={len(sel):<4} progress/turn {d_.mean():+.4f} (SE {se:.4f})   executed {100 * np.mean([r[2] for r in sel]):.0f}%")
        lab_rows = [r for r in rows if r[0] is not None]
        if len(lab_rows) > 10:
            print(f"      corr(v_exec, progress) = {np.corrcoef([r[0] for r in lab_rows], [r[1] for r in lab_rows])[0, 1]:+.2f}")
print("\nRead: PART 1 -- a large 'requested by nobody, correct' share means recorded progress did not come from"
      "\ninstructions. PART 3 -- if progress rises steeply from v_exec -1 to +1 in the live games (especially Random),"
      "\nthe simulated labels measure what following a director is worth, and can be used as a dense training target.")
