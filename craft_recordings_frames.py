"""
Frame test on the ORIGINAL CRAFT recordings.

1. Converts the recordings (HF dataset rows: one row per game turn) into the games
   format frame_check.py reads, with each director's target view built the way the
   original run_craft.py built it (D2's and D3's rows reversed). Then run
       python frame_check.py --games craft_recordings_games.json --sample 30
   and read SECTION 0: if D2's and D3's requests match the target in the "views order"
   (the reversed JSON the directors received) while their observations match the board
   in the "mirrored order" (= the stated perspective), the recordings have the mismatch.

2. Prints a Builder-placement check that needs no attribution field: on turns where
   exactly one director located a single place request and the board gained exactly one
   block, where did the Builder put it (the director's stated cell, the reversed cell,
   elsewhere), and was the placed block correct for the target?

    python craft_recordings_frames.py --hf-dataset Abhijnan/craft-benchmark-lean \
        --structures ../../data/structures_dataset_20.json [--split train] [--models gpt-4o,Qwen7B] \
        [--out craft_recordings_games.json]

Run from the directory that holds frame_check.py (it reuses its extractor). Silent
director messages and turns without a board are handled; the HF columns used are
model_label, structure_id, turn_number, D1_message..D3_message, structure_before.
"""
import argparse
import ast
import contextlib
import io
import json
import os
import re
import runpy
import sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--hf-dataset", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--structures", required=True, help="dataset file the recordings were played on (entries with id, structure)")
ap.add_argument("--models", default=None, help="comma-separated substrings of model_label to keep")
ap.add_argument("--out", default="craft_recordings_games.json")
ap.add_argument("--frame-check", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_check.py"))
args = ap.parse_args()
DIRS = ("D1", "D2", "D3")
CNAME = {"r": "red", "b": "blue", "g": "green", "y": "yellow", "o": "orange"}
STATED = {"D1": ["(0,0)", "(1,0)", "(2,0)"], "D2": ["(0,0)", "(0,1)", "(0,2)"], "D3": ["(0,2)", "(1,2)", "(2,2)"]}
SILENT = {"", "no message provided", "none", "silent", "(silent)"}


def board_of(b):
    """board as {"(i,j)": [codes]} from a dict or a JSON / Python-literal string."""
    if b is None:
        return {}
    if isinstance(b, str):
        try:
            b = json.loads(b)
        except Exception:
            try:
                b = ast.literal_eval(b)
            except Exception:
                return {}
    out = {}
    for k, v in (b or {}).items():
        nums = re.findall(r"\d", str(k))
        if len(nums) >= 2:
            out[f"({nums[0]},{nums[1]})"] = [str(x) for x in (v or []) if x and str(x) != "ns"]
    return out


def views_as_run_craft(target):
    """per-director target views as the original run_craft.py gave them: stated order, D2/D3 rows reversed."""
    def cell(c, layer):
        st = target.get(c) or []
        return {"color": CNAME.get(st[layer][0], "none") if layer < len(st) else "none", "size": 1}
    views = {}
    for d, cells in STATED.items():
        rows = {f"row_{l}": [cell(c, l) for c in cells] for l in range(3)}
        views[d] = {r: (v[::-1] if d in ("D2", "D3") else v) for r, v in rows.items()}
    return views


# ------------------------------------------------------------------ load
from datasets import load_dataset
rows = list(load_dataset(args.hf_dataset, split=args.split))
structs = json.load(open(args.structures))
structs = structs if isinstance(structs, list) else structs.get("structures", structs)
targets = {s["id"]: board_of(s["structure"]) for s in structs}
keep = [m.strip().lower() for m in args.models.split(",")] if args.models else None

games = defaultdict(list)
missing_target = Counter()
for r in rows:
    label = str(r.get("model_label"))
    if keep and not any(k in label.lower() for k in keep):
        continue
    sid = r.get("structure_id")
    if sid not in targets:
        missing_target[sid] += 1
        continue
    games[(label, sid)].append(r)

DATA = {"columns": ["CRAFT"], "games": []}
for (label, sid), rs in sorted(games.items()):
    rs.sort(key=lambda r: int(r["turn_number"]))
    boards = [board_of(r.get("structure_before")) for r in rs]
    turns = []
    for i, r in enumerate(rs):
        msgs = {}
        for d in DIRS:
            tx = str(r.get(f"{d}_message") or "").strip()
            msgs[d] = {"text": "" if tx.lower() in SILENT else tx}
        after = boards[i + 1] if i + 1 < len(boards) else (board_of(r.get("structure_after")) or boards[i])
        turns.append({"turn": int(r["turn_number"]), "messages": msgs, "board_after": after})
    DATA["games"].append({"structure_id": sid, "model_label": label, "complexity": "",
                          "target": targets[sid], "views": views_as_run_craft(targets[sid]),
                          "columns": {"CRAFT": {"start_board": boards[0], "turns": turns}}})
json.dump(DATA, open(args.out, "w"))
print(f"converted {len(DATA['games'])} recorded games ({sum(len(g['columns']['CRAFT']['turns']) for g in DATA['games'])} turns)"
      f" from {len({g['model_label'] for g in DATA['games']})} model labels -> {args.out}")
if missing_target:
    print(f"   rows skipped for structure ids not in --structures: {dict(missing_target)}")

# ------------------------------------------------------------------ Builder placement check
_argv = sys.argv
sys.argv = [args.frame_check, "--games", args.out]
with contextlib.redirect_stdout(io.StringIO()):
    FC = runpy.run_path(args.frame_check)
sys.argv = _argv
extract = FC["extract"]

res = Counter()
for g in DATA["games"]:
    x = g["columns"]["CRAFT"]
    before = x["start_board"]
    for t in x["turns"]:
        after = t["board_after"]
        grown = [(c, len(after.get(c, [])) - 1) for c in after if len(after.get(c, [])) == len(before.get(c, [])) + 1]
        changed = [c for c in set(after) | set(before) if after.get(c, []) != before.get(c, [])]
        reqs = {}
        for d in DIRS:
            ms = [m for m in extract(t["messages"][d]["text"], d)
                  if m["intent"] == "request" and m["pos"] is not None and m["layer"] is not None]
            if ms:
                reqs[d] = ms
        if len(reqs) == 1 and len(grown) == 1 and len(changed) == 1:
            d, ms = next(iter(reqs.items()))
            if len(ms) == 1 and ms[0]["pos"] != 1 and not ms[0].get("absolute"):   # left/right only, frame-dependent
                m = ms[0]
                stated, reverse = STATED[d][m["pos"]], STATED[d][2 - m["pos"]]
                cell, layer = grown[0]
                block = after[cell][layer]
                where = "stated cell" if cell == stated else "reversed cell" if cell == reverse else "other cell"
                tgt = g["target"].get(cell) or []
                res[(d, where, "correct" if layer < len(tgt) and tgt[layer] == block else "wrong")] += 1
        before = after
print("\nBUILDER PLACEMENT (turns where one director located a single left/right place request and one block was added):")
print(f"   {'':<4}{'stated cell':>22}{'reversed cell':>22}{'other cell':>20}   (correct / wrong for the target)")
for d in DIRS:
    cells = []
    for where in ("stated cell", "reversed cell", "other cell"):
        c, w = res[(d, where, "correct")], res[(d, where, "wrong")]
        cells.append(f"{c} / {w}")
    print(f"   {d:<4}{cells[0]:>22}{cells[1]:>22}{cells[2]:>20}")
print("\nIf the recordings have the mismatch: the Builder puts D2's and D3's blocks in the stated cell (it follows the"
      "\nprompts), but they are mostly wrong there, while D1's are mostly right. Now run:"
      f"\n   python {os.path.basename(args.frame_check)} --games {args.out} --sample 30")
