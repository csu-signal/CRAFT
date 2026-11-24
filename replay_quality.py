"""
Are replayed director messages as good under the message-only format as under <notes>?

Reads refusal_replay.csv (from refusal_replay.py) and the raw game logs it came from,
and scores every replayed message with the frame-aware extractor from frame_check.py,
against that turn's board (structure_before) and the game's target_structure:

  located     the message contains a located request (position + layer)
  placeable   a request at the next free layer of its cell, not already built
  good        a request whose block is the target's block there, at the next free layer
              (correct AND placeable, located where the Builder puts it: the stated frame,
              right for runs with --ref_fix)

Refused or fragmentary replies count as unusable. Two views:
  per call     good/placeable rate over ALL calls (a refusal counts as not good)
  per usable   rates among usable messages only (message quality, refusals aside)
and a PAIRED comparison by prompt (the same director-turn under two variants), so the
difference between variants is not confounded by which prompts were sampled.

    python replay_quality.py refusal_replay.csv --logs craft_results/<run>/**/craft_structure_*.json \
        [--frame-check frame_check.py] [--compare message,notes]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--logs", nargs="+", required=True, help="raw game logs (the files named in the CSV's 'log' column)")
ap.add_argument("--frame-check", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_check.py"))
ap.add_argument("--compare", default="message,notes", help="two variants for the paired comparison")
args = ap.parse_args()

# ---------------------------------------------------------------- extractor (code from frame_check.py, without running it)
src = open(args.frame_check).read()
ns = {"re": re, "Counter": Counter, "np": np}
exec(src[src.index("DIRS = ("):src.index("# ------------------------------------------------------------------ data")], ns)
exec(src[src.index("STATED = {"):src.index("def frames_of")], ns)
exec(src[src.index("# ------------------------------------------------------------------ extraction"):
         src.index("# ------------------------------------------------------------------ optional rescoring")], ns)
extract, score_mention, STATED = ns["extract"], ns["score_mention"], ns["STATED"]


def norm(board):
    out = {}
    for k, v in (board or {}).items():
        nums = re.findall(r"\d", str(k))
        if len(nums) >= 2:
            out[f"({nums[0]},{nums[1]})"] = [str(x) for x in (v or []) if x and str(x) != "ns"]
    return out


# ---------------------------------------------------------------- boards and targets from the logs
files = {}
for pat in args.logs:
    for p in glob.glob(pat):
        files[os.path.basename(p)] = p
STATE = {}
for name, path in files.items():
    try:
        d = json.load(open(path))
    except Exception:
        continue
    for g in d.get("games", []):
        tgt = norm(g.get("target_structure"))
        for t in g.get("turns", []):
            STATE[(name, int(t["turn_number"]))] = (norm(t.get("structure_before")), tgt)

rows = list(csv.DictReader(open(args.csv)))
missing = Counter()
scored = []
for r in rows:
    key = (r["log"], int(r["turn"]))
    if key not in STATE:
        missing[r["log"]] += 1
        continue
    board, target = STATE[key]
    msg = r["message"] or ""
    usable = len(msg) > 3 and not msg.startswith("<")
    lab = {"located": 0, "placeable": 0, "good": 0}
    if usable:
        for m in extract(msg, r["director"]):
            if m["intent"] != "request" or m["pos"] is None or m["layer"] is None:
                continue
            lab["located"] = 1
            st = score_mention(m, STATED[r["director"]], target, board)
            lab["placeable"] |= int(st["next_free"] and not st["done"])
            lab["good"] |= int(st["target_match"] and st["next_free"])
    scored.append({**r, "usable": int(usable), **lab})
if missing:
    print(f"rows skipped (log not given or not found): {sum(missing.values())} from {len(missing)} logs, e.g. {next(iter(missing))}")
if not scored:
    raise SystemExit("nothing scored: pass the logs named in the CSV with --logs")

variants = [v for v in ("current", "message", "notes", "native") if any(x["variant"] == v for x in scored)]
print(f"\nscored {len(scored)} replies from {len({(x['log'], x['turn'], x['director']) for x in scored})} prompts\n")
print(f"{'variant':<10}{'calls':>7}{'usable':>9}{'  per call: placeable':>23}{'good':>8}{'   per usable: located':>24}{'placeable':>11}{'good':>8}")
for v in variants:
    R = [x for x in scored if x["variant"] == v]
    U = [x for x in R if x["usable"]]
    pct = lambda L, k: f"{100 * np.mean([x[k] for x in L]):.0f}%" if L else "-"
    print(f"{v:<10}{len(R):>7}{pct(R, 'usable'):>9}{pct(R, 'placeable'):>23}{pct(R, 'good'):>8}{pct(U, 'located'):>24}{pct(U, 'placeable'):>11}{pct(U, 'good'):>8}")

# ---------------------------------------------------------------- paired by prompt
a, b = [v.strip() for v in args.compare.split(",")]
by = defaultdict(lambda: defaultdict(list))
for x in scored:
    by[(x["log"], x["turn"], x["director"])][x["variant"]].append(x)
print(f"\nPAIRED BY PROMPT: {a} minus {b} (each prompt's mean over its repeats)")
for k, lab in (("good", "good, per call"), ("placeable", "placeable, per call"), ("usable", "usable")):
    diffs = [np.mean([x[k] for x in V[a]]) - np.mean([x[k] for x in V[b]]) for V in by.values() if V.get(a) and V.get(b)]
    if len(diffs) > 1:
        dd = np.array(diffs); se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"   {lab:<22} mean {dd.mean():+.3f}  SE {se:.3f}  (n={len(dd)} prompts; {a} better on {(dd > 0).sum()}, worse on {(dd < 0).sum()})")
print("\nRead: if the paired 'good' and 'placeable' differences are near zero (within about 2 SE), the written notes"
      "\nare not improving the instructions, and message-only (no refusals) is the format to adopt.")
