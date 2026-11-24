"""
Frame-aware analysis of director messages in full CRAFT games.

Each director describes one wall of the 3x3 grid from its own side. This script
  1. derives every director's frame (which board cell is its left / middle / right)
     from the game's own goal views, and checks from the messages which ordering
     the directors actually used;
  2. extracts block mentions from each message (colour, size, horizontal position,
     layer, and whether the clause requests, observes or removes), maps them to
     (cell, layer) through the speaker's frame, and scores them against the target
     and the current board;
  3. compares directors by the policy's rank on each turn (top / middle / bottom);
  4. tests convergence on the two corner cells that two directors share;
  5. in conditions where the Builder heard several directors, relates the quality of
     the message it followed to the outcome (move executed, progress change).

    python frame_check.py --games haiku_games.html [--columns Baseline "Policy (MM6)"]
                          [--sample 40] [--checkpoint exports/<policy>.pt --rescore-columns Baseline]

Policy ranks come from the scores logged in each turn (each column's own policy:
the plain checkpoint for Baseline / Policy / Random, MM6 for the MM6 column). With
--checkpoint, the listed columns are rescored with that checkpoint instead (run from
phase1_refactor with FEATURE_MODE=embeddings_only, as for drop_one_check.py).

--sample N prints N random messages with what was extracted, for checking the
extractor by hand before trusting the tables.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--games", required=True, help="games-viewer HTML (const DATA = ...) or the extracted JSON")
ap.add_argument("--columns", nargs="*", default=None)
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--by-model", action="store_true", help="section 0 per director model (games with a model_label)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--rescore-columns", nargs="*", default=[])
ap.add_argument("--frame", choices=("stated", "views"), default="stated",
                help="frame for scoring mentions in sections 1-4: 'stated' = the prompts' perspective lines, i.e. where "
                     "the Builder puts a request (default); 'views' = the order of the target-view JSON")
args = ap.parse_args()
DIRS = ("D1", "D2", "D3")
COLORS = {"red": "r", "blue": "b", "green": "g", "yellow": "y", "orange": "o"}
CNAME = {v: k for k, v in COLORS.items()}

# ------------------------------------------------------------------ data
if args.games.endswith(".json"):
    DATA = json.load(open(args.games))
else:
    html = open(args.games, encoding="utf-8", errors="replace").read()
    script = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S)[0]
    i = script.find("const DATA = ") + len("const DATA = ")
    DATA, _ = json.JSONDecoder().raw_decode(script[i:])
COLS = args.columns or DATA["columns"]
for c in COLS:
    if c not in DATA["columns"]:
        raise SystemExit(f"column {c!r} not in {DATA['columns']}")

# ------------------------------------------------------------------ frames
WALLS = [[f"({r},0)" for r in range(3)], [f"(0,{c})" for c in range(3)],
         [f"({r},2)" for r in range(3)], [f"(2,{c})" for c in range(3)]]
CANDIDATES = [w for w in WALLS] + [w[::-1] for w in WALLS]


def view_match(game, d, cells):
    ok = n = 0
    for k in range(3):
        row = game["views"][d].get(f"row_{k}") or []
        for i, cell in enumerate(cells):
            st = game["target"].get(cell) or []
            want = CNAME.get(st[k][0]) if k < len(st) else None
            got = (row[i] or {}).get("color") if i < len(row) and row[i] else None
            n += 1
            ok += (want or None) == (got or None)
    return ok / n


STATED = {"D1": ["(0,0)", "(1,0)", "(2,0)"], "D2": ["(0,0)", "(0,1)", "(0,2)"], "D3": ["(0,2)", "(1,2)", "(2,2)"]}


def frames_of(game):
    return {d: max(CANDIDATES, key=lambda c: view_match(game, d, c)) for d in DIRS}


# ------------------------------------------------------------------ extraction
REQ = re.compile(r"\b(need\w*|should|place\w*|put\w*|add\w*|stack\w*|want\w*|target|goal|must|let'?s|please|requir\w*|go(?:es)?\s+(?:on|at|in)|build)\b")
OBS = re.compile(r"\b(see|currently|there'?s|there is|i have|right now|already|is at|sits?|shows?|existing|is in|are in|looks?)\b")
REM = re.compile(r"\b(remov\w*|take\s+(?:\w+\s+){0,3}(?:off|out|away)|wrong|replac\w*|swap\w*|shouldn'?t|incorrect|get\s+rid)\b")
POS = {"left": 0, "leftmost": 0, "right": 2, "rightmost": 2}
LAYER = {"bottom": 0, "lowest": 0, "ground": 0, "base": 0, "foundation": 0, "top": 2, "upper": 2, "highest": 2, "uppermost": 2}
ORD = {"first": 0, "second": 1, "third": 2}
LAYER_NOUNS = {"layer", "row", "level", "tier"}


def tokens(clause):
    return re.findall(r"[a-z]+|\d", clause.lower().replace("-", " "))


REQ_CUES = {"need", "needs", "needed", "should", "place", "placing", "put", "add", "adding", "want", "wants", "target",
            "goal", "require", "requires", "build", "start", "go", "goes", "stack", "must"}
OBS_CUES = {"see", "seeing", "saw", "currently", "sits", "sitting", "shows", "showing", "existing", "placed", "has", "got", "having"}
REM_CUES = {"remove", "removing", "removed", "swap", "swapping", "replace", "replacing", "wrong", "clear", "clearing",
            "take", "pull", "incorrect"}
# along the side walls, "back" and "front" are positions: back = the cell on the back row
BACK_FRONT = {"D1": {"back": 0, "front": 2}, "D3": {"back": 2, "front": 0}}
BACK_FRONT_CELL = {"D1": {"back": "(0,0)", "front": "(2,0)"}, "D3": {"back": "(0,2)", "front": "(2,2)"}}


def clause_intent(cl):
    l = cl.lower()
    if REM.search(l):
        return "remove"
    if REQ.search(l):
        return "request"
    if OBS.search(l):
        return "observe"
    return "unknown"


def locate_words(tk, director):
    pos_at, lay_at = [], []
    for i, w in enumerate(tk):
        nxt = tk[i + 1] if i + 1 < len(tk) else ""
        prv = tk[i - 1] if i > 0 else ""
        if nxt == "wall" or nxt == "side":
            continue
        if w in POS:
            if w == "right" and nxt in ("there", "now", "here", "away", "on", "next", "above", "below", "after",
                                        "before", "into", "at", "in", "by", "so", "behind", "beside", "under"):
                continue                                      # adverb ("right there", "right on top")
            pos_at.append((i, POS[w]))
        elif w in ("middle", "center", "centre"):
            (lay_at if nxt in LAYER_NOUNS else pos_at).append((i, 1))
        elif w in LAYER:
            if w == "top" and prv == "on" and nxt == "of":
                continue                                      # relative ("on top of X"), not a layer
            lay_at.append((i, LAYER[w]))
        elif w in ORD and nxt in LAYER_NOUNS:
            lay_at.append((i, ORD[w]))
        elif w in ("l", "row") and nxt in ("0", "1", "2"):
            lay_at.append((i, int(nxt)))
        elif w in BACK_FRONT.get(director, {}) and prv in ("the", "my", "at", "in", "your", "toward", "towards") \
                and nxt not in ("to", "and", "wall", "up", "into", "out"):
            pos_at.append((i, ("abs", BACK_FRONT_CELL[director][w])))
    return pos_at, lay_at


def extract(text, director):
    """-> list of mentions {color, size, pos, layer, intent, clause}.
    Sentences are the unit (so 'I see yellow at my bottom left, but I need red there' keeps both
    colours at one location); intent is taken from the nearest preceding cue word; a run of three
    colours is read as left-to-right positions (reversed for 'right to left'); 'there' / 'that spot'
    inherits the previous located mention."""
    out, last, carried_layer = [], (None, None), None
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        tk = tokens(sent)
        if not tk:
            continue
        pos_at, lay_at = locate_words(tk, director)
        sentence_layer = carried_layer if not lay_at else None   # "my bottom foundation is wrong. I need blue on my left, ..."
        if lay_at:
            carried_layer = lay_at[-1][1]
        cues = [(i, "request" if w in REQ_CUES else "observe" if w in OBS_CUES else "remove")
                for i, w in enumerate(tk) if w in REQ_CUES or w in OBS_CUES or w in REM_CUES]
        fallback = clause_intent(sent)
        cidx = [i for i, w in enumerate(tk) if w in COLORS]
        cue_idx = {c[0] for c in cues}
        groups, cur = [], []
        for i in cidx:
            if cur and (i - cur[-1] > 4 or any(cur[-1] < j < i for j in cue_idx)):
                groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        rtl = "right to left" in sent.lower()
        has_coords = bool(re.search(r"\(\s*\d\s*,\s*\d\s*\)", sent))
        anaphora = bool(re.search(r"\bthere\b|that spot|same spot|that position|that cell", sent.lower()))
        def near(L, i):
            """position/layer word for the colour at i: the first one after it (before the next colour,
            within 6 tokens), else the closest one before it (after the previous colour, within 8)."""
            nxt_c = min([j for j in cidx if j > i], default=len(tk))
            prv_c = max([j for j in cidx if j < i], default=-1)
            fwd = [(j, v) for j, v in L if i < j < nxt_c and j - i <= 6]
            if fwd:
                return min(fwd)[1]
            bwd = [(j, v) for j, v in L if prv_c < j < i and i - j <= 8]
            return max(bwd)[1] if bwd else None
        for grp in groups:
            listed = len(grp) == 3 and (lay_at or "left to right" in sent.lower() or rtl)
            for k, i in enumerate(grp):
                size = None
                for j in range(max(0, i - 3), min(len(tk), i + 3)):
                    if tk[j] in ("small", "little", "single"):
                        size = "s"
                    elif tk[j] in ("large", "big", "long", "double", "wide"):
                        size = "l"
                prev_cue = [c for c in cues if c[0] < i and i - c[0] <= 10]
                intent = prev_cue[-1][1] if prev_cue else fallback
                if listed:
                    lay = min(((abs(j - grp[1]), v) for j, v in lay_at), default=(None, sentence_layer))[1] if lay_at else sentence_layer
                    pos, layer = (2 - k if rtl else k), lay
                else:
                    pos, layer = near(pos_at, i), near(lay_at, i)
                    if pos is not None and layer is None and sentence_layer is not None:
                        layer = sentence_layer
                    if anaphora and (pos is None or layer is None) and last != (None, None):
                        pos = last[0] if pos is None else pos
                        layer = last[1] if layer is None else layer
                cell_abs = None
                if isinstance(pos, tuple):                       # back/front: a physical cell, not left/right
                    cell_abs = pos[1]
                    pos = STATED[director].index(cell_abs) if cell_abs in STATED.get(director, []) else 1
                if pos is not None and layer is not None:
                    last = (pos, layer)
                out.append({"color": COLORS[tk[i]], "size": size, "pos": pos, "layer": layer,
                            "intent": intent, "clause": sent.strip()[:160], "cell_abs": cell_abs,
                            "absolute": cell_abs is not None or has_coords})
    return out


def score_mention(m, frame, target, board):
    """adds cell and truth flags to a complete mention (pos and layer known)."""
    cell = m.get("cell_abs") or frame[m["pos"]]
    st, bd = target.get(cell) or [], board.get(cell) or []
    L = m["layer"]
    tgt = st[L] if L < len(st) else None
    cur = bd[L] if L < len(bd) else None
    m = dict(m, cell=cell)
    m["target_match"] = tgt is not None and tgt[0] == m["color"] and (m["size"] is None or tgt[1] == m["size"])
    m["board_match"] = cur is not None and cur[0] == m["color"]
    m["next_free"] = len(bd) == L
    m["done"] = cur is not None and tgt is not None and cur == tgt
    return m


# ------------------------------------------------------------------ optional rescoring
RESCORED = {}
if args.checkpoint and args.rescore_columns:
    import torch
    import pipeline_cache
    import features
    from arm_state import ArmState, ConstraintBelief, Satisfaction
    model, _ = pipeline_cache.load_exported_policy(args.checkpoint)
    model = model.to("cpu").eval()
    for g in DATA["games"]:
        for c in args.rescore_columns:
            x = g["columns"].get(c)
            if not x:
                continue
            hist = {d: [] for d in DIRS}
            board = x.get("start_board") or {}
            for t in x["turns"]:
                msgs = {d: ((t.get("messages") or {}).get(d) or {}).get("text") or "" for d in DIRS}
                for d in DIRS:
                    hist[d].append(ArmState(f"{d}_message", g["structure_id"], "live", msgs[d], int(t["turn"]),
                                            ConstraintBelief(0), Satisfaction(0.0), board, None))
                enc = [features.encode_arm_state(hist[d]) for d in DIRS]
                with torch.no_grad():
                    out = model(torch.stack([e[0] for e in enc]).cpu(), torch.stack([e[1] for e in enc]).cpu())
                    out = out[0] if isinstance(out, tuple) else out
                    s = out.squeeze(-1).float().cpu().tolist()
                RESCORED[(g["structure_id"], c, int(t["turn"]))] = dict(zip(DIRS, s))
                board = t.get("board_after") or board

# ------------------------------------------------------------------ walk all turns
records = []          # one per spoken message
turn_recs = []        # one per turn
by_dir_hits = Counter()
by_model_hits = Counter()
alt_hits = Counter()  # frame validation: request target-match under each ordering hypothesis
frame_report = Counter()
for g in DATA["games"]:
    fr = frames_of(g)
    for d in DIRS:
        frame_report[(d, tuple(fr[d]))] += 1
    mirror = {d: fr[d][::-1] for d in DIRS}
    for c in COLS:
        x = g["columns"].get(c)
        if not x:
            continue
        board = x.get("start_board") or {}
        prev_op = (x.get("start_craft") or {}).get("op") or 0.0
        for t in x["turns"]:
            msgs = {d: ((t.get("messages") or {}).get(d) or {}).get("text") or "" for d in DIRS}
            scores = RESCORED.get((g["structure_id"], c, int(t["turn"]))) or (t.get("policy") or {}).get("scores") or {}
            speaking = [d for d in DIRS if msgs[d].strip()]
            per = {}
            for d in speaking:
                ms = extract(msgs[d], d)
                fs = STATED[d] if args.frame == "stated" else fr[d]
                full = [score_mention(m, fs, g["target"], board) for m in ms if m["pos"] is not None and m["layer"] is not None]
                for m in ms:
                    if m["pos"] is not None and m["layer"] is not None and m["intent"] == "request" and not m.get("absolute"):
                        alt_hits[("views order", score_mention(m, fr[d], g["target"], board)["target_match"])] += 1
                        alt_hits[("mirrored order", score_mention(m, mirror[d], g["target"], board)["target_match"])] += 1
                        pm = random.Random(hash((g["structure_id"], d, m["clause"])) & 0xffff).sample(fr[d], 3)
                        alt_hits[("random order", score_mention(m, pm, g["target"], board)["target_match"])] += 1
                    if m["pos"] is not None and m["layer"] is not None and m["intent"] in ("request", "observe") and m["pos"] != 1 and not m.get("absolute"):
                        for hyp, frm in (("views order", fr[d]), ("mirrored order", mirror[d])):
                            sm = score_mention(m, frm, g["target"], board)
                            key = "target_match" if m["intent"] == "request" else "board_match"
                            by_dir_hits[(d, m["intent"], hyp, sm[key])] += 1
                            by_model_hits[(g.get("model_label", "-"), d, m["intent"], hyp, sm[key])] += 1
                req = [m for m in full if m["intent"] == "request"]
                obs = [m for m in full if m["intent"] == "observe"]
                rem = [m for m in full if m["intent"] == "remove"]
                rec = {"structure": g["structure_id"], "column": c, "turn": int(t["turn"]), "director": d, "text": msgs[d],
                       "mentions": ms, "full": full,
                       "n_mentions": len(ms), "n_full": len(full),
                       "req_precision": np.mean([m["target_match"] for m in req]) if req else np.nan,
                       "actionable": any(m["target_match"] and m["next_free"] for m in req),
                       "placeable": any(m["next_free"] and not m["done"] for m in req),   # public: checks the board only
                       "redundant": np.mean([m["done"] for m in req]) if req else np.nan,
                       "obs_accuracy": np.mean([m["board_match"] for m in obs]) if obs else np.nan,
                       "valid_removal": np.mean([m["board_match"] and not m["target_match"] for m in rem]) if rem else np.nan,
                       "has_request": bool(req)}
                per[d] = rec
                records.append(rec)
            op = t["craft"]["op"] if t.get("craft") else prev_op
            turn_recs.append({"structure": g["structure_id"], "column": c, "turn": int(t["turn"]), "scores": scores,
                              "speaking": speaking, "heard": list(t.get("heard") or []), "per": per,
                              "tag": (t.get("tag") or {}).get("tag"), "executed": bool(t.get("executed")),
                              "action": t.get("action"), "d_op": op - prev_op, "frame": {d: (STATED[d] if args.frame == "stated" else fr[d]) for d in DIRS}})
            prev_op = op
            board = t.get("board_after") or board

# ------------------------------------------------------------------ report
fmt = lambda v: "   -  " if v != v else f"{100 * v:5.0f}%"
print("=" * 104 + "\n0. FRAMES (derived from each game's goal views) AND WHICH ORDERING THE DIRECTORS USED\n" + "=" * 104)
for (d, cells), n in sorted(frame_report.items()):
    print(f"   {d}: left/middle/right = {', '.join(cells)}   ({n} games)")
for hyp in ("views order", "mirrored order", "random order"):  # mirrored = the stated perspective for D2 and D3
    yes, no = alt_hits[(hyp, True)], alt_hits[(hyp, False)]
    print(f"   requests matching the target under the {hyp:<15} {fmt(yes / max(yes + no, 1))}  (n={yes + no})")
print("   views order = the order of the target-view JSON; mirrored = its reverse, which for D2 and D3 is the")
print("   perspective stated in both prompts (and used by the Builder). For D1 the two coincide with views order.")
print("\n   per director, left/right mentions only (the middle cell is the same in both orders):")
print(f"   {'':<4}{'requests -> target, views order':>34}{'mirrored':>10}{'observations -> board, views order':>38}{'mirrored':>10}")
for d in DIRS:
    cells = []
    for intent in ("request", "observe"):
        for hyp in ("views order", "mirrored order"):
            y, n_ = by_dir_hits[(d, intent, hyp, True)], by_dir_hits[(d, intent, hyp, False)]
            cells.append(f"{fmt(y / max(y + n_, 1))} (n={y + n_})")
    print(f"   {d:<4}{cells[0]:>34}{cells[1]:>10}{cells[2]:>38}{cells[3]:>10}")

if args.by_model:
    print("\n   by director model (left/right mentions; reversed = the target-view JSON order, stated = the prompts' order):")
    print(f"   {'model':<38}{'dir':<5}{'requests: reversed':>20}{'stated':>9}{'observations: reversed':>25}{'stated':>9}")
    models = sorted({k[0] for k in by_model_hits})
    for mdl in models:
        for d in ("D2", "D3"):
            cells = []
            for intent in ("request", "observe"):
                for hyp in ("views order", "mirrored order"):
                    y, n_ = by_model_hits[(mdl, d, intent, hyp, True)], by_model_hits[(mdl, d, intent, hyp, False)]
                    cells.append((y / (y + n_) if y + n_ else float("nan"), y + n_))
            if cells[0][1] + cells[2][1] < 20:
                continue
            f_ = lambda c: f"{100 * c[0]:4.0f}% ({c[1]:>3})" if c[1] else "     -     "
            print(f"   {mdl[:37]:<38}{d:<5}{f_(cells[0]):>20}{f_(cells[1]):>9}{f_(cells[2]):>25}{f_(cells[3]):>9}")
print("\n" + "=" * 104 + "\n1. EXTRACTION COVERAGE\n" + "=" * 104)
R = records
print(f"   spoken messages: {len(R)}   with any colour mention: {fmt(np.mean([r['n_mentions'] > 0 for r in R]))}"
      f"   with a fully located mention (position + layer): {fmt(np.mean([r['n_full'] > 0 for r in R]))}")
inten = Counter(m["intent"] for r in R for m in r["full"])
print(f"   located mentions by clause type: {dict(inten)}")

print("\n" + "=" * 104 + "\n2. MESSAGE QUALITY BY POLICY RANK (turns where all three spoke)\n" + "=" * 104)
keys = (("req_precision", "requests correct"), ("actionable", "has actionable request"), ("redundant", "requests already done"),
        ("obs_accuracy", "observations accurate"), ("valid_removal", "removals valid"), ("n_full", "located mentions"))
for c in COLS:
    rows = [tr for tr in turn_recs if tr["column"] == c and len(tr["speaking"]) == 3 and len(tr["scores"]) == 3]
    if not rows:
        continue
    print(f"\n   [{c}]  {len(rows)} turns" + ("   (rescored with --checkpoint)" if c in args.rescore_columns and RESCORED else "   (logged scores)"))
    print(f"   {'rank':<8}" + "".join(f"{lab:>25}" for _, lab in keys))
    for k, lab in ((0, "top"), (1, "middle"), (2, "bottom")):
        recs = [tr["per"][sorted(DIRS, key=lambda d: -tr["scores"][d])[k]] for tr in rows]
        cells = []
        for key, _ in keys:
            v = [float(r[key]) for r in recs if r[key] == r[key]]
            cells.append(f"{np.mean(v):>20.2f} (n={len(v)})" if key == "n_full" else f"{fmt(np.mean(v) if v else np.nan):>17} (n={len(v):>3})")
        print(f"   {lab:<8}" + "".join(f"{x:>25}" for x in cells))

print("\n" + "=" * 104 + "\n3. SHARED CORNERS: do two directors who both locate a request on the same corner agree?\n" + "=" * 104)
agree = Counter()
for tr in turn_recs:
    by_cell = defaultdict(dict)
    for d, rec in tr["per"].items():
        for m in rec["full"]:
            if m["intent"] == "request":
                by_cell[(m["cell"], m["layer"])].setdefault(d, m)
    for key, dm in by_cell.items():
        if len(dm) < 2:
            continue
        ds = sorted(dm)
        same = len({dm[d]["color"] for d in ds}) == 1
        agree["pairs"] += 1
        agree["same colour"] += same
        if not same and len(tr["scores"]) == 3:
            wrong = [d for d in ds if not dm[d]["target_match"]]
            right = [d for d in ds if dm[d]["target_match"]]
            if len(wrong) == 1 and len(right) == 1:
                agree["one right, one wrong"] += 1
                agree["wrong one ranked lower"] += tr["scores"][wrong[0]] < tr["scores"][right[0]]
print(f"   corner pairs mentioned by two directors in the same turn: {agree['pairs']}   same colour: {fmt(agree['same colour'] / max(agree['pairs'], 1))}")
if agree["one right, one wrong"]:
    print(f"   disagreements with exactly one correct director: {agree['one right, one wrong']}   policy ranked the wrong one lower: "
          f"{fmt(agree['wrong one ranked lower'] / agree['one right, one wrong'])} (chance 50%)")

print("\n" + "=" * 104 + "\n4. DOES MESSAGE QUALITY PREDICT THE OUTCOME? (Builder heard several directors; followed one by attribution)\n" + "=" * 104)
for c in COLS:
    rows = [tr for tr in turn_recs if tr["column"] == c and len(tr["heard"]) > 1 and tr["tag"] in tr["per"]]
    if len(rows) < 10:
        continue
    print(f"\n   [{c}]  {len(rows)} attributable turns")
    for lab, f in (("followed message had an actionable correct request", lambda tr: tr["per"][tr["tag"]]["actionable"]),
                   ("followed message had a request placeable now (board only)", lambda tr: tr["per"][tr["tag"]]["placeable"]),
                   ("followed message's requests mostly correct (>=50%)", lambda tr: (tr["per"][tr["tag"]]["req_precision"] >= 0.5) if tr["per"][tr["tag"]]["req_precision"] == tr["per"][tr["tag"]]["req_precision"] else None)):
        for val in (True, False):
            g_ = [tr for tr in rows if f(tr) is val]
            if g_:
                print(f"   {lab} = {str(val):<5}: n={len(g_):<4} executed {100 * np.mean([tr['executed'] for tr in g_]):3.0f}%"
                      f"   progress change {np.mean([tr['d_op'] for tr in g_]):+.4f}")
    # drop-one: when the Builder followed the bottom-ranked speaking director
    sub = [tr for tr in rows if len(tr["scores"]) == 3]
    if sub:
        fb = [tr for tr in sub if tr["tag"] == min(tr["speaking"], key=lambda d: tr["scores"][d])]
        fo = [tr for tr in sub if tr not in fb]
        for lab, grp in (("followed the bottom-ranked speaker", fb), ("followed another speaker", fo)):
            if grp:
                print(f"   {lab:<37} n={len(grp):<4} actionable {fmt(np.mean([tr['per'][tr['tag']]['actionable'] for tr in grp]))}"
                      f"   executed {100 * np.mean([tr['executed'] for tr in grp]):3.0f}%   progress change {np.mean([tr['d_op'] for tr in grp]):+.4f}")

if args.sample:
    print("\n" + "=" * 104 + f"\nSAMPLE OF {args.sample} MESSAGES FOR CHECKING THE EXTRACTOR BY HAND\n" + "=" * 104)
    rng = random.Random(args.seed)
    for r in rng.sample(R, min(args.sample, len(R))):
        print(f"\n[{r['structure']} {r['column']} turn {r['turn']} {r['director']}] {r['text'][:400]!r}")
        full_iter = iter(r["full"])
        for m in r["mentions"]:
            fm = next(full_iter) if m["pos"] is not None and m["layer"] is not None else None
            loc = f"{fm['cell']} L{m['layer']}" if fm else f"pos={m['pos']} layer={m['layer']}"
            truth = f"  target {'Y' if fm['target_match'] else 'n'} / board {'Y' if fm['board_match'] else 'n'} / next-free {'Y' if fm['next_free'] else 'n'}" if fm else ""
            print(f"     {m['intent']:<8} {CNAME[m['color']]:<7} {m['size'] or '-':<2} {loc:<16}{truth}")
