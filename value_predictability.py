"""
Can a director's simulated value-if-followed (simulated_value.py) be predicted from
PUBLIC inputs -- in particular, which director on a turn is worth following?

Labels: v_meant (default) or v_exec in {+1 correct if followed, 0 not placeable,
-1 wrong if followed}, for every recorded director-turn whose message had a located
request. Messages, boards and other directors' messages come from the converted
recordings (craft_recordings_games.json); the target is NOT used for features.

Feature sets (all public; located through the same frame as the label):
  wall    the director's wall on the board (blocks, empty cells, min/max height),
          total blocks on the board, turn number
  req     the primary request: remove or place, large block, its layer, the gap between
          that layer and the cell's current height (0 = next free layer), shared corner,
          number of located requests, "done"/"empty" phrases, message length
  ctx     cross-director consistency: another director's request on the same cell and
          layer with the same colour (corroborated) or a different one (contradicted)
  tfidf   word / bigram TF-IDF of the message
  embed   MiniLM sentence embedding of the message (needs sentence-transformers)
  combos  wall+req, wall+req+ctx, wall+req+ctx+tfidf, all
Model: ridge (penalty chosen by internal CV) on the label; scores used for ranking.
Cross-validation: folds grouped by STRUCTURE and, separately, by DIRECTOR MODEL.

Metrics on the cross-validated predictions:
  AUC(+1)     label +1 vs the rest, all rows
  pairwise    within a turn, pairs of labelled directors with different labels: how often
              the prediction orders them correctly (chance 50%); also split by pair type:
              +1 vs 0 (placeability: largely public), +1 vs -1 (colour correctness: needs
              the target, so only indirect cues help), 0 vs -1

    python value_predictability.py --recordings craft_recordings_games.json \
        --labels simulated_labels.csv [--label v_meant] [--no-embed] [--folds 5]
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
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--recordings", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--label", default="v_meant", choices=("v_meant", "v_exec"))
ap.add_argument("--frame-check", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_check.py"))
ap.add_argument("--reversed-dirs", default="D2,D3")
ap.add_argument("--no-embed", action="store_true")
ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--target", default="binary", choices=("binary", "calibrated", "ordinal"),
                help="binary: correct-if-followed (+1) vs the rest (default); calibrated: +1 -> 1, -1 -> 0.17, 0 -> 0, "
                     "the progress per turn each label gave in the Haiku random games; ordinal: the raw -1/0/+1")
args = ap.parse_args()
DIRS = ("D1", "D2", "D3")
DONE = re.compile(r"\b(complete|completed|done|finished|locked|all set|perfect|matches|spot[- ]on)\b", re.I)
EMPTY = re.compile(r"\b(empty|bare|nothing|missing|still need)\b", re.I)

_argv = sys.argv
sys.argv = [args.frame_check, "--games", args.recordings]
with contextlib.redirect_stdout(io.StringIO()):
    FC = runpy.run_path(args.frame_check)
sys.argv = _argv
extract, STATED = FC["extract"], FC["STATED"]
REV = {d.strip() for d in args.reversed_dirs.split(",") if d.strip()}
FRAME = {d: (STATED[d][::-1] if (args.label == "v_meant" and d in REV) else STATED[d]) for d in DIRS}
SHARED = {"(0,0)", "(0,2)"}

# ------------------------------------------------------------------ labels
LAB = {}
for r in csv.DictReader(open(args.labels)):
    if r[args.label] != "":
        LAB[(r["model_label"], r["structure_id"], int(r["turn"]), r["director"])] = int(r[args.label])

# ------------------------------------------------------------------ rows
REC = json.load(open(args.recordings))
rows = []
for g in REC["games"]:
    x = g["columns"][next(iter(g["columns"]))]
    board = x.get("start_board") or {}
    for t in x["turns"]:
        texts = {d: ((t.get("messages") or {}).get(d) or {}).get("text") or "" for d in DIRS}
        prim = {}
        for d in DIRS:
            reqs = [m for m in extract(texts[d], d) if m["intent"] in ("request", "remove") and m["pos"] is not None and m["layer"] is not None] if texts[d].strip() else []
            if reqs:
                m = reqs[0]
                prim[d] = (m, m.get("cell_abs") or FRAME[d][m["pos"]], len(reqs))
        total = sum(len(v) for v in board.values())
        for d in DIRS:
            key = (g.get("model_label", ""), g["structure_id"], int(t["turn"]), d)
            if key not in LAB or d not in prim:
                continue
            m, cell, nreq = prim[d]
            heights = [len(board.get(c, [])) for c in STATED[d]]
            others = [prim[e] for e in DIRS if e != d and e in prim and prim[e][0]["intent"] == m["intent"]]
            same_spot = [o for o in others if o[1] == cell and o[0]["layer"] == m["layer"]]
            rows.append({
                "game": key[:2], "turn_key": key[:3], "model": key[0], "structure": key[1], "y": LAB[key], "text": texts[d],
                "wall": [sum(heights), sum(h == 0 for h in heights), min(heights), max(heights), total, int(t["turn"])],
                "req": [int(m["intent"] == "remove"), int(m["size"] == "l"), m["layer"], m["layer"] - len(board.get(cell, [])),
                        int(m["layer"] == len(board.get(cell, []))), int(cell in SHARED), nreq,
                        int(bool(DONE.search(texts[d]))), int(bool(EMPTY.search(texts[d]))), len(texts[d].split())],
                "ctx": [int(any(o[0]["color"] == m["color"] for o in same_spot)),
                        int(any(o[0]["color"] != m["color"] for o in same_spot)), len(others)],
            })
        board = t.get("board_after") or board
n = len(rows)
y = np.array([r["y"] for r in rows], float)                       # the simulated label, -1 / 0 / +1
MAP = {"binary": {1: 1.0, 0: 0.0, -1: 0.0}, "calibrated": {1: 1.0, 0: 0.0, -1: 0.17}, "ordinal": {1: 1.0, 0: 0.0, -1: -1.0}}[args.target]
yt = np.array([MAP[int(v)] for v in y])                             # what is fitted and ranked
print(f"labelled director-turns with a located request: {n}   label {args.label}: "
      + "  ".join(f"{v:+d}: {100 * np.mean(y == v):.0f}%" for v in (1, 0, -1))
      + f"   turns with >= 2 labelled directors: {sum(1 for k, c in __import__('collections').Counter(r['turn_key'] for r in rows).items() if c >= 2)}")

# ------------------------------------------------------------------ features
dense = {k: np.array([r[k] for r in rows], float) for k in ("wall", "req", "ctx")}
texts = [r["text"] for r in rows]
E = None
if not args.no_embed:
    from sentence_transformers import SentenceTransformer
    E = np.asarray(SentenceTransformer(args.embed_model).encode(texts, batch_size=128, show_progress_bar=False))
SETS = {"wall": ["wall"], "req": ["req"], "ctx": ["ctx"], "tfidf": ["tfidf"], "wall+req": ["wall", "req"],
        "wall+req+ctx": ["wall", "req", "ctx"], "wall+req+ctx+tfidf": ["wall", "req", "ctx", "tfidf"]}
if E is not None:
    SETS["embed"] = ["embed"]
    SETS["all"] = ["wall", "req", "ctx", "tfidf", "embed"]


def design(parts, tr, te):
    blocks_tr, blocks_te = [], []
    for p in parts:
        if p == "tfidf":
            tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([texts[i] for i in tr])
            blocks_tr.append(tv.transform([texts[i] for i in tr])); blocks_te.append(tv.transform([texts[i] for i in te]))
        else:
            M = E if p == "embed" else dense[p]
            sc = StandardScaler().fit(M[tr])
            blocks_tr.append(csr_matrix(sc.transform(M[tr]))); blocks_te.append(csr_matrix(sc.transform(M[te])))
    return hstack(blocks_tr).tocsr(), hstack(blocks_te).tocsr()


def auc(score, pos):
    p, q = score[pos], score[~pos]
    if not len(p) or not len(q):
        return float("nan")
    order = np.argsort(np.concatenate([p, q]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))


def pairwise(pred):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["turn_key"]].append(i)
    res = defaultdict(lambda: [0.0, 0])
    for idx in by.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if yt[i] == yt[j]:
                    continue
                hi, lo = (i, j) if yt[i] > yt[j] else (j, i)
                s = pred[hi] - pred[lo]
                w = 1.0 if s > 0 else 0.5 if s == 0 else 0.0
                for k in ("all", f"{int(y[hi]):+d} vs {int(y[lo]):+d}"):
                    res[k][0] += w; res[k][1] += 1
    return {k: (100 * v[0] / v[1], v[1]) for k, v in res.items()}


print(f"target: {args.target} (pairs ranked: labels whose target values differ)")
print(f"\n{'features':<22}{'grouping':<11}{'AUC(+1)':>9}{'pairwise':>17}{'+1 vs 0':>16}{'+1 vs -1':>16}{'0 vs -1':>16}")
for name, parts in SETS.items():
    for gname in ("structure", "model"):
        groups = np.array([r[gname] for r in rows])
        k = min(args.folds, len(set(groups)))
        if k < 2:
            continue
        pred = np.zeros(n)
        for tr, te in GroupKFold(n_splits=k).split(np.zeros(n), y, groups):
            Xtr, Xte = design(parts, tr, te)
            pred[te] = RidgeCV(alphas=(0.1, 1, 10, 100, 1000, 10000)).fit(Xtr, yt[tr]).predict(Xte)
        pw = pairwise(pred)
        f = lambda key: f"{pw[key][0]:5.1f}% ({pw[key][1]:>4})" if key in pw else "      -"
        print(f"{name:<22}{gname:<11}{auc(pred, y == 1):>9.3f}{f('all'):>17}{f('+1 vs +0'):>16}{f('+1 vs -1'):>16}{f('+0 vs -1'):>16}")
print("\nRead: 'pairwise' is the decision-relevant number (which director on a turn is worth following); 60-65%+ makes"
      "\na value-if-followed head worth training. '+1 vs 0' is mostly placeability (public); '+1 vs -1' needs the"
      "\ncolour to be right, which public inputs can only hint at (corroboration, message style).")
