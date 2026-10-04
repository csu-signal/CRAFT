"""Classify the Builder moves that did not match any oracle suggestion.

    python3 analyze_builder_deviations.py [results_dir]

Complements analyze_oracle_branch_usage.py, which asks which branch produced the
suggestions. This asks the opposite question: when the Builder ignored them, what
exactly did it do instead?

A deviating move is walked down the move tuple from coarsest field to finest, so
each turn lands in exactly one bucket:

  A  different cell entirely    no suggestion targets that position
  B  wrong action               right cell, place vs remove swapped
  C  wrong layer                right cell and action, different height
  D  wrong colour               right slot, same size, different colour
  E  wrong size                 right slot, same colour, small vs large
  F  wrong colour and size      right slot, both fields differ
  G  wrong span                 right block, different span partner

Category B is examined further. "Oracle said place, Builder said remove" on an
already-corrupt cell is the Builder proposing the repair the enumerator cannot
generate — scored as an error by the taxonomy, but the correct move.

Counts are over turns that have BOTH a logged oracle_moves list and an attempted
move, which is the only population where adherence is defined.
"""

import collections
import glob
import json
import os
import sys
import types

# ── Dependency shim ───────────────────────────────────────────────────────────
# agents/environment.py binds the OpenAI client at import time; nothing here makes
# a network call. Register a stub when the SDK is absent so the import resolves.
try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("openai")

    class _OfflineClient:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            raise RuntimeError(f"offline analysis; no OpenAI calls expected ({name})")

    _stub.OpenAI = _OfflineClient
    _stub.AzureOpenAI = _OfflineClient
    sys.modules["openai"] = _stub

DEFAULT_RESULTS_DIR = "results"
COLORS = {"y": "yellow", "o": "orange", "g": "green", "b": "blue", "r": "red"}
CATEGORIES = [
    "A. different cell entirely",
    "B. right cell, wrong ACTION",
    "C. right cell+action, wrong LAYER",
    "D. right slot, wrong COLOUR",
    "E. right slot, wrong SIZE",
    "F. right slot, wrong colour AND size",
    "G. right block, wrong SPAN",
]


def norm_pos(p):
    return "(" + ",".join(x.strip() for x in str(p).strip("()").split(",")) + ")"


def span_of(m):
    return norm_pos(m["span_to"]) if m.get("span_to") else None


def full_key(m):
    return (m.get("action"), norm_pos(m.get("position")), m.get("layer"),
            m.get("block"), span_of(m))


def describe(code):
    if not code:
        return "?"
    return f"{COLORS.get(code[0], '?')} {'large' if code.endswith('l') else 'small'}"


def classify(attempted, suggestions):
    """Return (category, detail) for a move that matched no suggestion."""
    pos = norm_pos(attempted.get("position"))
    same_pos = [m for m in suggestions if norm_pos(m.get("position")) == pos]
    if not same_pos:
        return CATEGORIES[0], None

    same_action = [m for m in same_pos if m.get("action") == attempted.get("action")]
    if not same_action:
        return CATEGORIES[1], f"oracle={same_pos[0]['action']} -> builder={attempted['action']}"

    same_layer = [m for m in same_action if m.get("layer") == attempted.get("layer")]
    if not same_layer:
        return CATEGORIES[2], attempted.get("layer", 0) - same_action[0].get("layer", 0)

    expected, got = same_layer[0].get("block"), attempted.get("block")
    if expected != got:
        if expected and got and expected[0] != got[0] and expected[1:] == got[1:]:
            return CATEGORIES[3], (expected, got)
        if expected and got and expected[0] == got[0]:
            return CATEGORIES[4], (expected, got)
        return CATEGORIES[5], (expected, got)
    return CATEGORIES[6], None


def iter_games(path):
    data = json.load(open(path))
    if isinstance(data, dict) and "games" in data:
        return data["games"]
    return data if isinstance(data, list) else [data]


def cell_is_corrupt(structure_before, target_structure, pos):
    target = {norm_pos(k): list(v) for k, v in target_structure.items()}
    current = {norm_pos(k): list(v) for k, v in structure_before.items()}
    cur, tgt = current.get(pos, []), target.get(pos, [])
    return any(cur[i] != tgt[i] for i in range(min(len(cur), len(tgt))))


def analyze(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*", "*", "craft_structure_*.json")))
    if not files:
        sys.exit(f"No result files under {results_dir}/*/*/craft_structure_*.json")

    cat = collections.Counter()
    per_model = collections.defaultdict(collections.Counter)
    rejected = collections.Counter()
    confusion = collections.Counter()
    swaps = collections.Counter()
    layer_offsets = collections.Counter()
    repair_right = repair_wrong = 0
    repair_examples = []
    turns = matched = 0

    for path in files:
        model = path.split(os.sep)[-3]
        for game in iter_games(path):
            for turn in game.get("turns", []):
                suggestions = turn.get("oracle_moves") or []
                attempted = turn.get("move_attempted") or {}
                if not suggestions or not attempted.get("action"):
                    continue
                turns += 1
                if any(full_key(m) == full_key(attempted) for m in suggestions):
                    matched += 1
                    continue

                category, detail = classify(attempted, suggestions)
                cat[category] += 1
                per_model[model][category] += 1
                if not turn.get("move_executed"):
                    rejected[category] += 1

                if category == CATEGORIES[1]:
                    swaps[detail] += 1
                elif category == CATEGORIES[2]:
                    layer_offsets[detail] += 1
                elif detail and category in CATEGORIES[3:6]:
                    confusion[detail] += 1

                # Was a place->remove swap actually the correct repair?
                if (attempted.get("action") == "remove"
                        and turn.get("structure_before") is not None
                        and category == CATEGORIES[1]):
                    pos = norm_pos(attempted.get("position"))
                    if cell_is_corrupt(turn["structure_before"], game["target_structure"], pos):
                        repair_right += 1
                        if len(repair_examples) < 5:
                            cur = {norm_pos(k): list(v)
                                   for k, v in turn["structure_before"].items()}.get(pos, [])
                            tgt = {norm_pos(k): list(v)
                                   for k, v in game["target_structure"].items()}.get(pos, [])
                            repair_examples.append((pos, cur, tgt, attempted.get("layer")))
                    else:
                        repair_wrong += 1

    return dict(files=files, turns=turns, matched=matched, cat=cat, per_model=per_model,
                rejected=rejected, confusion=confusion, swaps=swaps,
                layer_offsets=layer_offsets, repair_right=repair_right,
                repair_wrong=repair_wrong, repair_examples=repair_examples)


def report(r):
    rule = "-" * 78
    models = sorted({f.split(os.sep)[-3] for f in r["files"]})
    turns = r["turns"] or 1
    deviations = turns - r["matched"]

    print(rule)
    print("BUILDER DEVIATIONS — what the Builder did when it ignored the oracle")
    print(rule)
    print(f"\n  result files : {len(r['files'])}   conditions : {len(models)}")
    print(f"  turns with both a suggestion list and an attempted move : {turns}")
    print(f"    matched a suggestion : {r['matched']:>5} ({100 * r['matched'] / turns:.1f}%)")
    print(f"    deviated             : {deviations:>5} ({100 * deviations / turns:.1f}%)")

    print(f"\n  {'deviation':<40}{'n':>6}{'% turns':>10}{'% of dev':>10}{'engine rejected':>18}")
    for c in CATEGORIES:
        if not r["cat"][c]:
            continue
        n = r["cat"][c]
        print(f"  {c:<40}{n:>6}{100 * n / turns:>9.1f}%{100 * n / max(deviations, 1):>9.1f}%"
              f"{r['rejected'][c]:>18}")

    if r["swaps"]:
        print(f"\n  Action swaps: " + ", ".join(f"{k} ({v})" for k, v in r["swaps"].most_common()))
    if r["layer_offsets"]:
        print(f"  Layer offsets (builder - oracle): {dict(sorted(r['layer_offsets'].items()))}")

    if r["confusion"]:
        print(f"\n  Block confusions — oracle wanted -> Builder placed:\n")
        for (exp, got), n in r["confusion"].most_common(10):
            print(f"    {str(exp):>4} -> {str(got):<4}  ({describe(exp)} -> {describe(got)}){n:>6}")

    total_swap = r["repair_right"] + r["repair_wrong"]
    if total_swap:
        print(f"\n  Of the {total_swap} 'oracle said PLACE, Builder said REMOVE' turns:")
        print(f"    cell was ALREADY CORRUPT — Builder proposed the repair the enumerator")
        print(f"      cannot generate, and the taxonomy scores it as an error : {r['repair_right']}")
        print(f"    cell was clean — Builder simply wrong                      : {r['repair_wrong']}")
        if r["repair_examples"]:
            print(f"\n    examples where the Builder was right and the oracle was not:")
            for pos, cur, tgt, layer in r["repair_examples"]:
                print(f"      {pos} current={cur} target={tgt} -> removed layer {layer}")

    print(f"\n  Per condition:\n")
    print(f"  {'condition':<34}" + "".join(f"{c.split('.')[0]:>7}" for c in CATEGORIES))
    for m in models:
        print(f"  {m:<34}" + "".join(f"{r['per_model'][m][c]:>7}" for c in CATEGORIES))
    print(f"\n  A=cell  B=action  C=layer  D=colour  E=size  F=colour+size  G=span")
    print(rule)


if __name__ == "__main__":
    report(analyze(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR))
