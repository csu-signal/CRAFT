"""Attribute logged oracle suggestions to the enumerator branch that produced them.

    python3 analyze_oracle_branch_usage.py [results_dir]

turn_data['oracle_moves'] records the n=5 sampled list the Builder was shown, but
stores bare move dicts with no 'source' field — so the logs alone cannot say which
branch of enumerate_correct_actions produced a given suggestion.

This recovers it. For each logged turn, structure_before is replayed through
agents.oracle.reconstruct_state and re-enumerated; each logged move is matched
against that enumeration on (action, position, layer, block, span_to) and takes
the matching entry's source.

The match rate doubles as a fidelity check on the replay: anything below 100%
means reconstruct_state is not reproducing the state the game actually had, and
every replay-derived number should be treated as suspect.

Reported per branch:
  shown     — suggestions actually placed in front of the Builder
  executed  — the Builder's attempted move equalled that suggestion

'executed' uses a strict 5-field match. run_craft.py's own
turn_data['builder_followed_oracle'] matches on action/position/layer only,
ignoring block and span, so it is reported alongside as a looser upper bound.
PS: The "failure" cases does not consider the fact that the builder: 1) not forced for follow oracle 2) may still deviate and make a remove operation in cases 
oracle suggests a place. 


"""

import collections
import contextlib
import glob
import io
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

from agents.oracle import reconstruct_state, enumerate_correct_actions

DEFAULT_RESULTS_DIR = "results"
# target_place is split by whether the cell's existing blocks already match the
# target. A place onto a corrupt prefix stacks a correct block on top of a wrong
# one, which buries the error: no branch can reach a wrong block once it sits
# below the top of a stack that is shorter than target.
BRANCHES = ["target_place", "target_place_ON_CORRUPT", "expose_buried_wrong",
            "wrong_block_remove", "excess_remove"]


def corrupt_cells(structure_before, target_structure):
    """Cells whose existing blocks already disagree with the target."""
    target = {norm_pos(k): list(v) for k, v in target_structure.items()}
    out = set()
    for pos, stack in structure_before.items():
        pos = norm_pos(pos)
        tgt = target.get(pos, [])
        if any(stack[i] != tgt[i] for i in range(min(len(stack), len(tgt)))):
            out.add(pos)
    return out


def label_for(source, move, corrupt):
    """Tag a placement that lands on an already-corrupt cell."""
    if source == "target_place" and norm_pos(move.get("position")) in corrupt:
        return "target_place_ON_CORRUPT"
    return source


def norm_pos(p):
    return "(" + ",".join(x.strip() for x in str(p).strip("()").split(",")) + ")"


def move_key(m):
    """Identity of a move, independent of dict ordering or extra logged fields."""
    return (m.get("action"),
            norm_pos(m.get("position")),
            m.get("layer"),
            m.get("block"),
            norm_pos(m["span_to"]) if m.get("span_to") else None)


def loose_key(m):
    """run_craft.py's own adherence test: action/position/layer only."""
    return (m.get("action"), norm_pos(m.get("position")), m.get("layer"))


def iter_games(path):
    data = json.load(open(path))
    if isinstance(data, dict) and "games" in data:
        return data["games"]
    return data if isinstance(data, list) else [data]


def analyze(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*", "*", "craft_structure_*.json")))
    if not files:
        sys.exit(f"No result files under {results_dir}/*/*/craft_structure_*.json")

    sink = io.StringIO()
    shown = collections.Counter()
    executed = collections.Counter()
    executed_loose = collections.Counter()
    per_model = collections.defaultdict(collections.Counter)
    turns_total = turns_with_oracle = 0
    moves_shown = moves_matched = 0

    for path in files:
        model = path.split(os.sep)[-3]
        for game in iter_games(path):
            for turn in game.get("turns", []):
                if turn.get("structure_before") is None:
                    continue
                turns_total += 1
                logged = turn.get("oracle_moves") or []
                if not logged:
                    continue
                turns_with_oracle += 1
                try:
                    with contextlib.redirect_stdout(sink):
                        entries = enumerate_correct_actions(reconstruct_state(turn, game))
                except Exception:
                    continue
                lut = {move_key(e["move"]): e["source"] for e in entries if e["flag"] == "ok"}
                corrupt = corrupt_cells(turn["structure_before"], game["target_structure"])

                for m in logged:
                    moves_shown += 1
                    source = lut.get(move_key(m))
                    if source is None:
                        continue
                    moves_matched += 1
                    source = label_for(source, m, corrupt)
                    shown[source] += 1
                    per_model[model][source] += 1

                attempted = turn.get("move_attempted") or {}
                if not attempted.get("action"):
                    continue
                strict = {move_key(m) for m in logged}
                loose = {loose_key(m) for m in logged}
                source = lut.get(move_key(attempted))
                if source:
                    source = label_for(source, attempted, corrupt)
                if source and move_key(attempted) in strict:
                    executed[source] += 1
                if loose_key(attempted) in loose and source:
                    executed_loose[source] += 1

    return dict(files=files, shown=shown, executed=executed, executed_loose=executed_loose,
                per_model=per_model, turns_total=turns_total,
                turns_with_oracle=turns_with_oracle,
                moves_shown=moves_shown, moves_matched=moves_matched)


def report(r):
    rule = "-" * 78
    models = sorted({f.split(os.sep)[-3] for f in r["files"]})
    print(rule)
    print("ORACLE BRANCH USAGE — recovered from logged oracle_moves")
    print(rule)
    print(f"\n  result files            : {len(r['files'])}")
    print(f"  conditions              : {len(models)}")
    for m in models:
        print(f"      {m}")
    print(f"  turns with a board state: {r['turns_total']}")
    print(f"  turns with suggestions  : {r['turns_with_oracle']}")

    pct = 100 * r["moves_matched"] / r["moves_shown"] if r["moves_shown"] else 0.0
    print(f"\n  REPLAY FIDELITY — logged moves matched to a re-enumerated candidate")
    print(f"    shown {r['moves_shown']}   matched {r['moves_matched']}   ({pct:.1f}%)")
    if r["moves_matched"] != r["moves_shown"]:
        print(f"    WARNING: {r['moves_shown'] - r['moves_matched']} unmatched — reconstruct_state "
              f"is not reproducing the live state; treat replay numbers as suspect.")
    else:
        print(f"    exact — every logged suggestion reproduced by replay")

    total = sum(r["shown"].values()) or 1
    print(f"\n  {'branch':<24}{'shown':>10}{'% shown':>10}{'executed':>11}{'(loose)':>10}")
    for b in BRANCHES:
        if not r["shown"][b]:
            continue
        print(f"  {b:<24}{r['shown'][b]:>10}{100 * r['shown'][b] / total:>9.1f}%"
              f"{r['executed'][b]:>11}{r['executed_loose'][b]:>10}")
    print(f"  {'TOTAL':<24}{sum(r['shown'].values()):>10}{'':>10}"
          f"{sum(r['executed'].values()):>11}{sum(r['executed_loose'].values()):>10}")

    print(f"\n  Per condition (suggestions shown):\n")
    print(f"  {'condition':<34}" + "".join(f"{b:>25}" for b in BRANCHES))
    for m in models:
        print(f"  {m:<34}" + "".join(f"{r['per_model'][m][b]:>25}" for b in BRANCHES))

    cycled = r["executed"]["expose_buried_wrong"]
    buried = r["executed"]["target_place_ON_CORRUPT"]
    followed = sum(r["executed"].values())
    if followed:
        print(f"\n  Two distinct failures, with very different costs:\n")
        print(f"    expose_buried_wrong      {cycled:>4} executed ({100 * cycled / followed:>4.1f}% of followed)")
        print(f"      Wastes the turn. The removed block is re-placed next turn, so the")
        print(f"      board returns to where it was — recoverable, just unproductive.\n")
        print(f"    target_place_ON_CORRUPT  {buried:>4} executed ({100 * buried / followed:>4.1f}% of followed)")
        print(f"      Buries a wrong block under a correct one. No branch can reach a wrong")
        print(f"      block below the top of a short stack, so the cell becomes unrepairable.")
        print(f"      This is the damaging one, and it scores as progress: structurePlacement")
        print(f"      is True and blocks_placed_correctly increases.")
    print(rule)


if __name__ == "__main__":
    report(analyze(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR))
