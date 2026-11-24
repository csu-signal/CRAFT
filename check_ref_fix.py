"""
Structural check of the --ref_fix change, from raw full-game logs (craft_structure_*.json).

For every game and director it reads the director prompt logged on the first turn and
checks:
  1. the order of the TARGET VIEW JSON the director received, against the game's target
     structure: "stated" (left to right as the prompt's perspective line says) or
     "reversed" (the original CRAFT swap)
  2. which bottom-layer worked example the prompt contains: "fixed" or "original"

Expected:   --ref_fix      D1, D2, D3 stated    example fixed
            without flag   D1 stated, D2 and D3 reversed    example original

    python check_ref_fix.py craft_results/<run>/**/craft_structure_*.json
"""
import json
import re
import sys

STATED = {"D1": ["(0,0)", "(1,0)", "(2,0)"], "D2": ["(0,0)", "(0,1)", "(0,2)"], "D3": ["(0,2)", "(1,2)", "(2,2)"]}
CNAME = {"r": "red", "b": "blue", "g": "green", "y": "yellow", "o": "orange"}


def norm(board):
    return {re.sub(r"\s", "", k): v for k, v in (board or {}).items()}


def target_rows(target, d):
    return [[CNAME.get(target[c][l][0], "none") if l < len(target.get(c) or []) else "none" for c in STATED[d]]
            for l in range(3)]


def view_in_prompt(prompt):
    i = prompt.rfind("### YOUR TARGET VIEW")              # the real view comes last; an earlier section uses the phrase too
    if i < 0:
        return None
    j = prompt.find("{", i)
    view, _ = json.JSONDecoder().raw_decode(prompt[j:])
    return [[c.get("color") for c in view[f"row_{l}"]] for l in range(3)]


def example_variant(prompt):
    i = prompt.find("GIVEN THE FOLLOWING VIEWS OF THE BOTTOM LAYER")
    if i < 0:
        return "absent"
    cols = re.findall(r'"color":\s*"(\w+)"', prompt[i:i + 1500])[:9]
    if cols[3:9] == ["yellow", "yellow", "blue", "blue", "green", "green"]:
        return "fixed"
    if cols[3:9] == ["blue", "yellow", "yellow", "green", "green", "blue"]:
        return "original"
    return "unrecognised"


ok_all = True
for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"{path}: unreadable ({type(e).__name__})")
        continue
    flag = d.get("experiment_info", {}).get("ref_fix")
    for g in d.get("games", []):
        target = norm(g.get("target_structure"))
        t0 = (g.get("turns") or [{}])[0]
        verdict, variants = {}, set()
        for dd in ("D1", "D2", "D3"):
            prompt = t0.get(f"director_prompt_{dd}")
            if not prompt:
                verdict[dd] = "no prompt logged"
                continue
            got, want = view_in_prompt(prompt), target_rows(target, dd)
            verdict[dd] = ("stated" if got == want else "reversed" if got == [r[::-1] for r in want]
                           else "neither (check the view)")
            variants.add(example_variant(prompt))
        mode = "ref_fix" if flag else ("original" if flag is False else
                                       ("ref_fix?" if verdict.get("D2") == "stated" else "original?"))
        want_v = {"D1": "stated", "D2": "stated", "D3": "stated"} if mode.startswith("ref_fix") else \
                 {"D1": "stated", "D2": "reversed", "D3": "reversed"}
        want_e = "fixed" if mode.startswith("ref_fix") else "original"
        ok = all(verdict[k] == v for k, v in want_v.items() if verdict[k] != "no prompt logged") and variants == {want_e}
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {path.split('/')[-1]:<38} mode={mode:<10} "
              + "  ".join(f"{k}:{v}" for k, v in verdict.items()) + f"   example={','.join(sorted(variants))}")
print("\nall checks passed" if ok_all else "\nsome checks FAILED -- see above"
      + "\n(mode with '?' was inferred from the views because the log has no experiment_info.ref_fix)")
