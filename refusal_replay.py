"""
Does the director prompt's response format trigger refusals?

Replays logged director prompts (director_prompt_D* in raw game logs) through the
Anthropic API under several versions of the RESPONSE FORMAT block, and records the
stop_reason of every reply:

  current   the prompt exactly as logged: a private <think> block, then <message>
  message   no thinking block: reply with the <message> block only
  notes     the same two-block format with the first block renamed <notes>
            ("working notes") instead of <think> ("private reasoning")
  native    (with --native) message-only prompt, with the API's own extended thinking
            enabled instead of a written thinking block

    python refusal_replay.py logs/craft_structure_*.json [--n 30] [--repeats 2]
        [--model claude-haiku-5-5] [--max-tokens 3000] [--variants current,message,notes]
        [--native] [--out refusal_replay.csv] [--dry-run]

Prompts are sampled across the given logs, half from director turns whose original
call was refused (if available) and half from ones that were not, so both kinds are
represented. --dry-run prints the format block of each variant for one prompt and
makes no API calls. Needs ANTHROPIC_API_KEY (or CLAUDE_API_KEY).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--model", default=None, help="default: the director model recorded in the logs")
ap.add_argument("--max-tokens", type=int, default=3000)
ap.add_argument("--variants", default="current,message,notes")
ap.add_argument("--native", action="store_true", help="add the 'native' variant (API extended thinking)")
ap.add_argument("--thinking-budget", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="refusal_replay.csv")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

FORMAT_RE = re.compile(r"(### RESPONSE FORMAT\s*)<think>.*?</think>\s*(<message>.*?</message>)", re.S)
MESSAGE_ONLY = ("### RESPONSE FORMAT\n\n    Reply with only this block:\n\n    <message>\n"
                "    [Natural human speech only — no coordinates, no codes, no layer numbers]\n    </message>\n")
NOTES_BLOCK = ("<notes>\n    [Working notes — use coordinates freely here to work out what's needed]\n    </notes>\n\n    ")


def variant(prompt, name):
    if name == "current":
        return prompt
    m = FORMAT_RE.search(prompt)
    if not m:
        raise ValueError("RESPONSE FORMAT block not found in the logged prompt")
    if name in ("message", "native"):
        out = prompt[:m.start()] + MESSAGE_ONLY + prompt[m.end():]
        return out.replace("(use only in your thinking)", "(for your own reference; never put coordinates in your message)")
    if name == "notes":
        return prompt[:m.start()] + m.group(1) + NOTES_BLOCK + m.group(2) + prompt[m.end():]
    raise ValueError(name)


def stop_details_of(resp):
    """-> (category, explanation) of a refusal's stop_details, or (None, None)."""
    sd = getattr(resp, "stop_details", None)
    if sd is None:
        sd = (getattr(resp, "model_extra", None) or {}).get("stop_details")
    if sd is None:
        return None, None
    if hasattr(sd, "model_dump"):
        sd = sd.model_dump()
    elif not isinstance(sd, dict):
        sd = dict(getattr(sd, "__dict__", {}))
    return sd.get("category"), sd.get("explanation")


def message_of(text):
    m = re.search(r"<message>\s*(.*?)\s*(?:</message>|$)", text, re.S | re.I)
    msg = (m.group(1) if m else re.sub(r"<(think|notes)>.*?</\1>", "", text, flags=re.S | re.I)).strip()
    return msg


# ------------------------------------------------------------------ collect logged prompts
items, model_seen = [], Counter()
for path in args.logs:
    d = json.load(open(path))
    cond = d.get("experiment_info", {}).get("condition")
    for g in d.get("games", []):
        for t in g.get("turns", []):
            refused = defaultdict(bool)
            for c in (t.get("token_usage") or {}).get("calls", []):
                if c.get("role") == "director":
                    refused[c.get("director")] |= c.get("finish_reason") == "refusal"
                    model_seen[c.get("model") or c.get("director_model")] += 1
            for D in ("D1", "D2", "D3"):
                p = t.get(f"director_prompt_{D}")
                if p and FORMAT_RE.search(p):
                    items.append({"log": os.path.basename(path), "condition": cond, "structure": g.get("structure_id"),
                                  "turn": t.get("turn_number"), "director": D, "orig_refused": refused[D], "prompt": p})
if not items:
    sys.exit("no logged director prompts with a RESPONSE FORMAT block were found")
model = args.model or (model_seen.most_common(1)[0][0] if model_seen else None)
if not model:
    sys.exit("no model recorded in the logs; pass --model")
rng = random.Random(args.seed)
ref = [x for x in items if x["orig_refused"]]; ok = [x for x in items if not x["orig_refused"]]
rng.shuffle(ref); rng.shuffle(ok)
half = args.n // 2
sample = ref[:half] + ok[:args.n - min(half, len(ref))]
sample = sample[:args.n]
names = [v.strip() for v in args.variants.split(",") if v.strip()] + (["native"] if args.native else [])
print(f"{len(items)} logged director prompts ({len(ref)} originally refused); sampled {len(sample)} "
      f"({sum(x['orig_refused'] for x in sample)} originally refused); model {model}; variants {names}; "
      f"{args.repeats} repeats -> {len(sample) * len(names) * args.repeats} calls")

if args.dry_run:
    p = sample[0]["prompt"]
    for name in names:
        v = variant(p, name)
        i = v.find("### RESPONSE FORMAT")
        print(f"\n===== {name} =====\n{v[i:i + 420]}")
    sys.exit(0)

# ------------------------------------------------------------------ replay
import anthropic
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"))
rows = []
for k, it in enumerate(sample, 1):
    system = f"You are Director {it['director']} in a collaborative LEGO construction task."
    for name in names:
        prompt = variant(it["prompt"], name)
        for r in range(args.repeats):
            kw = dict(model=model, max_tokens=args.max_tokens, system=system, messages=[{"role": "user", "content": prompt}])
            if name == "native":
                kw["thinking"] = {"type": "enabled", "budget_tokens": args.thinking_budget}
                kw["max_tokens"] = max(args.max_tokens, args.thinking_budget + 1024)
            try:
                resp = client.messages.create(**kw)
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                stop, out_tok = resp.stop_reason, resp.usage.output_tokens
                category, explanation = stop_details_of(resp) if stop == "refusal" else (None, None)
            except Exception as e:
                text, stop, out_tok = "", f"error: {type(e).__name__}", 0
                category, explanation = None, None
            msg = message_of(text)
            rows.append({**{x: it[x] for x in ("log", "condition", "structure", "turn", "director", "orig_refused")},
                         "variant": name, "repeat": r, "stop_reason": stop,
                         "refusal_category": category or "", "refusal_explanation": (explanation or "")[:200],
                         "output_tokens": out_tok,
                         "message_chars": len(msg), "usable_message": int(len(msg) > 3 and not msg.startswith("<")), "message": msg[:300]})
            time.sleep(0.2)
    done = [x for x in rows if x["log"] == it["log"] and x["turn"] == it["turn"] and x["director"] == it["director"]]
    print(f"  [{k}/{len(sample)}] {it['structure']} turn {it['turn']} {it['director']} (orig refused: {it['orig_refused']}): "
          + "  ".join(f"{n} {sum(x['stop_reason'] == 'refusal' for x in done if x['variant'] == n)}/{args.repeats}" for n in names))

with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ summary
print(f"\n{'variant':<10}{'calls':>7}{'refusals':>10}{'rate':>7}{'  on originally refused prompts':>32}{'  on the others':>17}"
      f"{'usable msg':>12}{'mean words':>12}{'errors':>8}")
for name in names:
    R = [x for x in rows if x["variant"] == name]
    rr = lambda sub: (sum(x["stop_reason"] == "refusal" for x in sub), len(sub))
    a, n = rr(R); a1, n1 = rr([x for x in R if x["orig_refused"]]); a0, n0 = rr([x for x in R if not x["orig_refused"]])
    words = [len(x["message"].split()) for x in R if x["usable_message"]]
    err = sum(str(x["stop_reason"]).startswith("error") for x in R)
    print(f"{name:<10}{n:>7}{a:>10}{100 * a / max(n, 1):>6.0f}%{f'{a1}/{n1}':>32}{f'{a0}/{n0}':>17}"
          f"{100 * sum(x['usable_message'] for x in R) / max(n, 1):>11.0f}%{(sum(words) / len(words) if words else 0):>12.1f}{err:>8}")
print("\nrefusal categories (stop_details.category) by variant:")
for name in names:
    cats = Counter(x["refusal_category"] or "(none reported)" for x in rows if x["variant"] == name and x["stop_reason"] == "refusal")
    print(f"   {name:<10}" + ("  ".join(f"{c}: {n}" for c, n in cats.most_common()) if cats else "no refusals"))
expl = next((x["refusal_explanation"] for x in rows if x["refusal_explanation"]), "")
if expl:
    print(f"   e.g. explanation: {expl!r}")
if any(x["stop_reason"] == "refusal" for x in rows) and not any(x["refusal_category"] for x in rows):
    print("   (no stop_details came back: upgrade the anthropic SDK, `pip install -U anthropic`, so the field is exposed)")
print(f"\nper-call results written to {args.out}. A variant whose refusal rate drops sharply while keeping a usable"
      f"\nmessage (and similar message length) is the one to adopt; check a few of its messages in the CSV by hand.")
