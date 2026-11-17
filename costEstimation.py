#!/usr/bin/env python3
"""
costEstimation.py -- cost of CRAFT games from the token counts run_craft.py saves.

Reads game_results['token_usage']['by_model'] from run_craft.py's game JSONs
(craft_<id>_<run>.json, a {"experiment_info", "games": [...]} file) and prices
every model's input and output tokens at the per-million-token prices you give.

Prices are per 1M tokens, input then output, from your provider's pricing page.
Give them on the command line:

    python costEstimation.py craft_results/qwen_noTools_pooled \\
        --price claude-haiku-4-5=1.00,5.00 --price gpt-4o-mini=0.15,0.60

or in a JSON file (optional "cache_read" / "cache_write" prices per 1M, for
Anthropic prompt caching; a 3rd and 4th number do the same on the command line):

    {"claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
     "gpt-4o-mini":      {"input": 0.15, "output": 0.60}}

    python costEstimation.py craft_results/ --prices prices.json --csv game_costs.csv

The numbers above are placeholders: use your provider's current prices.

Which tokens are billed
  * source "api": the provider's own counts.
  * source "estimate" (--token_estimate runs): the *_corrected counts
    (scaled by token_estimator.CORRECTION), as token_estimate.json reports them.
  * source "local_tokenizer" (local transformers directors): priced only if
    you give that model a price (local runs normally cost nothing).
  * Gemini thinking tokens are billed as output; OpenAI reasoning tokens are
    already inside its output count.
  * Anthropic prompt caching (run_craft.py --prompt_cache): input_tokens is the
    uncached input only; cache reads and writes are priced separately. Without
    explicit prices, reads cost the model's documented multiple of the input
    price (0.1x; 0.05x for Opus/Sonnet 5.5; 0.025x for Fable/Mythos 5.1-5.2) and
    writes 1.25x (5-minute cache) or 2x (1-hour cache, from experiment_info).
    no_cache_cost prices every input token at the plain input price, so
    cache_savings shows what caching saved (negative if writes outweighed reads).
  * Calls with no usage reported, or that errored, can't be priced; they are
    counted and reported, never silently treated as free.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def parse_price_args(items):
    prices = {}
    for item in items or []:
        try:
            model, vals = item.split("=", 1)
            parts = [float(v) for v in vals.split(",")]
        except ValueError:
            sys.exit(f"--price '{item}': expected MODEL=INPUT,OUTPUT[,CACHE_READ[,CACHE_WRITE]] (dollars per 1M tokens)")
        if len(parts) not in (2, 3, 4):
            sys.exit(f"--price '{item}': give 2-4 numbers: input, output[, cache_read[, cache_write]]")
        prices[model.strip()] = {"input": parts[0], "output": parts[1],
                                 **({"cache_read": parts[2]} if len(parts) >= 3 else {}),
                                 **({"cache_write": parts[3]} if len(parts) == 4 else {})}
    return prices


def load_prices(path, cli_items):
    prices = {}
    if path:
        with open(path) as f:
            for model, p in json.load(f).items():
                prices[model] = {k: float(v) for k, v in p.items()}
    prices.update(parse_price_args(cli_items))   # command line wins
    if not prices:
        sys.exit("No prices given. Use --price MODEL=INPUT,OUTPUT (per 1M tokens) or --prices prices.json.")
    return prices


def game_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "craft_*.json"), recursive=True)
        else:
            files.append(p)
    return sorted(set(f for f in files if f.endswith(".json")))


def cache_read_multiplier(model):
    """Cache-hit price as a multiple of the input price (prompt-caching docs)."""
    m = model.lower()
    if any(x in m for x in ("fable-5-2", "mythos-5-2", "fable-5-1", "mythos-5-1")):
        return 0.025
    if any(x in m for x in ("opus-5-5", "sonnet-5-5")):
        return 0.05
    return 0.1


def billable(bucket, model, source):
    """(input, output, cache_read, cache_write) tokens to bill for one by_model bucket.
    Estimate runs use the *_corrected counts throughout."""
    def get(field):
        if source == "estimate":
            return bucket.get(f"{field}_corrected", bucket.get(field, 0)) or 0
        return bucket.get(field, 0) or 0
    inp, out = get("input_tokens"), get("output_tokens")
    if "gemini" in model.lower():
        out += bucket.get("thinking_tokens", 0) or 0
    claude = model.lower().startswith("claude")
    cache_read = get("cache_read_input_tokens") if claude else 0
    cache_write = get("cache_creation_input_tokens") if claude else 0
    return inp, out, cache_read, cache_write


def price_game(game, prices, cache_ttl="5m"):
    """Cost breakdown for one game from its token_usage.by_model block."""
    usage = (game.get("token_usage") or {}).get("by_model")
    if usage is None:
        return None
    write_mult = 2.0 if cache_ttl == "1h" else 1.25
    by_role = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                                   "cache_write_tokens": 0, "cost": 0.0, "no_cache_cost": 0.0})
    missing, unpriced_calls = set(), 0
    for key, bucket in usage.items():
        role, model, source = (key.split(" | ") + ["?", "?", "?"])[:3]
        unpriced_calls += bucket.get("errors", 0) + bucket.get("calls_without_usage", 0)
        inp, out, cache_read, cache_write = billable(bucket, model, source)
        r = by_role[role]
        r["input_tokens"] += inp + cache_read + cache_write     # total input, cached or not
        r["output_tokens"] += out
        r["cache_read_tokens"] += cache_read
        r["cache_write_tokens"] += cache_write
        p = prices.get(model)
        if p is None:
            if source != "local_tokenizer" and (inp or out or cache_read or cache_write):
                missing.add(model)
            continue
        read_price = p.get("cache_read", p["input"] * cache_read_multiplier(model))
        write_price = p.get("cache_write", p["input"] * write_mult)
        r["cost"] += (inp * p["input"] + out * p["output"]
                      + cache_read * read_price + cache_write * write_price) / 1_000_000
        r["no_cache_cost"] += ((inp + cache_read + cache_write) * p["input"] + out * p["output"]) / 1_000_000
    total = sum(r["cost"] for r in by_role.values())
    no_cache = sum(r["no_cache_cost"] for r in by_role.values())
    return {"by_role": dict(by_role), "total": total, "no_cache_total": no_cache,
            "missing_prices": sorted(missing), "unpriced_calls": unpriced_calls}


def main():
    ap = argparse.ArgumentParser(description="Cost of CRAFT games from game_results['token_usage'].")
    ap.add_argument("paths", nargs="+", help="Game JSON files or folders (searched for craft_*.json).")
    ap.add_argument("--price", action="append", metavar="MODEL=IN,OUT[,CACHE]",
                    help="Dollars per 1M input and output tokens for MODEL (repeatable).")
    ap.add_argument("--prices", help="JSON file: {model: {input, output[, cache_read]}} per 1M tokens.")
    ap.add_argument("--csv", help="Also write one row per game to this CSV.")
    args = ap.parse_args()

    prices = load_prices(args.prices, args.price)
    files = game_files(args.paths)
    if not files:
        sys.exit("No game JSON files found.")

    rows, skipped, missing_all = [], 0, set()
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {f}: {e}")
            continue
        info = data.get("experiment_info", {})
        for game in data.get("games", []):
            cost = price_game(game, prices, cache_ttl=info.get("prompt_cache_ttl") or "5m")
            if cost is None:
                skipped += 1
                continue
            missing_all |= set(cost["missing_prices"])
            role = cost["by_role"]
            rows.append({
                "file": os.path.relpath(f),
                "structure": game.get("structure_id"),
                "condition": game.get("condition") or info.get("condition"),
                "director_model": info.get("models", {}).get("director"),
                "builder_model": info.get("models", {}).get("builder"),
                "token_estimate": bool(info.get("token_estimate")) or any(
                    k.endswith("| estimate") for k in game["token_usage"]["by_model"]),
                "turns": game.get("turns_taken"),
                "completed": game.get("completed"),
                "director_input_tokens": role.get("director", {}).get("input_tokens", 0),
                "director_output_tokens": role.get("director", {}).get("output_tokens", 0),
                "builder_input_tokens": role.get("builder", {}).get("input_tokens", 0),
                "builder_output_tokens": role.get("builder", {}).get("output_tokens", 0),
                "director_cost": round(role.get("director", {}).get("cost", 0.0), 6),
                "builder_cost": round(role.get("builder", {}).get("cost", 0.0), 6),
                "other_cost": round(sum(v["cost"] for k, v in role.items()
                                        if k not in ("director", "builder")), 6),
                "total_cost": round(cost["total"], 6),
                "prompt_cache": bool(info.get("prompt_cache")),
                "cache_read_tokens": sum(v["cache_read_tokens"] for v in role.values()),
                "cache_write_tokens": sum(v["cache_write_tokens"] for v in role.values()),
                "no_cache_cost": round(cost["no_cache_total"], 6),
                "cache_savings": round(cost["no_cache_total"] - cost["total"], 6),
                "unpriced_calls": cost["unpriced_calls"],
                "missing_prices": ",".join(cost["missing_prices"]),
            })

    if not rows:
        sys.exit(f"No games with token_usage found ({skipped} games without it: run with the "
                 f"updated run_craft.py to record tokens).")

    # Per-game table
    show_cache = any(r["cache_read_tokens"] or r["cache_write_tokens"] or r["prompt_cache"] for r in rows)
    print(f"{'structure':<16}{'condition':<10}{'director':<24}{'builder':<24}{'turns':>6}"
          f"{'dir $':>11}{'builder $':>11}{'total $':>11}"
          + (f"{'no-cache $':>12}{'saved $':>10}" if show_cache else ""))
    for r in rows:
        print(f"{str(r['structure']):<16}{str(r['condition']):<10}{str(r['director_model'])[:23]:<24}"
              f"{str(r['builder_model'])[:23]:<24}"
              f"{str(r['turns']):>6}{r['director_cost']:>11.4f}{r['builder_cost']:>11.4f}{r['total_cost']:>11.4f}"
              + (f"{r['no_cache_cost']:>12.4f}{r['cache_savings']:>10.4f}" if show_cache else "")
              + ("  (estimate)" if r["token_estimate"] else ""))

    # Totals by director model x builder model x condition
    groups = defaultdict(list)
    for r in rows:
        groups[(r["director_model"], r["builder_model"], r["condition"], r["prompt_cache"])].append(r)
    print("\nBy director model, builder model, condition and prompt caching:")
    print(f"  {'director':<24}{'builder':<24}{'condition':<10}{'cache':<7}{'games':>6}{'mean $/game':>13}{'total $':>11}")
    for (dm, bm, cond, pc), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tot = sum(r["total_cost"] for r in rs)
        print(f"  {str(dm)[:23]:<24}{str(bm)[:23]:<24}{str(cond):<10}{'on' if pc else 'off':<7}{len(rs):>6}"
              f"{tot / len(rs):>13.4f}{tot:>11.4f}")
    grand = sum(r["total_cost"] for r in rows)
    print(f"\nAll {len(rows)} games: ${grand:.4f} (mean ${grand / len(rows):.4f} per game)")
    if show_cache:
        nc = sum(r["no_cache_cost"] for r in rows)
        rd = sum(r["cache_read_tokens"] for r in rows); wr = sum(r["cache_write_tokens"] for r in rows)
        print(f"Prompt caching: ${nc:.4f} without caching -> ${grand:.4f} with it "
              f"(saved ${nc - grand:.4f}, {100 * (nc - grand) / nc if nc else 0:.1f}%); "
              f"cache read {rd} tokens, cache write {wr} tokens.")
        if any(r["prompt_cache"] and not (r["cache_read_tokens"] or r["cache_write_tokens"]) for r in rows):
            print("Note: some games ran with --prompt_cache but cached nothing; the prompt prefix is "
                  "probably below the model's minimum cacheable length (4096 tokens for Claude Haiku 4.5).")

    if skipped:
        print(f"Note: {skipped} game(s) had no token_usage (recorded before token logging) and were skipped.")
    if missing_all:
        print(f"WARNING: no price for {sorted(missing_all)}; their tokens are counted but cost $0 above. "
              f"Add --price MODEL=IN,OUT.")
    unpriced = sum(r["unpriced_calls"] for r in rows)
    if unpriced:
        print(f"Note: {unpriced} call(s) reported no usage or errored and could not be priced.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Per-game costs written to {args.csv}")


if __name__ == "__main__":
    main()