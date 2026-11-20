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

Call logs (token_calls_<run_id>.jsonl)
  run_craft.py also appends every model call to a per-run JSONL file the moment
  it completes. These are priced too, grouped by run, in their own section: they
  include runs that were interrupted or whose game files were never written, so
  their total is the one to compare with the Claude Console / provider bill.
  (Game files only exist for games that finished.)

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
    the "Prompt caching savings by model" table shows what caching saved for each
    model (negative if writes outweighed reads).
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


def price_bucket(bucket, model, source, prices, write_mult):
    """(cost, no_cache_cost, (input, output, cache_read, cache_write), priced?) for one
    by_model bucket or one logged call."""
    inp, out, cache_read, cache_write = billable(bucket, model, source)
    p = prices.get(model)
    if p is None:
        return 0.0, 0.0, (inp, out, cache_read, cache_write), False
    read_price = p.get("cache_read", p["input"] * cache_read_multiplier(model))
    write_price = p.get("cache_write", p["input"] * write_mult)
    cost = (inp * p["input"] + out * p["output"] + cache_read * read_price + cache_write * write_price) / 1_000_000
    no_cache = ((inp + cache_read + cache_write) * p["input"] + out * p["output"]) / 1_000_000
    return cost, no_cache, (inp, out, cache_read, cache_write), True


def new_model_tally():
    return {"roles": set(), "calls": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0, "no_cache_cost": 0.0}


def add_to_model_tally(tally, model, role, cost, no_cache, cache_read, cache_write, calls=0):
    t = tally.setdefault(model, new_model_tally())
    t["roles"].add(str(role))
    t["calls"] += calls
    t["cache_read"] += cache_read
    t["cache_write"] += cache_write
    t["cost"] += cost
    t["no_cache_cost"] += no_cache


def print_cache_by_model(tally, title):
    """Prompt-caching savings per model: what each model cost with caching vs
    pricing all of its input at the plain input price."""
    rows = [(m, t) for m, t in tally.items() if t["cost"] or t["no_cache_cost"]]
    if not rows:
        return
    print(f"\n{title}")
    print(f"  {'model':<24}{'roles':<18}{'cache read':>12}{'cache write':>13}"
          f"{'with cache $':>14}{'no cache $':>12}{'saved $':>10}{'saved %':>9}")
    for m, t in sorted(rows, key=lambda kv: -(kv[1]["no_cache_cost"] - kv[1]["cost"])):
        saved = t["no_cache_cost"] - t["cost"]
        pct = 100 * saved / t["no_cache_cost"] if t["no_cache_cost"] else 0.0
        note = "" if (t["cache_read"] or t["cache_write"]) else "  (nothing cached)"
        print(f"  {m[:23]:<24}{','.join(sorted(t['roles']))[:17]:<18}{t['cache_read']:>12}{t['cache_write']:>13}"
              f"{t['cost']:>14.4f}{t['no_cache_cost']:>12.4f}{saved:>10.4f}{pct:>8.1f}%{note}")
    if any(not (t["cache_read"] or t["cache_write"]) and m.lower().startswith("claude") for m, t in rows):
        print("  (nothing cached) on a Claude model: caching was off for those runs, or the prompt prefix was "
              "below the model's minimum cacheable length (4096 tokens for Claude Haiku 4.5).")


def call_log_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "token_calls_*.jsonl"), recursive=True)
        elif p.endswith(".jsonl"):
            files.append(p)
    return sorted(set(files))


def price_call_logs(files, prices, model_tally=None):
    """Per-run totals over every logged call (and per-model totals into model_tally)."""
    runs = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue      # a line cut off by a hard kill
                rid = rec.get("run_id") or os.path.basename(f)
                r = runs.setdefault(rid, {"file": os.path.relpath(f), "first": rec.get("ts"), "last": rec.get("ts"),
                                          "calls": 0, "errors": 0, "unpriced_calls": 0, "missing": set(),
                                          "models": set(), "roles": defaultdict(lambda: [0, 0, 0.0]),
                                          "cost": 0.0, "no_cache_cost": 0.0,
                                          "summary_written": False})
                r["last"] = rec.get("ts") or r["last"]
                r["calls"] += 1
                if rec.get("error") or rec.get("input_tokens") is None:
                    r["errors"] += int(bool(rec.get("error")))
                    r["unpriced_calls"] += 1
                    continue
                model, source = str(rec.get("model")), rec.get("source", "api")
                write_mult = 2.0 if rec.get("prompt_cache") == "1h" else 1.25
                cost, nc, (inp, out, rd, wr), priced = price_bucket(rec, model, source, prices, write_mult)
                if not priced and source != "local_tokenizer":
                    r["missing"].add(model)
                r["models"].add(f"{rec.get('role')}={model}")
                role = r["roles"][rec.get("role", "?")]
                role[0] += inp + rd + wr
                role[1] += out
                role[2] += cost
                r["cost"] += cost
                r["no_cache_cost"] += nc
                if model_tally is not None and priced:
                    add_to_model_tally(model_tally, model, rec.get("role", "?"), cost, nc, rd, wr, calls=1)
    for rid, r in runs.items():
        folder = os.path.dirname(r["file"])
        r["summary_written"] = os.path.exists(os.path.join(folder, f"token_usage_{rid}.json"))
    return runs


def print_call_log_section(runs):
    print("\nEvery logged call (token_calls_*.jsonl), by run -- compare this total with your provider's console:")
    print(f"  {'run id':<22}{'started':<21}{'calls':>6}{'errors':>7}{'dir in/out':>20}{'builder in/out':>20}{'cost $':>10}")
    for rid, r in sorted(runs.items(), key=lambda kv: str(kv[1]["first"])):
        d, b = r["roles"].get("director", [0, 0, 0]), r["roles"].get("builder", [0, 0, 0])
        print(f"  {rid:<22}{str(r['first'])[:19]:<21}{r['calls']:>6}{r['errors']:>7}"
              f"{f'{d[0]}/{d[1]}':>20}{f'{b[0]}/{b[1]}':>20}{r['cost']:>10.4f}"
              + ("" if r["summary_written"] else "  (no end-of-run summary: interrupted?)"))
    total = sum(r["cost"] for r in runs.values())
    calls = sum(r["calls"] for r in runs.values())
    print(f"  All logged calls: {calls} calls in {len(runs)} run(s), ${total:.4f}")
    missing = set().union(*(r["missing"] for r in runs.values())) if runs else set()
    if missing:
        print(f"  WARNING: no price for {sorted(missing)}; their calls cost $0 above.")
    unpriced = sum(r["unpriced_calls"] for r in runs.values())
    if unpriced:
        print(f"  Note: {unpriced} logged call(s) errored or reported no usage (failed calls aren't billed).")
    return total


def price_game(game, prices, cache_ttl="5m"):
    """Cost breakdown for one game from its token_usage.by_model block."""
    usage = (game.get("token_usage") or {}).get("by_model")
    if usage is None:
        return None
    write_mult = 2.0 if cache_ttl == "1h" else 1.25
    by_role = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                                   "cache_write_tokens": 0, "cost": 0.0, "no_cache_cost": 0.0})
    missing, unpriced_calls = set(), 0
    by_model = {}
    for key, bucket in usage.items():
        role, model, source = (key.split(" | ") + ["?", "?", "?"])[:3]
        unpriced_calls += bucket.get("errors", 0) + bucket.get("calls_without_usage", 0)
        cost, nc, (inp, out, cache_read, cache_write), priced = price_bucket(bucket, model, source, prices, write_mult)
        r = by_role[role]
        r["input_tokens"] += inp + cache_read + cache_write     # total input, cached or not
        r["output_tokens"] += out
        r["cache_read_tokens"] += cache_read
        r["cache_write_tokens"] += cache_write
        if not priced:
            if source != "local_tokenizer" and (inp or out or cache_read or cache_write):
                missing.add(model)
            continue
        r["cost"] += cost
        r["no_cache_cost"] += nc
        bm = by_model.setdefault(model, {"roles": set(), "calls": 0, "cost": 0.0, "no_cache_cost": 0.0,
                                         "cache_read": 0, "cache_write": 0})
        bm["roles"].add(role)
        bm["calls"] += bucket.get("calls", 0)
        bm["cost"] += cost
        bm["no_cache_cost"] += nc
        bm["cache_read"] += cache_read
        bm["cache_write"] += cache_write
    total = sum(r["cost"] for r in by_role.values())
    no_cache = sum(r["no_cache_cost"] for r in by_role.values())
    return {"by_role": dict(by_role), "by_model": by_model, "total": total, "no_cache_total": no_cache,
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
    log_files = call_log_files(args.paths)
    if not files and not log_files:
        sys.exit("No game JSON files or token_calls_*.jsonl call logs found.")

    rows, skipped, missing_all = [], 0, set()
    game_model_tally = {}
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
            for m, bm in cost["by_model"].items():
                add_to_model_tally(game_model_tally, m, ",".join(sorted(bm["roles"])), bm["cost"],
                                   bm["no_cache_cost"], bm["cache_read"], bm["cache_write"], calls=bm["calls"])
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
        if log_files:
            log_tally = {}
            print_call_log_section(price_call_logs(log_files, prices, log_tally))
            if any(t["cache_read"] or t["cache_write"] for t in log_tally.values()):
                print_cache_by_model(log_tally, "Prompt caching savings by model (every logged call):")
            print("(No finished games with token_usage found; the call logs above cover every logged call.)")
            return
        sys.exit(f"No games with token_usage found ({skipped} games without it: run with the "
                 f"updated run_craft.py to record tokens).")

    # Per-game table
    show_cache = any(r["cache_read_tokens"] or r["cache_write_tokens"] or r["prompt_cache"] for r in rows)
    print(f"{'structure':<16}{'condition':<10}{'director':<24}{'builder':<24}{'turns':>6}"
          f"{'dir $':>11}{'builder $':>11}{'total $':>11}")
    for r in rows:
        print(f"{str(r['structure']):<16}{str(r['condition']):<10}{str(r['director_model'])[:23]:<24}"
              f"{str(r['builder_model'])[:23]:<24}"
              f"{str(r['turns']):>6}{r['director_cost']:>11.4f}{r['builder_cost']:>11.4f}{r['total_cost']:>11.4f}"
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
        print_cache_by_model(game_model_tally, "Prompt caching savings by model (finished games):")

    if skipped:
        print(f"Note: {skipped} game(s) had no token_usage (recorded before token logging) and were skipped.")
    if missing_all:
        print(f"WARNING: no price for {sorted(missing_all)}; their tokens are counted but cost $0 above. "
              f"Add --price MODEL=IN,OUT.")
    unpriced = sum(r["unpriced_calls"] for r in rows)
    if unpriced:
        print(f"Note: {unpriced} call(s) reported no usage or errored and could not be priced.")

    if log_files:
        log_tally = {}
        log_total = print_call_log_section(price_call_logs(log_files, prices, log_tally))
        if any(t["cache_read"] or t["cache_write"] for t in log_tally.values()):
            print_cache_by_model(log_tally, "Prompt caching savings by model (every logged call):")
        if log_total - grand > 0.0005:
            print(f"  The call logs total ${log_total - grand:.4f} more than the finished games above: "
                  f"calls from interrupted or unfinished games.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Per-game costs written to {args.csv}")


if __name__ == "__main__":
    main()