"""
Within-turn evaluation.

Loads a checkpoint, samples G completions per frozen turn in the eval pool
(build_eval_pool.py), and reports the four metrics from the within-turn
experiment design: best-candidate rate, regret, compliance rate, and
P(CLARIFY). No environment stepping -- every candidate's overall_progress
was already computed by enumerate_correct_actions when the pool was built,
so scoring is just comparing the parsed move against that list.

    python eval_within_turn.py --checkpoint craft_echo_runs/.../final_model --label echo
    python eval_within_turn.py --checkpoint craft_echo_runs/.../final_model --label episode_return
    python eval_within_turn.py --checkpoint craft_echo_runs/.../final_model --label rloo_per_turn
    python eval_within_turn.py --base_model Qwen/Qwen2.5-1.5B-Instruct --label base   # zero-shot, no checkpoint

Run once per condition (with a different --checkpoint/--label each time);
each run appends one row to --out_csv, building up the reporting table.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from agents.builder_agent import BuilderAgent

from build_eval_pool import DEFAULT_POOL_PATH

# Reused only for its pure prompt/parse helpers -- see rollout.py's module
# docstring for why a dummy key is passed.
_BUILDER_HELPER = BuilderAgent(api_key="unused-local-generation-only")


def load_pool(path=DEFAULT_POOL_PATH):
    with open(path) as f:
        return json.load(f)


def load_policy(base_model_name, checkpoint=None, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint or base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Real builder prompts run ~2700-2800 tokens (see train.py's matching
    # note) -- truncate from the left so a too-small max_length below drops
    # older static reference material rather than the live board state /
    # discussion / oracle candidates at the end of the prompt.
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint).to(device)
    model.eval()
    return model, tokenizer


def sample_completions(model, tokenizer, prompt_text, n_samples, max_new_tokens, temperature, device):
    input_ids = tokenizer.encode(
        prompt_text, return_tensors="pt", truncation=True, max_length=3584, add_special_tokens=False,
    ).to(device)
    completions = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output[0][input_ids.shape[1]:]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return completions


def _match_candidate(move, oracle_candidates):
    """oracle_candidates: full entries (from build_eval_pool.py), each with a
    "move" sub-dict and "overall_progress". Returns the matching full entry,
    or None if the parsed move doesn't match any candidate."""
    for c in oracle_candidates:
        m = c["move"]
        if (move.get("action") == m["action"]
                and move.get("position") == m["position"]
                and move.get("layer") == m["layer"]):
            return c
    return None


def evaluate(model, tokenizer, pool, n_samples=8, max_new_tokens=150, temperature=1.0, device="cuda:0"):
    n_total = 0
    n_compliant = 0
    n_best = 0
    n_clarify = 0
    regrets = []

    for turn_state in pool:
        oracle_candidates = turn_state["oracle_moves"]  # full entries, with overall_progress
        if not oracle_candidates:
            continue
        best_progress = max(c["overall_progress"] for c in oracle_candidates)
        oracle_moves = [c["move"] for c in oracle_candidates]  # plain move dicts, for the prompt

        prompt_text = _BUILDER_HELPER.create_builder_prompt(
            director_discussion=turn_state["director_discussion"],
            current_state=turn_state["structure_before"],
            available_blocks=turn_state["available_blocks"],
            oracle_moves=oracle_moves,
        )
        completions = sample_completions(
            model, tokenizer, prompt_text, n_samples, max_new_tokens, temperature, device,
        )

        for text in completions:
            n_total += 1
            first_line = text.split("\n")[0].strip()
            move = _BUILDER_HELPER.parse_builder_response(first_line)

            if move.get("action") == "clarify":
                n_clarify += 1
                clarification = move.get("clarification", "") or ""
                is_parse_failure = clarification.startswith(("Could not parse", "Parse error"))
                if not is_parse_failure:
                    n_compliant += 1  # a genuine CLARIFY is compliant, just has no candidate match
                continue

            matched = _match_candidate(move, oracle_candidates)
            if matched is None:
                continue  # off-list -- non-compliant, no regret contribution

            n_compliant += 1
            regrets.append(best_progress - matched["overall_progress"])
            if matched["overall_progress"] >= best_progress - 1e-9:
                n_best += 1

    return {
        "n_total": n_total,
        "compliance_rate": n_compliant / n_total if n_total else float("nan"),
        "clarify_rate": n_clarify / n_total if n_total else float("nan"),
        "best_candidate_rate": n_best / len(regrets) if regrets else float("nan"),
        "regret_mean": float(np.mean(regrets)) if regrets else float("nan"),
        "regret_std": float(np.std(regrets)) if regrets else float("nan"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Within-turn evaluation of a CRAFT builder checkpoint")
    parser.add_argument("--pool", type=str, default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint dir; omit for zero-shot base")
    parser.add_argument("--label", type=str, required=True,
                        help="row label for the results table, e.g. echo / episode_return / rloo_per_turn / base")
    parser.add_argument("--n_samples", type=int, default=8, help="G -- samples per frozen turn")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out_csv", type=str, default="within_turn_results.csv")
    args = parser.parse_args()

    pool = load_pool(args.pool)
    model, tokenizer = load_policy(args.base_model, args.checkpoint)
    results = evaluate(
        model, tokenizer, pool,
        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    results["label"] = args.label
    print(json.dumps(results, indent=2))

    out_csv = Path(args.out_csv)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "best_candidate_rate", "regret_mean", "regret_std",
            "compliance_rate", "clarify_rate", "n_total",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(results)
    print(f"appended row to {out_csv}")
