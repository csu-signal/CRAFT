"""
The full-game eval protocol, shared by eval_full_game.py (local models) and
baseline_sanity_check.py (API builders). Everything that has to be identical
across conditions for the comparison to be fair lives here: which structures,
how many episodes each, per-episode seeds and starting boards, the oracle
setting, prompting style, and how results are recorded. The two scripts only
supply a generate_fn.

Pairing: episode `rep` of structure `s` gets the same seed -- hence the same
starting board, director speaking order and oracle candidate sample -- in
every condition, so conditions can be compared structure by structure.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from agents.builder_agent import BUILDER_SYSTEM_PROMPT_ORACLE, BUILDER_SYSTEM_PROMPT_BASE
from data_split import BENCHMARK_PATH, DEFAULT_TRAIN_POOL_PATH
from eval_results import EpisodeLog, aggregate_episodes, episode_metrics, save_eval_results
from rollout import run_builder_episode, seeded_part_type

COT_INSTRUCTION = (
    " Before answering, reason step by step: identify which director(s) are giving an actionable "
    "instruction, translate it into board coordinates using that director's frame of reference, and "
    "check block colour, size and layer against the current board. Write your reasoning first, then "
    "give your final answer as the LAST line, in exactly one of the PLACE/REMOVE/CLARIFY formats."
)

DEFAULT_MAX_TOKENS = {"default": 220, "cot": 1024}


def builder_system_prompt(oracle_shown, prompt_style="default"):
    base = BUILDER_SYSTEM_PROMPT_ORACLE if oracle_shown else BUILDER_SYSTEM_PROMPT_BASE
    return base + (COT_INSTRUCTION if prompt_style == "cot" else "")


def add_protocol_args(parser):
    g = parser.add_argument_group("eval protocol (keep identical across conditions)")
    g.add_argument("--structures_path", type=str, default=str(BENCHMARK_PATH),
                   help="held-out structures JSON (default: the shipped 20-structure benchmark)")
    g.add_argument("--train_pool", type=str, default=str(DEFAULT_TRAIN_POOL_PATH),
                   help="training pool, checked for overlap with --structures_path")
    g.add_argument("--n_structures", type=int, default=None, help="default: all structures in the file")
    g.add_argument("--episodes_per_structure", type=int, default=1)
    g.add_argument("--max_turns", type=int, default=20)
    g.add_argument("--oracle_n", type=int, default=20,
                   help="oracle candidates sampled per turn (also used for scoring when hidden)")
    g.add_argument("--no_oracle", action="store_true",
                   help="ablation: sample the candidates as usual but don't show them to the builder")
    g.add_argument("--part_type", default="seeded", choices=["seeded", "empty", "random"],
                   help="starting board: 'seeded' = deterministic partial start per (structure, episode), "
                        "matching training's distribution; 'empty' = run_craft.py's benchmark setting; "
                        "'random' = legacy unseeded behaviour (breaks pairing -- don't use for results)")
    g.add_argument("--prompt_style", default="default", choices=["default", "cot"])
    g.add_argument("--director_model", type=str, default="gpt-4.1-mini")
    g.add_argument("--seed", type=int, default=42)
    g = parser.add_argument_group("bookkeeping")
    g.add_argument("--label", type=str, required=True,
                   help="unique per condition, e.g. base_7b, cot_7b, echo, rloo, grpo, base_72b, api_gpt-5.4")
    g.add_argument("--run_name", type=str, default=None)
    g.add_argument("--resume", action="store_true", help="continue a crashed run (same --run_name)")
    g.add_argument("--out_dir", type=str, default="eval_results")
    g.add_argument("--out_csv", type=str, default=None, help="default: <out_dir>.csv")
    g.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    g.add_argument("--log_every_episodes", type=int, default=5)


def _structure_key(s):
    return json.dumps(s["structure"], sort_keys=True)


def load_eval_structures(structures_path, train_pool_path):
    structures = json.loads(Path(structures_path).read_text())
    if Path(train_pool_path).exists():
        train = {_structure_key(s) for s in json.loads(Path(train_pool_path).read_text())}
        leaked = [i for i, s in enumerate(structures) if _structure_key(s) in train]
        if leaked:
            raise SystemExit(f"structures {leaked} in {structures_path} also appear in the training pool "
                             f"{train_pool_path} -- not a held-out eval set")
    else:
        print(f"[eval] warning: training pool {train_pool_path} not found, skipping overlap check")
    digest = hashlib.sha256(Path(structures_path).read_bytes()).hexdigest()[:12]
    return structures, digest


def episode_seed(seed, structure_idx, rep):
    return seed * 1_000_003 + structure_idx * 1_000 + rep


def run_eval(args, generate_fn, config, checkpoint=None, director_setup=None):
    """Runs every (structure, episode) not already done, saving as it goes.
    `config` is the condition-specific part (model, decoding); protocol
    settings are added here so every result file records them the same way."""
    director_setup = director_setup or {}
    structures, digest = load_eval_structures(args.structures_path, args.train_pool)
    n_structures = args.n_structures or len(structures)
    structure_indices = list(range(min(n_structures, len(structures))))

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = args.run_name or f"fullgame_{args.label}_seed{args.seed}_{now}"
    # eval_results/ -> eval_results.csv, eval_results_no_oracle/ -> eval_results_no_oracle.csv
    out_csv = args.out_csv or f"{Path(args.out_dir).as_posix().rstrip('/')}.csv"
    config = {
        **config,
        "structures_path": str(args.structures_path), "structures_sha": digest,
        "n_structures": len(structure_indices), "episodes_per_structure": args.episodes_per_structure,
        "max_turns": args.max_turns, "oracle_n": args.oracle_n, "oracle_in_prompt": not args.no_oracle,
        "part_type": args.part_type, "prompt_style": args.prompt_style,
        "director_mode": director_setup.get("director_mode", "api"), "director_model": args.director_model,
        "seed": args.seed,
    }
    print(f"[eval] {run_name}: " + json.dumps(config))

    log = EpisodeLog(args.out_dir, run_name, resume=args.resume)
    use_wandb = args.report_to == "wandb"
    if use_wandb:
        import wandb
        wandb.init(project=os.getenv("WANDB_PROJECT", "craft_echo"), entity=os.getenv("WANDB_ENTITY"),
                   name=run_name, group="eval_full_game", config={"label": args.label, **config},
                   resume="allow" if args.resume else None, id=run_name if args.resume else None)

    total = len(structure_indices) * args.episodes_per_structure
    window = []
    for structure_idx in structure_indices:
        for rep in range(args.episodes_per_structure):
            if log.done(structure_idx, rep):
                continue
            ep_seed = episode_seed(args.seed, structure_idx, rep)
            part_type = {"seeded": seeded_part_type(ep_seed), "empty": "empty", "random": None}[args.part_type]
            print(f"[eval] episode {len(log.episodes) + 1}/{total} structure={structure_idx} rep={rep} "
                  f"part_type={part_type}")
            episode = run_builder_episode(
                structure_data=structures[structure_idx],
                generate_fn=generate_fn,
                structure_index=structure_idx,
                run_id=args.seed,
                oracle_n=args.oracle_n,
                max_turns=args.max_turns,
                director_model_name=director_setup.get("director_model_path", args.director_model),
                director_mode=director_setup.get("director_mode", "api"),
                director_api_key=os.getenv("OPENAI_API_KEY"),
                shared_director_model=director_setup.get("director_pipe"),
                shared_director_tokenizer=director_setup.get("director_tok"),
                seed=ep_seed,
                part_type=part_type,
                oracle_in_prompt=not args.no_oracle,
                command_pick="last" if args.prompt_style == "cot" else "first",
                retry_directors=True,
            )
            record = {"structure_idx": structure_idx, "rep": rep, "seed": ep_seed, **episode_metrics(episode)}
            log.append(record)
            window.append(record)
            if len(window) >= args.log_every_episodes or len(log.episodes) == total:
                agg = aggregate_episodes(window)
                print("  [window] " + " ".join(f"{k}={v['mean']:.3f}" for k, v in agg.items()))
                if use_wandb:
                    wandb.log({f"window/{k}": v["mean"] for k, v in agg.items()}, step=len(log.episodes))
                window = []

    per_episode = sorted(log.episodes, key=lambda e: (e["structure_idx"], e["rep"]))
    agg = save_eval_results(args.out_dir, out_csv, args.label, checkpoint, run_name, config, per_episode)
    print("\n" + "=" * 60 + f"\nFINAL over {len(per_episode)} episodes ({run_name}) -- per-episode mean +- std")
    for k, v in agg.items():
        print(f"  {k:28s} = {v['mean']:.4f} +- {v['std']:.4f}")
    failures = sum(e["director_failure_rate"] * 3 * e["episode_length"] for e in per_episode)
    if failures:
        print(f"  WARNING: {failures:.0f} director calls still failed after retries -- check API quota/keys")
    if use_wandb:
        wandb.summary.update({f"final/{k}_{s}": v[s] for k, v in agg.items() for s in ("mean", "std", "sem")})
        wandb.finish()
    return agg
