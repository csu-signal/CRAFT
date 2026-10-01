"""
Does the builder's simulate_move tool (agents/builder_tools.py, wired up via
BuilderAgent.generate_move_with_tools) meaningfully cut the invalid_move_rate
that's dominated every baseline_sanity_check.py run so far?

simulate_move dry-runs a proposed move against a *copy* of the board and
returns a specific hint on failure ("Count stack height...", "Neither
position nor span_to can be (1,1) or (2,1)...") -- exactly the failure
categories (wrong layer, invisible-cell spans) that showed up as the
dominant invalid-move causes for both Qwen and GPT-4o-mini this week. This
script runs the same harness (same frozen directors, same reward code, same
game engine) as baseline_sanity_check.py, but drives the builder through
BuilderAgent.generate_move_with_tools instead of a single-shot generate_fn,
since the tool loop needs the live game_state object (to dry-run against)
rather than just a prompt string.

Only meaningful for an API builder -- generate_move_with_tools is built on
OpenAI's native function-calling API and has no local-model equivalent, so
this is not something the trainable Qwen policy can use as-is.

Usage:
    python baseline_tools_check.py --director_mode api --director_model gpt-4.1-mini --report_to wandb
"""
import argparse
import copy
import datetime
import os
import random
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from agents.builder_agent import BuilderAgent
from agents.environment import EnhancedGameState, get_oracle_moves
from data_split import load_training_pool
from local_model_utils import load_local_director_pipeline
from reward import compute_builder_reward
from rollout import build_frozen_directors, director_discussion_text, _oracle_match, _is_parse_failure

load_dotenv()

LOCAL_DIRECTOR_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

BUILDER_LOG_KEYS = [
    "progress_delta", "matched_oracle", "parse_failure", "move_invalid", "completed",
    "off_list_penalty", "invalid_move_penalty", "clarify_penalty",
    "completion_bonus", "efficiency_bonus", "action", "training_reward",
]


def _safe_mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_rate(bools):
    vals = [1.0 if bool(v) else 0.0 for v in bools if v is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(reward_infos, group_sizes):
    L = {k: [info[k] for info in reward_infos if k in info] for k in BUILDER_LOG_KEYS}
    rewards = [info.get("training_reward", 0.0) for info in reward_infos]
    return {
        "builder/progress_delta_mean": _safe_mean(L["progress_delta"]),
        "builder/completed_rate": _safe_rate(L["completed"]),
        "builder/oracle_match_rate": _safe_rate([v for v in L["matched_oracle"] if v is not None]),
        "builder/clarify_rate": _safe_rate([a == "clarify" for a in L["action"]]),
        "builder/parse_failure_rate": _safe_rate(L["parse_failure"]),
        "builder/invalid_move_rate": _safe_rate(L["move_invalid"]),
        "builder/mean_episode_length": float(sum(group_sizes) / len(group_sizes)) if group_sizes else float("nan"),
        "reward": _safe_mean(rewards),
        "reward_std": (torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0),
    }


def run_tool_episode(
    builder_agent, structure_data, structure_index, run_id, oracle_n, max_turns,
    director_model_name, director_mode, director_api_key,
    shared_director_model, shared_director_tokenizer, max_simulations, seed=None,
):
    target_structure = structure_data["structure"]
    target_spans = {int(k): v for k, v in structure_data["spans"].items()}
    game_state = EnhancedGameState(
        target_structure=copy.deepcopy(target_structure), target_spans=target_spans, partComplete=True,
    )
    director_agents = build_frozen_directors(
        structure_index, run_id, director_model_name=director_model_name, api_key=director_api_key,
        director_mode=director_mode, shared_director_model=shared_director_model,
        shared_director_tokenizer=shared_director_tokenizer,
    )
    target_director_views = game_state.get_target_director_views()
    rng = random.Random(seed if seed is not None else (structure_index * 7919 + run_id))

    rewards, reward_infos = [], []
    conversation_history = []

    for turn in range(max_turns):
        game_state.increment_turn()
        full_board_state = game_state.get_director_views()
        public_conversation = "\n".join(conversation_history)
        director_order = ["D1", "D2", "D3"]
        rng.shuffle(director_order)

        director_responses = {}
        for did in director_order:
            resp = director_agents[did].generate_response(
                current_view=full_board_state, target_view=target_director_views[did],
                conversation_history=public_conversation, available_blocks=game_state.available_blocks,
            )
            director_responses[did] = resp
            conversation_history.append(f"{did}: {resp['public_message']}")
        discussion = director_discussion_text(director_responses)

        oracle_moves = get_oracle_moves(game_state, n=oracle_n, rng=rng)

        move = builder_agent.generate_move_with_tools(
            director_discussion=discussion, game_state=game_state,
            max_simulations=max_simulations, oracle_moves=oracle_moves,
        )
        matched_oracle = _oracle_match(move, oracle_moves)
        parse_failure = _is_parse_failure(move)

        move_invalid = False
        if move.get("action") == "clarify":
            progress_delta, completed = 0.0, False
            conversation_history.append(f"Builder: {move.get('clarification', '')}")
        else:
            success, progress_data, structurePlacement, sidePlacement, overallState = (
                game_state.execute_move(move)
            )
            if success:
                progress_delta = progress_data["progress_delta"]
                completed = game_state.is_complete()
            else:
                progress_delta, completed = 0.0, False
                move_invalid = True
                print(f"  [execute_move failed] {progress_data.get('error', '<no reason given>')}")
            conversation_history.append(f"Builder: {move.get('confirmation', '')}")

        turns_remaining = max_turns - (turn + 1)
        reward, reward_info = compute_builder_reward(
            move_dict=move, progress_delta=progress_delta, matched_oracle=matched_oracle,
            completed=completed, turns_remaining_at_completion=turns_remaining, parse_failure=parse_failure,
            move_invalid=move_invalid,
        )
        reward_info["turn"] = turn
        print(
            f"  [rollout-tools] structure={structure_index} turn={turn + 1}/{max_turns} "
            f"action={move.get('action')} reward={reward:.3f} progress_delta={progress_delta:.3f} completed={completed}"
        )
        rewards.append(reward)
        reward_infos.append(reward_info)
        if completed:
            break

    return {"rewards": rewards, "reward_infos": reward_infos}


def run(
    builder_model="gpt-4o-mini",
    director_model_name="gpt-4.1-mini",
    director_mode="api",
    director_gpu=1,
    director_max_new_tokens=448,
    director_quantize=None,
    max_simulations=3,
    oracle_n=20,
    max_turns=20,
    n_structures=15,
    episodes_per_structure=1,
    seed=42,
    train_pool_path=None,
    log_every_episodes=5,
    report_to="wandb",
    run_name=None,
):
    print(f"[baseline-tools] builder_model={builder_model!r} max_simulations={max_simulations}")
    print(f"[baseline-tools] director_mode={director_mode!r} director_model={director_model_name!r}")

    structures = load_training_pool(train_pool_path) if train_pool_path else load_training_pool()
    rng = random.Random(seed)
    structure_indices = rng.sample(range(len(structures)), min(n_structures, len(structures)))

    director_pipe, director_tok, director_model_path = None, None, director_model_name
    if director_mode == "local":
        director_model_path = LOCAL_DIRECTOR_MODELS.get(director_model_name, director_model_name)
        print(f"loading shared local director model {director_model_path} on cuda:{director_gpu}...")
        director_pipe, director_tok = load_local_director_pipeline(
            director_model_path, quantize=director_quantize, gpus=[director_gpu],
            max_new_tokens=director_max_new_tokens,
        )
    else:
        print(f"using OpenAI API directors: {director_model_path} (no local model loaded)")

    builder_agent = BuilderAgent(model_name=builder_model)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = run_name or f"tools_{builder_model}_{director_model_name}_sim{max_simulations}_seed{seed}_{now}"

    use_wandb = report_to == "wandb"
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "craft_echo"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            group="baseline_tools_check",
            config={
                "builder_model": builder_model, "director_model": director_model_name,
                "director_mode": director_mode, "max_simulations": max_simulations,
                "oracle_n": oracle_n, "max_turns": max_turns,
                "n_structures": n_structures, "episodes_per_structure": episodes_per_structure,
            },
        )

    all_infos_window, group_sizes_window = [], []
    all_infos_total, group_sizes_total = [], []
    episode_num = 0
    total_episodes = len(structure_indices) * episodes_per_structure

    for structure_idx in structure_indices:
        structure_data = structures[structure_idx]
        for _ in range(episodes_per_structure):
            episode_num += 1
            print(f"[baseline-tools] episode {episode_num}/{total_episodes} (structure_idx={structure_idx})")
            episode = run_tool_episode(
                builder_agent=builder_agent, structure_data=structure_data, structure_index=structure_idx,
                run_id=seed, oracle_n=oracle_n, max_turns=max_turns,
                director_model_name=director_model_path, director_mode=director_mode,
                director_api_key=os.getenv("OPENAI_API_KEY") if director_mode == "api" else None,
                shared_director_model=director_pipe, shared_director_tokenizer=director_tok,
                max_simulations=max_simulations, seed=seed * 7919 + episode_num,
            )
            all_infos_window.extend(episode["reward_infos"])
            group_sizes_window.append(len(episode["rewards"]))
            all_infos_total.extend(episode["reward_infos"])
            group_sizes_total.append(len(episode["rewards"]))

            if episode_num % log_every_episodes == 0 or episode_num == total_episodes:
                metrics = summarize(all_infos_window, group_sizes_window)
                print(f"  [{episode_num}/{total_episodes}] " + " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in metrics.items()))
                if use_wandb:
                    wandb.log(metrics, step=episode_num)
                all_infos_window, group_sizes_window = [], []

    final_metrics = summarize(all_infos_total, group_sizes_total)
    print("\n" + "=" * 60)
    print(f"FINAL over {total_episodes} episodes ({run_name})")
    for k, v in final_metrics.items():
        print(f"  {k:35s} = {v:.4f}")
    print("=" * 60)

    if use_wandb:
        wandb.summary.update({f"final_{k}": v for k, v in final_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline check: builder WITH simulate_move tool-calling enabled")
    parser.add_argument("--builder_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--director_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--director_mode", type=str, default="api", choices=["api", "local"])
    parser.add_argument("--director_gpu", type=int, default=1)
    parser.add_argument("--director_max_new_tokens", type=int, default=448)
    parser.add_argument("--director_quantize", type=str, default=None, choices=[None, "4bit", "8bit"])
    parser.add_argument("--max_simulations", type=int, default=3, help="simulate_move calls allowed per turn")
    parser.add_argument("--oracle_n", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--n_structures", type=int, default=15, help="smaller than baseline_sanity_check's 30 -- each turn can cost up to max_simulations+1 API calls now")
    parser.add_argument("--episodes_per_structure", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_pool", type=str, default=None)
    parser.add_argument("--log_every_episodes", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run(
        builder_model=args.builder_model,
        director_model_name=args.director_model,
        director_mode=args.director_mode,
        director_gpu=args.director_gpu,
        director_max_new_tokens=args.director_max_new_tokens,
        director_quantize=args.director_quantize,
        max_simulations=args.max_simulations,
        oracle_n=args.oracle_n,
        max_turns=args.max_turns,
        n_structures=args.n_structures,
        episodes_per_structure=args.episodes_per_structure,
        seed=args.seed,
        train_pool_path=args.train_pool,
        log_every_episodes=args.log_every_episodes,
        report_to=args.report_to,
        run_name=args.run_name,
    )
