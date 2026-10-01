"""
Builds a frozen within-turn evaluation pool.

For a batch of (structure, turn) states, this captures the board state, the
director discussion, and the oracle candidate list (with each candidate's
overall_progress) once and saves it. Every checkpoint compared in
eval_within_turn.py is later scored against this exact same frozen pool, keeping the
comparison across checkpoints uncounfounded: if directors were resampled
fresh per evaluation, a score difference between two checkpoints couldn't be
attributed to the checkpoints rather than to which discussion happened to be
drawn.

A real BuilderAgent (API model, matching CRAFT's own default builder) is
used purely as the *reference policy that advances the board between
snapshots* -- it is not one of the checkpoints being evaluated. Its only job
here is to produce a plausible sequence of turns to sample states from.

Usage:
    python build_eval_pool.py --turns_per_structure 6
"""
import argparse
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from agents.builder_agent import BuilderAgent
from agents.environment import EnhancedGameState

from data_split import load_benchmark_structures
from rollout import build_frozen_directors, director_discussion_text, sample_oracle_candidates, candidate_moves_only

load_dotenv()

DEFAULT_POOL_PATH = Path(__file__).resolve().parent / "data" / "within_turn_eval_pool.json"


def build_pool(
    structures=None,
    turns_per_structure=6,
    oracle_n=5,
    reference_builder_model="gpt-4o-mini",
    director_model_name="gpt-4.1-mini",
    max_turns=15,
    seed=0,
    out_path=DEFAULT_POOL_PATH,
):
    random.seed(seed)  # EnhancedGameState.__init__ draws partType from the global `random` module
    structures = structures or load_benchmark_structures()
    reference_builder = BuilderAgent(model_name=reference_builder_model)
    rng = random.Random(seed)
    pool = []

    for structure_idx, structure_data in enumerate(structures):
        target_structure = structure_data["structure"]
        target_spans = {int(k): v for k, v in structure_data["spans"].items()}
        game_state = EnhancedGameState(
            target_structure=target_structure, target_spans=target_spans, partComplete=True,
        )
        target_director_views = game_state.get_target_director_views()
        director_agents = build_frozen_directors(
            structure_idx, run_id=seed, director_model_name=director_model_name,
        )
        conversation_history = []
        captured = 0

        for turn in range(max_turns):
            if captured >= turns_per_structure:
                break
            game_state.increment_turn()
            full_board_state = game_state.get_director_views()
            public_conversation = "\n".join(conversation_history)
            director_order = ["D1", "D2", "D3"]
            rng.shuffle(director_order)

            director_responses = {}
            for did in director_order:
                resp = director_agents[did].generate_response(
                    current_view=full_board_state,
                    target_view=target_director_views[did],
                    conversation_history=public_conversation,
                    available_blocks=game_state.available_blocks,
                )
                director_responses[did] = resp
                conversation_history.append(f"{did}: {resp['public_message']}")

            discussion = director_discussion_text(director_responses)
            oracle_candidates = sample_oracle_candidates(game_state, n=oracle_n, rng=rng)

            if not oracle_candidates:
                continue

            oracle_moves = candidate_moves_only(oracle_candidates)

            pool.append({
                "structure_idx": structure_idx,
                "turn": turn,
                "structure_before": {k: list(v) for k, v in game_state.current_structure.items()},
                "available_blocks": list(game_state.available_blocks),
                "director_discussion": discussion,
                "oracle_moves": oracle_candidates,   # full entries, WITH overall_progress
            })
            captured += 1

            move = reference_builder.generate_move(
                director_discussion=discussion,
                current_state=game_state.current_structure,
                available_blocks=game_state.available_blocks,
                oracle_moves=oracle_moves,
            )
            if move.get("action") == "clarify":
                conversation_history.append(f"Builder: {move.get('clarification', '')}")
                continue
            success, _, _, _, overallState = game_state.execute_move(move)
            conversation_history.append(f"Builder: {move.get('confirmation', '')}")
            if overallState:
                break

        print(f"structure {structure_idx}: captured {captured} turns")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pool, f)
    print(f"wrote {len(pool)} frozen turns -> {out_path}")
    return pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns_per_structure", type=int, default=6)
    parser.add_argument("--oracle_n", type=int, default=5)
    parser.add_argument("--reference_builder_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--director_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(DEFAULT_POOL_PATH))
    args = parser.parse_args()
    build_pool(
        turns_per_structure=args.turns_per_structure,
        oracle_n=args.oracle_n,
        reference_builder_model=args.reference_builder_model,
        director_model_name=args.director_model,
        max_turns=args.max_turns,
        seed=args.seed,
        out_path=args.out,
    )
