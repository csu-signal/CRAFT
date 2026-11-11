# API mode, all models, all structures
# python run_craft.py --mode api

# # API mode, single model, single structure
# python run_craft.py --mode api --director gpt-4o-mini --structures 0

# # Local mode, specific model, with oracle
# python run_craft.py --mode local --director qwen-7b --oracle --oracle_n 5
# python run_craft.py --mode api --director gpt-4o-mini --oracle --oracle_n 5 --structures 0,1,2 --turns 5
# python run_craft.py --mode api --director gpt-4o-mini --structures 0,1,2 --turns 5
# # Local mode, no tools, specific structures
# python run_craft.py --mode local --director qwen-7b --no_tools --structures 0,1,2

# # Custom output dir
# python run_craft.py --mode api --director gpt-4o --output my_results --turns 15#

# # Utterance-selection conditions (full games), see "UTTERANCE SELECTION" below:
# #   baseline -- the Builder hears every director (the original game)
# #   policy   -- the Builder hears only the policy's chosen director
# #   random   -- the Builder hears one director drawn uniformly at random
# # Every turn of every game logs all three selections (the policy and random
# # picks are computed as "shadow" choices even when they don't drive the game).
# python run_craft.py --mode api --director gpt-4o-mini --structures 0,1,2 --turns 20 \
#     --conditions baseline,policy,random --policy_path /path/to/policy.pt

# # Token estimate (no API calls): every builder/director call is answered by the
# # local Qwen model in token_estimator.py (next to this script) and its tokens are
# # recorded. Use the API model names you plan to pay for; results go to a separate
# # "_token_estimate" folder and the totals to token_estimate.json there.
# python run_craft.py --mode api --director claude-haiku-4-5 --builder claude-haiku-4-5 \
#     --structures 0 --turns 20 --conditions baseline,policy,random \
#     --policy_path /path/to/policy.pt --token_estimate
# # Same, with the estimator served by Ollama (no 30 GB download; runs on a laptop):
# #   ollama pull qwen2.5:14b-instruct && ollama serve
# python run_craft.py ... --token_estimate --estimate_backend ollama

import os
import json
import copy
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from agents.common_ground_agent import CommonGroundAgent
# from chair_agent import ChairAgent
from local_model_utils import load_local_director_model, load_local_director_pipeline
from structure_generator_v2 import get_director_views as get_director_views_fn
from agents.environment import (
    EnhancedGameState, get_oracle_moves
)
from agents.director_agent import DirectorAgent
from agents.builder_agent import BuilderAgent, BuilderType
from agents.oracle import enumerate_correct_actions
import inspect
import time
import sys

print("EnhancedGameState from:", inspect.getfile(EnhancedGameState))
print("get_director_views sig:", inspect.signature(EnhancedGameState.get_director_views))

print([m for m in dir(EnhancedGameState) if not m.startswith("_")])

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def view_cell(v):
    return (v["color"], v["size"])

def view_distance(curr, targ):
    mism, total = 0, 0
    for d in ["D1", "D2", "D3"]:
        for r in ["row_0", "row_1", "row_2"]:
            for j in range(3):
                total += 1
                if view_cell(curr[d][r][j]) != view_cell(targ[d][r][j]):
                    mism += 1
    return mism, total

def compare_views(label, a, b):
    if a == b:
        print(f"[OK] {label}: views match")
        return True
    print(f"[MISMATCH] {label}")
    for d in ["D1", "D2", "D3"]:
        if a[d] != b[d]:
            print(f"  Director {d} differs:")
            print("  stored   :", json.dumps(a[d]))
            print("  recomputed:", json.dumps(b[d]))
    return False

def chair_history_only_directors(conversation_history: list[str], max_chars=2000) -> str:
    lines = [ln for ln in conversation_history if ln.startswith(("D1:", "D2:", "D3:"))]
    s = "\n".join(lines)
    return s[-max_chars:]


# ─────────────────────────────────────────
# UTTERANCE SELECTION (baseline / policy / random)
# ─────────────────────────────────────────
#
# Each turn, after the directors have spoken, three selections are computed:
#   baseline -- every speaking director (what the original game gives the Builder)
#   policy   -- the policy's pick (only if a policy is loaded)
#   random   -- a uniform draw from the speaking directors
# The game's --conditions value decides which one the Builder actually hears;
# all three are logged in turn_data['utterance_selection'] regardless.
#
# Directors always hear each other: conversation_history still gets every
# director's message. Only the Builder's input is restricted.

DIRECTOR_IDS = ("D1", "D2", "D3")
CONDITIONS = ("baseline", "policy", "random")
# Older name for 'baseline', still accepted on the command line.
CONDITION_ALIASES = {"all": "baseline"}
SILENT_MESSAGES = {"", "no message provided", "none", "null", "n/a"}


def is_silent_message(text) -> bool:
    return (str(text or "")).strip().lower() in SILENT_MESSAGES


def random_pick(seed, *identity, pool: List[str]) -> Optional[str]:
    """
    Uniform draw from `pool`, seeded from --random_seed plus the identity of the
    decision (structure, run, turn). Uses its own random.Random, so it never
    touches the global RNG that orders the directors, and the draw for a given
    turn is the same in every condition.
    """
    if not pool:
        return None
    key = "|".join(str(x) for x in (seed,) + identity)
    return random.Random(key).choice(sorted(pool))


def load_policy_scorer(policy_path, code_dir=None, feature_mode=None, feature_tier=None,
                       allow_silent=False):
    """
    Load the trained index policy with the same PolicyScorer the multi-turn
    rollout harness uses (run_multiTurn_builder_local.py, in this folder), so
    directors are scored identically in both places.
    """
    try:
        from run_multiTurn_builder_local import PolicyScorer, find_policy_code_dir
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[policy] Could not import PolicyScorer from run_multiTurn_builder_local.py ({e}). "
                 f"Run from the CRAFT folder that contains that file.")
    p = Path(policy_path)
    return PolicyScorer(
        p, find_policy_code_dir(p, Path(code_dir) if code_dir else None),
        feature_mode=feature_mode, feature_tier=feature_tier, allow_silent=allow_silent,
    )


class PolicySelector:
    """
    Per-game wrapper around PolicyScorer. Keeps each director's ArmState
    history for THIS game and scores D1-D3 each turn exactly as the policy was
    trained: history + the current turn's candidate arm -> encode -> argmax.

    score_turn() is called after the directors speak (before the Builder
    moves); commit() is called once the turn's outcome is known, adding the
    current turn to the history with its after-board and "followed" tag.
    """

    def __init__(self, scorer, target_structure, structure_id, model_combo):
        from run_multiTurn_builder_local import make_sat_fn
        self.scorer = scorer
        self.sat_fn = make_sat_fn(target_structure)
        self.structure_id = structure_id
        self.combo = model_combo
        self.hist = {d: [] for d in DIRECTOR_IDS}
        self._pending = None

    def score_turn(self, turn_number, board_before, messages: Dict[str, str], speaking: List[str]):
        before = copy.deepcopy(board_before)
        self._pending = {
            d: self.scorer.make_arm(d, messages.get(d, ""), turn_number, self.structure_id,
                                    self.combo, before, before, tag=0,
                                    sat=self.sat_fn(d, before))
            for d in DIRECTOR_IDS
        }
        if not speaking:
            return None, None, None
        scores = self.scorer.score({d: self.hist[d] + [self._pending[d]] for d in DIRECTOR_IDS})
        chosen, raw = self.scorer.choose(scores, speaking)
        return chosen, raw, scores

    def commit(self, followed: Optional[str], board_after, move_failed: bool):
        """followed: the director the Builder heard alone (policy/random
        conditions), or None when it heard everyone (tag 0, as for CRAFT
        history in the rollout harness)."""
        if self._pending is None:
            return
        Belief = self.scorer.arm_mod.ConstraintBelief
        for d, arm in self._pending.items():
            arm.constraint = Belief(0 if followed is None else (1 if d == followed else -1))
            arm.structureAfter = copy.deepcopy(board_after)
            arm.moveFailed = "True" if move_failed else ""
            self.hist[d].append(arm)
        self._pending = None


# ─────────────────────────────────────────
# TOKEN ESTIMATE (--token_estimate)
# ─────────────────────────────────────────
#
# Every agent keeps its model client in `.client`, whichever provider it routes
# to (OpenAI, Anthropic, Gemini's OpenAI endpoint, or Ollama). For an estimate,
# those clients are replaced with EstimatorClients from token_estimator.py, so
# the calls themselves are unchanged and nothing reaches an API or Ollama.
# Gemini directors build a new google.genai Client on every call, so that class
# is replaced in __main__. Each recorded call is tagged with the role, director
# model and condition, so the totals can be priced per model.

ESTIMATE_DUMMY_KEYS = ("OPENAI_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


def make_estimator_client(est, role):
    """An EstimatorClient that tags each call it records with `role`."""
    from token_estimator import EstimatorClient

    class TaggedEstimatorClient(EstimatorClient):
        def _tag(self, response):
            with self.est.lock:
                self.est.calls[-1]["role"] = role
            return response

        def _openai_create(self, *a, **k):
            return self._tag(super()._openai_create(*a, **k))

        def _anthropic_create(self, *a, **k):
            return self._tag(super()._anthropic_create(*a, **k))

        def _gemini_generate(self, *a, **k):
            return self._tag(super()._gemini_generate(*a, **k))

    return TaggedEstimatorClient(est)


def attach_estimator(est, builder_agent, director_agents, common_ground_agent=None):
    """Point every agent at the estimator instead of its real client."""
    builder_agent.client = make_estimator_client(est, "builder")
    for agent in director_agents.values():
        if getattr(agent, "use_api", True):
            agent.client = make_estimator_client(est, "director")
    if common_ground_agent is not None and hasattr(common_ground_agent, "client"):
        common_ground_agent.client = make_estimator_client(est, "common_ground")


def summarize_estimate(est) -> Dict:
    """Totals overall, by role, and by director model x condition x role,
    scaled by token_estimator.CORRECTION like est.report()."""
    from token_estimator import CORRECTION

    def empty():
        return {"calls": 0, "truncated_calls": 0, "input_tokens": 0, "output_tokens": 0}

    by_role, by_group = defaultdict(empty), defaultdict(empty)
    for c in est.calls:
        role = c.get("role", "unknown")
        group = f"{c.get('director_model', '?')} | {c.get('condition', '?')} | {role}"
        for bucket in (by_role[role], by_group[group]):
            bucket["calls"] += 1
            bucket["truncated_calls"] += int(c["truncated"])
            bucket["input_tokens"] += c["input_tokens"]
            bucket["output_tokens"] += c["output_tokens"]
    for bucket in list(by_role.values()) + list(by_group.values()):
        bucket["input_tokens"] = round(bucket["input_tokens"] * CORRECTION)
        bucket["output_tokens"] = round(bucket["output_tokens"] * CORRECTION)
    return {"totals": est.report(), "by_role": dict(by_role),
            "by_director_model_condition_role": dict(sorted(by_group.items()))}


def condition_tag(condition, policy_path=None, random_seed=0):
    if condition == "policy":
        return f"cond-policy-{Path(policy_path).stem}"
    if condition == "random":
        return f"cond-random-rseed{random_seed}"
    return "cond-baseline"


# ─────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────

def run_craft_experiments(
    dataset_path="structures_dataset_v2.json",
    structure_index=0,
    director_model_name="gpt-4.1-mini",
    builder_model_name="gpt-4o",
    common_ground_model_name="gpt-4.1-mini",
    api_key="None",
    max_turns=15,
    output_dir="craft_results",
    use_common_ground=False,
    run = 1,
    lastPartType = None,
    shared_model = None,
    shared_tokenizer = None,
    builder_tool_use = True,
    use_oracle=False,
    num_oracle = None,
    max_tokens = None,
    condition = "baseline",
    policy_scorer = None,
    policy_path = None,
    random_seed = 0,
    token_estimator = None,
    builder_parse_retries = 2,
    builder_max_tokens = None,
    builder_one_line_instruction = True,
):
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_names = ['Turn', 'Failed Move', 'Clarify', 'Structure Placement', 'Side Placement', 'Overall Structure State', 'Group Agreement', 'Transcription' ]
    correctness = pd.DataFrame(columns=col_names)

    # ── Load structure ────────────────────────────────────────
    with open(dataset_path, "r") as f:
        loaded_structures = json.load(f)

    sample = loaded_structures[structure_index]

    # Fix span keys: JSON saves them as strings, need ints
    target_spans = {int(k): v for k, v in sample['spans'].items()}

    print(f"\nLoaded: {sample['id']} (complexity={sample['complexity']})")
    print(f"Metadata: {sample['metadata']}")

    swapped_dir_views = {d: {row: items[::-1] for row, items in rows.items()} if d in ["D2", "D3"] else rows for d, rows in sample["director_views"].items()}
    target_structures_list = [{
        'id'            : sample['id'],
        'complexity'    : sample['complexity'],
        'structure'     : sample['structure'],
        'spans'         : target_spans,
        'director_views': swapped_dir_views,

        'metadata'      : sample['metadata']
    }]

    # ── Init results ──────────────────────────────────────────
    all_results = {
    'experiment_info': {
        'timestamp'      : current_time,
        'dataset_path'   : dataset_path,
        'structure_index': structure_index,
        'max_turns'      : max_turns,
        'models'         : {
            'director': director_model_name,
            'builder' : builder_model_name
        },

        'use_oracle'     : use_oracle,
        'num_oracle'     : num_oracle,
        'builder_tool_use': builder_tool_use,
        'use_common_ground': use_common_ground,
        'run'            : run,
        'max_tokens'     : max_tokens,
        'partial_completion': lastPartType,
        # utterance selection
        'condition'      : condition,
        'policy_path'    : str(policy_path) if policy_path else None,
        'policy_feature_mode': policy_scorer.cfg.FEATURE_MODE if policy_scorer else None,
        'policy_feature_tier': policy_scorer.cfg.FEATURE_TIER if policy_scorer else None,
        'policy_loader': getattr(policy_scorer, 'loader', None) if policy_scorer else None,
        'policy_arch': getattr(policy_scorer, 'arch', None) if policy_scorer else None,   # incl. recov_head / recov_weight
        'random_seed'    : random_seed,
        'builder_parse_retries': builder_parse_retries,
        'builder_max_tokens': builder_max_tokens,       # None = provider default (see BuilderAgent)
        'builder_one_line_instruction': builder_one_line_instruction,
            },
        'games': []
    }
    #add director builder combo to name the outer dir
    model_combo_dir = f"{output_dir}/{director_model_name}_{builder_model_name}/"
    os.makedirs(model_combo_dir, exist_ok=True)

    md_path  = f"{model_combo_dir}/craft_{sample['id']}_{run}.md"
    json_path = f"{model_combo_dir}/craft_{sample['id']}_{run}.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    main_grice_correctness_filename = f"{model_combo_dir}/craft_{sample['id']}_correctness_{run}.csv"

    with open(md_path, 'w') as f:
        f.write(f"# CRAFT Results — {sample['structure']}\n\n")
        f.write(f"Condition: {condition}\n\n")

    # ── Run each structure ────────────────────────────────────
    for idx, structure_data in enumerate(tqdm(target_structures_list)):

        print(f"\n===== Structure {idx+1}: {structure_data['id']} =====")
        if shared_model is None:
            print("calling api dirs ")

            director_agents = {
                did: DirectorAgent(
                    director_id=did,
                    use_api=True,
                    api_key=api_key,
                    model_name=director_model_name,
                    structure_index=structure_index,
                    run=run,
                    max_tokens=max_tokens,
                )
                for did in ["D1", "D2", "D3"]
            }
        else:
            print("calling local dirs ")
            director_agents = {
            did: DirectorAgent(
                director_id=did,
                use_api=False,
                model_name=director_model_name,
                local_model=shared_model,
                local_tokenizer=shared_tokenizer,
                structure_index=structure_index,
                run=run,
                max_tokens=max_tokens,
            )
            for did in ["D1", "D2", "D3"]
        }


        builder_agent = BuilderAgent(api_key=api_key, model_name=builder_model_name)
        builder_agent.max_parse_retries = builder_parse_retries
        builder_agent.one_line_instruction = builder_one_line_instruction
        if builder_max_tokens:
            builder_agent.max_output_tokens = builder_max_tokens
        all_results['experiment_info']['builder_max_tokens_used'] = builder_agent.max_output_tokens

        common_ground_agent = None
        if use_common_ground:
            common_ground_agent = CommonGroundAgent(
                use_api=True,
                api_key=api_key,
                model_name=common_ground_model_name
            )

        if token_estimator is not None:
            attach_estimator(token_estimator, builder_agent, director_agents, common_ground_agent)

        # A fresh policy history for every game.
        policy_selector = None
        if policy_scorer is not None:
            policy_selector = PolicySelector(
                policy_scorer, structure_data['structure'], structure_data['id'],
                model_combo=f"{director_model_name}+{builder_model_name}",
            )

        try:
            game_result, correctness, lastPartType = run_single_game(
                structure_data=structure_data,
                director_agents=director_agents,
                builder_agent=builder_agent,
                # chair_agent=chair_agent,
                common_ground_agent=common_ground_agent,

                max_turns=max_turns,
                structure_idx=idx,
                correctness=correctness,
                use_common_ground=use_common_ground,
                run = run,
                lastPartType = lastPartType,
                use_tools = builder_tool_use,
                use_oracle=use_oracle,
                num_oracle = num_oracle,
                condition = condition,
                policy_selector = policy_selector,
                random_seed = random_seed,
            )
        except Exception as e:
            print(f"Error in structure {idx}: {e}")
            game_result = {
                'structure_id': structure_data['id'],
                'error': str(e),
                'completed': False,
                'turns_taken': 0,
                'final_progress': 0.0,
                'condition': condition,
            }

        all_results['games'].append(game_result)

        with open(md_path, 'a') as f:
            f.write(f"\n## {structure_data['id']}\n")
            f.write(f"- Condition: {condition}\n")
            f.write(f"- Turns: {game_result['turns_taken']}\n")
            f.write(f"- Progress: {game_result.get('final_progress', 0):.3f}\n")
            f.write(f"- Completed: {game_result.get('completed', False)}\n")
            if game_result.get('selection_summary'):
                f.write(f"- Selection: {json.dumps(game_result['selection_summary'])}\n")

        correctness.index.name = 'Transcript'
        print(correctness)
        correctness.to_csv(main_grice_correctness_filename)

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nDone. Results: {json_path}")
    return all_results, lastPartType


# ─────────────────────────────────────────
# SINGLE GAME
# ─────────────────────────────────────────

def run_single_game(
    structure_data,
    director_agents,
    builder_agent,
    # chair_agent,
    common_ground_agent,

    correctness,
    max_turns=15,
    structure_idx=0,
    use_common_ground=False,
    run = 1,
    lastPartType = None,
    use_tools = True,
    use_oracle=False,
    num_oracle = None,
    condition = "baseline",
    policy_selector = None,
    random_seed = 0,

):
    requestCount = 0
    start = time.time()
    target_structure    = structure_data['structure']
    target_spans        = structure_data['spans']          # already int keys
    target_director_views = structure_data['director_views'] #possible inconsistency: use current live target views from game state:  target_director_views = get_director_views_fn(target_structure, spans=target_spans)
    #target_director_views = get_director_views_fn(target_structure, spans=target_spans)
    # ── Init game state ───────────────────────────────────────
    partComplete = True
    game_state = EnhancedGameState(
        target_structure=target_structure,
        target_spans=target_spans,
        partComplete=partComplete,
        partType = lastPartType
    )

    # ── Sanity check: stored views vs recomputed ──────────────
    tmp = copy.deepcopy(game_state.current_structure)
    game_state.current_structure = copy.deepcopy(target_structure)
    recomputed = game_state.get_director_views()
    game_state.current_structure = tmp

    # compare against RAW views (before swap) not swapped_dir_views
    raw_views = {
        d: {row: items[::-1] for row, items in rows.items()}
        if d in ["D2", "D3"] else rows
        for d, rows in recomputed.items()
    }
    compare_views("TARGET VIEW CONSISTENCY BEFORE GAME STARTS", target_director_views, raw_views)


    conversation_history = []

    game_results = {
        'structure_id'        : structure_data['id'],
        'structure_idx'       : structure_idx,
        'complexity'          : structure_data['complexity'],
        'target_structure'    : target_structure,
        'target_spans'        : {str(k): v for k, v in target_spans.items()},
        'target_director_views': target_director_views,
        'completed'           : False,
        'turns_taken'         : 0,
        'final_progress'      : 0.0,
        'final_structure'     : {},
        'stopping_reason'     : 'max_turns_reached',
        'partialCompletion'   : partComplete,
        'condition'           : condition,
    }

    if(partComplete):
        lastPartType = game_state.partType
        game_results['partialCompletionCategory'] = game_state.partType

    for d in director_agents.values():
        game_results[d.director_id + " Archetype"] = d.archetype

    game_results['turns'] = []

    # Per-game counts of what each selection method picked.
    selection_counts = {
        'turns_selected': 0,
        'policy_picks': {d: 0 for d in DIRECTOR_IDS},
        'random_picks': {d: 0 for d in DIRECTOR_IDS},
        'policy_equals_random': 0,
        'policy_argmax_was_silent': 0,
        'builder_heard_single_director': 0,
        'no_director_spoke': 0,
    }
    # Builder replies that needed a format re-prompt, and ones that never parsed.
    parse_counts = {
        'builder_turns': 0,
        'turns_with_retry': 0,
        'retries_total': 0,
        'unparsed_after_retries': 0,
        'model_clarify': 0,
        'rejected_replies': 0,
        'rejected_by_reason': {},       # empty | no_move_line | parse_error | other
        'rejected_hit_output_cap': 0,   # rejected replies that stopped at the output cap
    }

    # ── MAIN LOOP ─────────────────────────────────────────────
    for turn in range(max_turns):
        print(f"\n--- Turn {turn+1}/{max_turns} [condition={condition}] ---")
        game_state.increment_turn()
        cValues = {}
        cValues['Turn']=(turn + 1)
        turn_data = {
            'turn_number'     : turn + 1,
            'timestamp'       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'structure_before': copy.deepcopy(game_state.current_structure),
            'spans_before'    : copy.deepcopy(game_state.current_spans),
        }

        try:
            # ── Directors ─────────────────────────────────────
            # Full board state for directors to reason with
            full_board_state = game_state.get_director_views()

            mism, total = view_distance(full_board_state, target_director_views)
            print(f"  View mismatches vs target: {mism}/{total}")

            public_conversation = "\n".join(conversation_history)
            director_order = random.choices(["D1", "D2", "D3"], k=3)
            random.shuffle(director_order)

            director_responses = {}
            for did in director_order:
                print(f"  {did} responding...")

                director_prompt = director_agents[did].get_director_prompt(
                    current_view=full_board_state,           # full board, all layers
                    target_view=target_director_views[did],  # partial — director's perspective only
                    conversation_history=public_conversation,
                    available_blocks=game_state.available_blocks
                )
                resp = director_agents[did].generate_response(
                    current_view=full_board_state,           # full board, all layers
                    target_view=target_director_views[did],  # partial — director's perspective only
                    conversation_history=public_conversation,
                    available_blocks=game_state.available_blocks
                )
                requestCount += 1

                if(resp["internal_thinking"].__contains__("429")):
                    end = time.time()
                    length = end - start

                    print("\n\n####################################")
                    print(resp["internal_thinking"])
                    print("Time: " + str(length))
                    print("Requests: " + str(requestCount))
                    print("####################################\n\n")
                    sys.exit(1)

                director_responses[did] = resp
                conversation_history.append(f"{did}: {resp['public_message']}")

                print(f"  {did}: {resp['public_message'][:80]}")
                turn_data[f'director_prompt_{did}'] = director_prompt
                # print(f"  {did} INTENT: {intent}")
            turn_data['director_responses'] = director_responses

            def is_validResponse(val):
                return val[1]['public_message'] != "No message provided"

            # ── Builder ───────────────────────────────────────
            director_discussion_full = "\n".join(
                f"{did}: {r['public_message']}"
                for did, r in filter(is_validResponse, director_responses.items())
            )

            # ── Utterance selection ───────────────────────────
            # director_order samples with replacement, so a director can be
            # missing this turn; missing and silent directors can't be picked.
            messages = {d: (director_responses[d]['public_message'] if d in director_responses else "")
                        for d in DIRECTOR_IDS}
            speaking = [d for d in DIRECTOR_IDS if not is_silent_message(messages[d])]

            selection = {
                'condition': condition,
                'speaking': speaking,
                'director_order': director_order,
                'messages': messages,
                'baseline': {'method': 'baseline', 'directors': speaking,
                             'utterance': director_discussion_full},
                'policy': None,
                'random': None,
            }

            policy_choice = None
            if policy_selector is not None:
                chosen, raw, scores = policy_selector.score_turn(
                    turn + 1, turn_data['structure_before'], messages, speaking)
                if chosen is not None:
                    policy_choice = chosen
                    selection['policy'] = {
                        'method': 'policy', 'choice': chosen, 'raw_argmax': raw,
                        'argmax_was_silent': raw != chosen, 'scores': scores,
                        'utterance': f"{chosen}: {messages[chosen]}",
                    }
                    selection_counts['policy_picks'][chosen] += 1
                    selection_counts['policy_argmax_was_silent'] += int(raw != chosen)
                    print(f"  [POLICY] scores "
                          + " ".join(f"{d}={scores[d]:+.4f}" for d in DIRECTOR_IDS)
                          + f" -> {chosen}" + (f" (raw argmax {raw} silent)" if raw != chosen else ""))

            random_choice = random_pick(random_seed, structure_data['id'], run, turn + 1, pool=speaking)
            if random_choice is not None:
                selection['random'] = {
                    'method': 'random', 'choice': random_choice, 'pool': speaking,
                    'p_choice': round(1.0 / len(speaking), 6),
                    'seed_key': [random_seed, structure_data['id'], run, turn + 1],
                    'utterance': f"{random_choice}: {messages[random_choice]}",
                }
                selection_counts['random_picks'][random_choice] += 1
                print(f"  [RANDOM] draw from {speaking} -> {random_choice}")

            if policy_choice is not None and random_choice is not None:
                selection_counts['policy_equals_random'] += int(policy_choice == random_choice)

            # What the Builder actually hears this turn.
            if condition == "policy" and policy_choice is not None:
                builder_discussion = selection['policy']['utterance']
                builder_heard = [policy_choice]
                used_method = 'policy'
            elif condition == "random" and random_choice is not None:
                builder_discussion = selection['random']['utterance']
                builder_heard = [random_choice]
                used_method = 'random'
            else:
                # 'baseline', or no director spoke (nothing to select from)
                builder_discussion = director_discussion_full
                builder_heard = list(speaking)
                used_method = 'baseline'
            if not speaking:
                selection_counts['no_director_spoke'] += 1
            selection_counts['turns_selected'] += 1
            selection_counts['builder_heard_single_director'] += int(used_method != 'baseline')

            selection['used_method'] = used_method
            selection['builder_heard'] = builder_heard
            selection['builder_input'] = builder_discussion
            turn_data['utterance_selection'] = selection
            turn_data['director_discussion_used'] = builder_discussion
            print(f"  [SELECTION] condition={condition} used={used_method} builder hears {builder_heard}")

            oracle_moves = None
            if use_oracle:
                rng = random.Random(structure_idx * 1000 +  (turn + 1))

                oracle_moves = get_oracle_moves(game_state, n=num_oracle, rng=rng)
                print(f"  [ORACLE] {len(oracle_moves)} correct moves available this turn with num oracle:{num_oracle}:")
                for m in oracle_moves:
                    span_str = f" → {m['span_to']}" if m.get('span_to') else ""
                    print(f"    {m['action'].upper()} {m.get('block','')} "
                        f"@ {m['position']} layer {m['layer']}{span_str}")

            builder_prompt = builder_agent.get_builder_prompt(
                director_discussion=builder_discussion,
                current_state=game_state.current_structure,
                available_blocks=game_state.available_blocks,
                use_tools=use_tools,
                oracle_moves=oracle_moves,
                builderPromptAddin = BUILDER_PROMPT
            )

            if use_tools:
                builder_move = builder_agent.generate_move_with_tools(
                director_discussion=builder_discussion,
                game_state=game_state,  # pass state so tool can simulate
                oracle_moves=oracle_moves,
                check_prompt_tokens=True  # ← only when needed
            )
            else:
                builder_move = builder_agent.generate_move(
                director_discussion=builder_discussion,
                current_state=game_state.current_structure,  # pass state so tool can simulate
                available_blocks=game_state.available_blocks,
                oracle_moves=oracle_moves,
            )

            print(f"  Builder move: {builder_move}")

            # Parse retries (see BuilderAgent._retry_unparsed): a reply that
            # couldn't be parsed is re-prompted up to --builder_parse_retries
            # times; only if it still fails is the turn recorded as a CLARIFY.
            attempts = getattr(builder_agent, 'last_attempts', 1) or 1
            parse_failed = bool(getattr(builder_agent, 'last_parse_failed', False))
            turn_data['builder_attempts'] = attempts
            turn_data['builder_parse_failed'] = parse_failed
            turn_data['builder_raw_text'] = getattr(builder_agent, 'last_raw_text', None)
            # Every rejected reply this turn: attempt, reason, hit_output_cap, finish_reason, text.
            rejected = list(getattr(builder_agent, 'last_rejected', []) or [])
            turn_data['builder_rejected_replies'] = rejected
            parse_counts['rejected_replies'] += len(rejected)
            for rec in rejected:
                parse_counts['rejected_by_reason'][rec['reason']] = \
                    parse_counts['rejected_by_reason'].get(rec['reason'], 0) + 1
                parse_counts['rejected_hit_output_cap'] += int(rec['hit_output_cap'])
            parse_counts['builder_turns'] += 1
            parse_counts['turns_with_retry'] += int(attempts > 1)
            parse_counts['retries_total'] += attempts - 1
            parse_counts['unparsed_after_retries'] += int(parse_failed)
            parse_counts['model_clarify'] += int(builder_move.get('action') == 'clarify' and not parse_failed)
            cValues['Transcription'] = director_discussion_full
            cValues['Condition'] = condition
            cValues['Selection Method'] = used_method
            cValues['Builder Heard'] = ",".join(builder_heard)
            cValues['Policy Choice'] = policy_choice if policy_choice else "N/A"
            cValues['Random Choice'] = random_choice if random_choice else "N/A"
            cValues['Builder Attempts'] = attempts
            cValues['Parse Failed'] = parse_failed
            cValues['Rejected Reasons'] = ",".join(
                r['reason'] + ("+cap" if r['hit_output_cap'] else "") for r in rejected) or "N/A"
            cValues['Failed Move']="N/A"
            cValues['Structure Placement']="N/A"
            cValues['Side Placement']="N/A"
            cValues['Overall Structure State']="N/A"
            cValues['Group Agreement']='N/A'
            #requestCount += 1

            # ── Execute ───────────────────────────────────────
            move_failed = False
            if builder_move['action'] == 'clarify':
                conversation_history.append(f"Builder: {builder_move['clarification']}")
                turn_data['move_executed'] = False
                turn_data['clarification'] = builder_move['clarification']
                turn_data['builder_prompt'] = builder_prompt
                cValues['Clarify'] = True

            else:
                cValues['Clarify'] = "N/A"
                if 'confirmation' in builder_move:
                    conversation_history.append(f"Builder: {builder_move['confirmation']}")
                    turn_data['builder_confirmation'] = builder_move.get('confirmation', '')
                    turn_data['builder_move_raw'] = copy.deepcopy(builder_move)

                success, progress_data, structurePlacement, sidePlacement, overall = game_state.execute_move(builder_move)
                move_failed = not success

                if use_oracle and oracle_moves:
                    attempted = builder_move
                    match = any(
                        attempted.get('action') == m['action'] and
                        attempted.get('position') == m['position'] and
                        attempted.get('layer') == m['layer']
                        for m in oracle_moves
                    )
                    print(f"  [ORACLE CHECK] builder chose from oracle list: {match}")

                    turn_data['oracle_moves'] = [
                        {k: v for k, v in m.items()} for m in oracle_moves
                    ] if oracle_moves else []

                    turn_data['builder_followed_oracle'] = any(
                        builder_move.get('action') == m['action'] and
                        builder_move.get('position') == m['position'] and
                        builder_move.get('layer') == m['layer']
                        for m in oracle_moves
                    ) if oracle_moves else None


                turn_data['builder_prompt'] = builder_prompt
                turn_data['move_attempted'] = copy.deepcopy(builder_move)
                turn_data['move_executed']  = success
                turn_data['progress_data']  = progress_data

                cValues['Failed Move']=(not success)
                turn_data['failed_move']=(not success)
                if(success):
                    turn_data['correct_structure_placement']=structurePlacement
                    turn_data['correct_side_placement']=sidePlacement
                    turn_data['overall_structure_correctness']=overall
                    cValues['Structure Placement']=(structurePlacement)
                    cValues['Side Placement']=(sidePlacement)
                    cValues['Overall Structure State']=(overall)
                cValues['Group Agreement']='N/A'
                turn_data['director_discussion_full'] = director_discussion_full
                # turn_data['director_discussion_used'] is set in the selection block above

                if success:
                    print(f"  Move OK. Progress: {progress_data['metrics']['overall_progress']:.3f}")
                    conversation_history.append(
                        f"Builder: Placed {builder_move['block']} at {builder_move['position']}"
                        + (f"↔{builder_move['span_to']}" if builder_move.get('span_to') else "")
                        + f" layer {builder_move['layer']}. "
                        f"Current board: {json.dumps(game_state.current_structure)}"
                    )
                    if use_common_ground and common_ground_agent is not None:
                        print("Analyzing common ground...")
                        common_ground_result = common_ground_agent.generate_common_ground(
                            director_responses=director_responses,
                            current_board_state=game_state.current_structure,
                            conversation_history="\n".join(conversation_history),
                            last_move=progress_data
                        )
                        turn_data['common_ground'] = common_ground_result
                        agreement = common_ground_result.get('agreement', "No")
                        cValues['Group Agreement'] = agreement
                    else:
                        cValues['Group Agreement'] = "Disabled"
                if not success:
                    err = progress_data.get('error', 'Unknown')
                    msg = f"Move failed: {err}. Please give different instructions."
                    #removing board state as well after builder failed move. too much info in context.
                    conversation_history.append(
                    f"Builder: FAILED — {err}. Board unchanged: {json.dumps(game_state.current_structure)}"
                )
                    print(f"  Move FAILED: {err}")

            # Add this turn to the policy's per-game history (after-board and
            # which director the Builder heard alone, if any).
            if policy_selector is not None:
                policy_selector.commit(
                    followed=builder_heard[0] if used_method != 'baseline' else None,
                    board_after=game_state.current_structure,
                    move_failed=move_failed,
                )

            # ── Completion check ──────────────────────────────
            if game_state.is_complete():
                print("  Target achieved!")
                turn_data['target_achieved'] = True
                game_results['stopping_reason'] = 'target_achieved'
                game_results['turns'].append(turn_data)
                break
            else:
                turn_data['target_achieved'] = False
            turn_data['conversation_snapshot'] = list(conversation_history)
            # Trim conversation history
            # conversation_history = conversation_history[-8:] #harsher trimming
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-40:]

            game_results['turns'].append(turn_data)

            # New row data as a single-row DataFrame
            new_row_df = pd.DataFrame([cValues])

            # Concatenate the original and new DataFrames
            correctness = pd.concat([correctness, new_row_df], ignore_index=True)
            print(correctness)

        except Exception as e:
            print(f"  Error turn {turn+1}: {e}")
            turn_data['error'] = str(e)
            game_results['turns'].append(turn_data)
            break

    # ── Finalize ──────────────────────────────────────────────
    game_results['turns_taken']     = len(game_results['turns'])
    game_results['final_structure'] = copy.deepcopy(game_state.current_structure)
    game_results['final_spans']     = copy.deepcopy(game_state.current_spans)
    game_results['selection_summary'] = selection_counts
    game_results['builder_parse_summary'] = parse_counts
    if parse_counts['turns_with_retry'] or parse_counts['unparsed_after_retries']:
        print(f"  [builder] parse retries this game: {parse_counts['retries_total']} over "
              f"{parse_counts['turns_with_retry']} turn(s); still unparsed after retries: "
              f"{parse_counts['unparsed_after_retries']}; rejected replies by reason: "
              f"{parse_counts['rejected_by_reason']} ({parse_counts['rejected_hit_output_cap']} hit the output cap)")

    if game_state.progress_tracker.progress_history:
        last = game_state.progress_tracker.progress_history[-1]
        game_results['final_progress'] = last.get('metrics', {}).get('overall_progress', 0.0)
        game_results['completed']      = game_results['final_progress'] >= 0.95

    game_results['progress_summary'] = game_state.get_progress_summary()

    print(f"\nGame done [{condition}]: {game_results['turns_taken']} turns, "
          f"progress={game_results['final_progress']:.3f}, "
          f"completed={game_results['completed']}")

    return game_results, correctness, lastPartType

if __name__ == "__main__":
    import argparse
    load_dotenv()

    parser = argparse.ArgumentParser(description="CRAFT Runner")
    parser.add_argument("--mode",           type=str, default="local",
                        choices=["api", "local"],
                        help="Director mode: 'api' for frontier models, 'local' for open-weight")
    parser.add_argument("--director",       type=str, default=None,
                        help="Specific director model to run (api: model name, local: key from LOCAL_MODELS)")
    parser.add_argument("--builder",        type=str, default="gpt-4o-mini",
                        help="Builder model name")
    parser.add_argument("--builderPrompt",  type=str, default="Base",
                            help="Builder prompt add in name")
    parser.add_argument("--dataset",        type=str, default="/home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json",
                        help="Path to structures dataset JSON")
    parser.add_argument("--output",         type=str, default=None,
                        help="Output directory (default: auto-generated from builder model)")
    parser.add_argument("--turns",          type=int, default=20,
                        help="Max turns per game")
    parser.add_argument("--run",            type=int, default=3,
                        help="Run index for deterministic seeding")
    parser.add_argument("--oracle",         action="store_true",
                        help="Enable oracle candidate moves for builder")
    parser.add_argument("--oracle_n",       type=int, default=5,
                        help="Number of oracle moves to show per turn")
    parser.add_argument("--no_tools",       action="store_true",
                        help="Disable builder tool use (simulate_move)")
    parser.add_argument("--builder_parse_retries", type=int, default=2,
                        help="Re-prompt the Builder for the PLACE/REMOVE/CLARIFY format up to N times "
                             "when its reply can't be parsed, before the turn counts as a CLARIFY "
                             "(same as the rollout harness). 0 = old behaviour.")
    parser.add_argument("--builder_max_tokens", type=int, default=None,
                        help="Output cap for the Builder's move call and its retries. Default: the "
                             "rollout harness's values (400; 4096 for Claude). 250 = old full-game value.")
    parser.add_argument("--builder_no_one_line_instruction", action="store_true",
                        help="Leave 'Output EXACTLY ONE line...' out of the Builder's system prompt "
                             "(the old full-game prompt). By default it is included, as in the rollout harness.")
    parser.add_argument("--structures",     type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated structure indices to run (e.g. '0,1,5'). Default: all")
    parser.add_argument("--quantize",       type=str, default=None,
                        choices=["4bit", "8bit"],
                        help="Quantization for local models (qwen-72b/32b default 4bit)")
    parser.add_argument("--max_tokens",     type=int, default=None,
                        help="Override max output tokens for director (overrides per-model defaults)")
    # ── Utterance-selection conditions ────────────────────────
    parser.add_argument("--conditions",     type=str, default="baseline",
                        help="Comma-separated conditions, each run as its own full game per structure: "
                             "'baseline' (Builder hears every director, the original game), 'policy' "
                             "(Builder hears only the policy's pick), 'random' (Builder hears one random "
                             "director). E.g. 'baseline,policy,random'. 'all' is accepted as an old name "
                             "for 'baseline'. Default: baseline")
    parser.add_argument("--policy_path",    type=str, default=None,
                        help="Trained index policy (.pt). Required for the 'policy' condition; if given, "
                             "the policy's pick is also logged (not used) in the other conditions.")
    parser.add_argument("--policy_code_dir", type=str, default=None,
                        help="Folder with the policy project's config.py/features.py/models.py/arm_state.py. "
                             "Default: searched next to --policy_path.")
    parser.add_argument("--policy_feature_mode", type=str, default=None,
                        help="Override FEATURE_MODE for the policy (must match training).")
    parser.add_argument("--policy_feature_tier", type=str, default=None,
                        help="Override FEATURE_TIER for the policy (must match training).")
    parser.add_argument("--policy_allow_silent", action="store_true",
                        help="Let the policy pick a silent director (default: best-scoring director who spoke).")
    parser.add_argument("--random_seed",    type=int, default=0,
                        help="Seed for the random condition's draws (combined with structure, run and turn).")
    parser.add_argument("--token_estimate", action="store_true",
                        help="Estimate API token usage instead of calling any model: every builder and "
                             "director call is answered by the local Qwen model in token_estimator.py "
                             "and its tokens are recorded. Needs token_estimator.py next to this script.")
    parser.add_argument("--estimate_max_tokens", type=int, default=512,
                        help="(--token_estimate) Output cap for calls that don't set max_tokens themselves.")
    parser.add_argument("--estimate_backend", choices=["transformers", "ollama"], default="transformers",
                        help="(--token_estimate) 'transformers' (default): Qwen2.5-14B loaded locally, "
                             "~30 GB GPU. 'ollama': the model in --estimate_model served by Ollama "
                             "(ollama_token_estimator.py next to this script).")
    parser.add_argument("--estimate_model_path", type=str, default=None,
                        help="(--token_estimate) Local folder (or Hugging Face id) for the estimator's "
                             "Qwen2.5-14B-Instruct, instead of downloading it to the Hugging Face "
                             "cache. With --estimate_backend ollama, used for the tokenizer only.")
    parser.add_argument("--estimate_model", type=str, default="qwen2.5:14b-instruct",
                        help="(--estimate_backend ollama) Ollama model that answers the calls.")
    parser.add_argument("--estimate_num_ctx", type=int, default=16384,
                        help="(--estimate_backend ollama) Starting context size; grows automatically "
                             "if a prompt needs more, so prompts are never cut.")
    args = parser.parse_args()

    if args.token_estimate:
        # Agents build their real clients before the estimator replaces them,
        # and some SDKs refuse to construct without a key. Placeholder keys
        # only fill missing ones; with the estimator no request is ever sent.
        for k in ESTIMATE_DUMMY_KEYS:
            os.environ.setdefault(k, "token-estimate")

    api_key = os.getenv('OPENAI_API_KEY')

    # ── Config ────────────────────────────────────────────────
    DIRECTOR_MODE = args.mode
    BUILDER_MODEL = args.builder
    BUILDER_PROMPT = BuilderType(args.builderPrompt)
    DATASET_PATH  = args.dataset
    MAX_TURNS     = args.turns
    RUN           = args.run
    USE_ORACLE    = args.oracle
    ORACLE_N      = args.oracle_n
    USE_TOOLS     = not args.no_tools
    # ── Auto-generate output dir from config ──────────────────
    oracle_tag = f"oracle{ORACLE_N}" if USE_ORACLE else "no_oracle"
    tools_tag  = "tools" if USE_TOOLS else "no_tools"
    run_tag    = f"run{RUN}"
    print(args)

    BASE_OUTPUT_DIR = args.output or (
        f"craft_results/"
        f"{DIRECTOR_MODE}/"
        f"{oracle_tag}_{tools_tag}_{run_tag}")

    # ── Conditions and policy ─────────────────────────────────
    CONDITIONS_TO_RUN = [CONDITION_ALIASES.get(c.strip(), c.strip())
                         for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in CONDITIONS_TO_RUN if c not in CONDITIONS]
    if bad:
        raise ValueError(f"Unknown condition(s) {bad}. Choose from {list(CONDITIONS)}")
    if "policy" in CONDITIONS_TO_RUN and not args.policy_path:
        raise ValueError("The 'policy' condition needs --policy_path.")

    POLICY_SCORER = None
    if args.policy_path:
        POLICY_SCORER = load_policy_scorer(
            args.policy_path, code_dir=args.policy_code_dir,
            feature_mode=args.policy_feature_mode, feature_tier=args.policy_feature_tier,
            allow_silent=args.policy_allow_silent,
        )

    def output_dir_for(condition):
        """'baseline' without --output keeps the original folder, so existing runs line up."""
        tag = condition_tag(condition, args.policy_path, args.random_seed)
        if args.output:
            return os.path.join(args.output, tag)
        out = BASE_OUTPUT_DIR if condition == "baseline" else f"{BASE_OUTPUT_DIR}_{tag}"
        # Estimate runs never write into real result folders.
        return f"{out}_token_estimate" if args.token_estimate else out

    EST = None
    if args.token_estimate:
        try:
            from token_estimator import TokenEstimator, EstimatorClient
        except Exception as e:
            raise SystemExit(f"[estimate] Could not import token_estimator.py ({e}). "
                             f"Put it next to run_craft.py.")
        if args.estimate_model_path:
            # TokenEstimator reads token_estimator.MODEL when it is created, so
            # pointing it at a local copy avoids the download entirely.
            import token_estimator as _te
            _te.MODEL = args.estimate_model_path
            print(f"[estimate] estimator model/tokenizer from: {args.estimate_model_path}")
        if args.estimate_backend == "ollama":
            try:
                import ollama_token_estimator as _ote
                if args.estimate_model_path:
                    _ote.TOKENIZER_MODEL = args.estimate_model_path
                from ollama_token_estimator import OllamaTokenEstimator
            except Exception as e:
                raise SystemExit(f"[estimate] Could not import ollama_token_estimator.py ({e}). "
                                 f"Put it next to run_craft.py.")
            EST = OllamaTokenEstimator(model=args.estimate_model, max_tokens=args.estimate_max_tokens,
                                       num_ctx=args.estimate_num_ctx)
        else:
            print("[estimate] Loading the token estimator (one model for the whole run)...")
            EST = TokenEstimator(max_tokens=args.estimate_max_tokens)
        # Gemini directors create google.genai.Client() inside every call.
        try:
            from google import genai as _genai
            _genai.Client = lambda *a, **k: make_estimator_client(EST, "director")
        except ImportError:
            pass
        if DIRECTOR_MODE == "local":
            print("[estimate] NOTE: --mode local directors run through transformers and are not "
                  "estimated; only the builder's calls will be counted. Use --mode api with the "
                  "director model you plan to pay for.")
        if USE_TOOLS:
            print("[estimate] NOTE: the estimator can't make simulate_move tool calls, so each "
                  "builder turn is estimated as a single call. A real tool-using builder makes "
                  "up to 3 extra round trips per turn, each resending the prompt, so builder "
                  "tokens are a lower bound. Run with --no_tools for an exact like-for-like count.")
    ESTIMATE_DIR = (args.output or f"{BASE_OUTPUT_DIR}_token_estimate") if args.token_estimate else None

    print(f"Conditions: {CONDITIONS_TO_RUN}  ->  "
          + ", ".join(f"{c}: {output_dir_for(c)}" for c in CONDITIONS_TO_RUN))

    LOCAL_MODELS = {
        "mistral-7b":       "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen-7b":          "Qwen/Qwen2.5-7B-Instruct",
        #"llama-8b":         "meta-llama/Llama-3.1-8B-Instruct",
        #"qwen-14b":         "Qwen/Qwen2.5-14B-Instruct",
        #"qwen-32b":         "Qwen/Qwen2.5-32B-Instruct",
        #"gemma-9b":         "google/gemma-2-9b-it",
        #"deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        #"qwen-72b":         "Qwen/Qwen2.5-72B-Instruct",
    }

    # LOCAL_MODELS = {
    #     # "qwen-7b":          "/data/open-weight-llms/models/qwen-7b",
    #     # "qwen7b_dpo_r32_from_sft" : "/data/open-weight-llms/models/qwen-7b",
    #     # "qwen7b_sft_r32_builder" :"/data/open-weight-llms/models/qwen-7b",
    #     "qwen7b_ipo_r32_from_sft" : "/data/open-weight-llms/models/qwen-7b",

    #     # "qwen-14b":         "/data/open-weight-llms/models/qwen-14b",
    #     # "qwen-32b":         "/data/open-weight-llms/models/qwen-32b",

    #     # "llama-8b":         "/data/open-weight-llms/models/llama-8b",
    #     # "llama-8b_dpo_r32_from_sft":  "/data/open-weight-llms/models/llama-8b",
    #     # "llama-8b_r32_baseline": "/data/open-weight-llms/models/llama-8b",
    #     "llama8b_ipo_r32_from_sft" : "/data/open-weight-llms/models/llama-8b"

    #     # "mistral-7b":       "/data/open-weight-llms/models/mistral-7b",
    #     # "gemma-9b":         "/data/open-weight-llms/models/gemma-9b",
    #     # "deepseek-v2-lite": "/data/open-weight-llms/models/deepseek-v2-lite",
    #     # "qwen-72b":         "/data/open-weight-llms/models/qwen-72b",
    # }

    API_DIRECTOR_MODELS = [
        "gemini-3-flash-preview",
        "gpt-4o",
        #"gpt-5.4-mini",
        "gpt-4o-mini",
        #"gpt-4.1-mini",
        #"gemini-2.5-flash",
        #"gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
        #"claude-haiku-4-5",
        #"claude-sonnet-4-6",
    ]

    MAX_TOKENS_BY_MODEL = {
        "gpt-5":                          2000,
        "gpt-4o-mini":                    2000,
        "gpt-4.1-mini":                   2000,
        "gpt-4o":                         2000,
        "claude-haiku-4-5":               3000,
        "claude-haiku-5-5":               3000,
        "claude-sonnet-4-6":              3000,
        "gemini-2.5-flash":               3000,
        "gemini-2.5-flash-lite":          2000,
        "gemini-3-flash-preview":         2000,
        "gemini-3.1-flash-lite-preview":  2000,
    }
    DEFAULT_MAX_TOKENS = 500

    # ── Structure selection ───────────────────────────────────
    with open(DATASET_PATH, "r") as f:
        n_structures = len(json.load(f))
    print(f"Total structures: {n_structures}")

    if args.structures:
        structure_indices = [int(i) for i in args.structures.split(",")]
    else:
        structure_indices = list(range(n_structures))
    print(f"Running structures: {structure_indices}")

    # ── Model selection ───────────────────────────────────────
    if args.director:
        if DIRECTOR_MODE == "api":
            models_to_run_api = [args.director]
            models_to_run_local = {}
        else:
            if args.director not in LOCAL_MODELS:
                raise ValueError(f"Unknown local model key: {args.director}. Choose from {list(LOCAL_MODELS.keys())}")
            models_to_run_local = {args.director: LOCAL_MODELS[args.director]}
            models_to_run_api = []
    else:
        models_to_run_api = API_DIRECTOR_MODELS
        models_to_run_local = LOCAL_MODELS

    # def get_part_type_for_structure(structure_index, run):
    #     seed = hash((structure_index, run)) % (2**32)
    #     rng  = random.Random(seed)
    #     return rng.choice(EnhancedGameState.PARTIAL_OPTIONS)

    # print("\nPartType sequence for this run:")
    # for i in structure_indices:
    #     pt = get_part_type_for_structure(i, RUN)
    #     print(f"  structure_{i:02d} → {pt}")

    def run_all_structures(director_model_name, condition, shared_model=None, shared_tokenizer=None):
        output_dir = output_dir_for(condition)
        first_call = len(EST.calls) if EST is not None else 0
        for structure_index in structure_indices:
            partType = "empty"
            print(f"\n  [{director_model_name} | {condition}] structure={structure_index}/{n_structures-1} "
                  f"partType={partType}")
            try:
                results, _ = run_craft_experiments(
                    dataset_path=DATASET_PATH,
                    structure_index=structure_index,
                    director_model_name=director_model_name,
                    builder_model_name=BUILDER_MODEL,
                    api_key=api_key,
                    max_turns=MAX_TURNS,
                    output_dir=output_dir,
                    use_common_ground=False,
                    run=RUN,
                    lastPartType=partType,
                    shared_model=shared_model,
                    shared_tokenizer=shared_tokenizer,
                    builder_tool_use=USE_TOOLS,
                    use_oracle=USE_ORACLE,
                    num_oracle=ORACLE_N,
                    max_tokens=args.max_tokens if args.max_tokens else MAX_TOKENS_BY_MODEL.get(director_model_name, DEFAULT_MAX_TOKENS),
                    condition=condition,
                    policy_scorer=POLICY_SCORER,
                    policy_path=args.policy_path,
                    random_seed=args.random_seed,
                    token_estimator=EST,
                    builder_parse_retries=args.builder_parse_retries,
                    builder_max_tokens=args.builder_max_tokens,
                    builder_one_line_instruction=not args.builder_no_one_line_instruction,
                )
                print(f"  Done [{director_model_name} | {condition}] structure={structure_index} "
                      f"games={len(results['games'])}")
            except Exception as e:
                import traceback
                print(f"  FAILED [{director_model_name} | {condition}] structure={structure_index}: {e}")
                traceback.print_exc()
                continue
        if EST is not None:
            for c in EST.calls[first_call:]:
                c["director_model"] = director_model_name
                c["condition"] = condition

    # ── Run ───────────────────────────────────────────────────
    if DIRECTOR_MODE == "api":
        for model_name in models_to_run_api:
            print(f"\n{'='*60}\nAPI DIRECTOR: {model_name}\n{'='*60}")
            for condition in CONDITIONS_TO_RUN:
                print(f"\n{'-'*60}\nCONDITION: {condition}\n{'-'*60}")
                run_all_structures(director_model_name=model_name, condition=condition)

    elif DIRECTOR_MODE == "local":
        for model_name, model_path in models_to_run_local.items():
            print(f"\n{'='*60}\nLOADING MODEL: {model_name}\n{'='*60}")
            try:
                shared_model, shared_tokenizer = load_local_director_pipeline(
                    model_path,
                    quantize=args.quantize or ("4bit" if model_name in ("qwen-72b", "qwen-32b") else None),
                )
                print(f"Loaded: {shared_model}")
            except Exception as e:
                print(f"FAILED to load {model_name}: {e}")
                continue

            # All conditions reuse the loaded model, so it is loaded only once.
            for condition in CONDITIONS_TO_RUN:
                print(f"\n{'-'*60}\nCONDITION: {condition}\n{'-'*60}")
                run_all_structures(
                    director_model_name=model_name,
                    condition=condition,
                    shared_model=shared_model,
                    shared_tokenizer=shared_tokenizer,
                )

            print(f"\nReleasing {model_name} from memory...")
            try:
                del shared_model, shared_tokenizer
            except Exception:
                pass
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU memory released.\n")

    if EST is not None:
        estimate = summarize_estimate(EST)
        estimate["sample"] = {
            "director_models": list(models_to_run_api) or list(models_to_run_local),
            "builder": BUILDER_MODEL, "conditions": CONDITIONS_TO_RUN,
            "structures": structure_indices, "max_turns": MAX_TURNS, "builder_tools": USE_TOOLS,
            "estimate_backend": args.estimate_backend,
            "estimate_model": args.estimate_model if args.estimate_backend == "ollama" else None,
        }
        os.makedirs(ESTIMATE_DIR, exist_ok=True)
        est_path = os.path.join(ESTIMATE_DIR, "token_estimate.json")
        with open(est_path, "w") as f:
            json.dump(estimate, f, indent=2)
        print("\n" + "=" * 60 + "\nTOKEN ESTIMATE (tokens already scaled by CORRECTION)\n" + "=" * 60)
        print(json.dumps(estimate, indent=2))
        print(f"\nSample: {len(structure_indices)} structure(s) x {len(CONDITIONS_TO_RUN)} condition(s) "
              f"x {len(estimate['sample']['director_models'])} director model(s), up to {MAX_TURNS} turns. "
              f"Scale by (full run size / sample size). Saved to {est_path}")

    print("All done.")