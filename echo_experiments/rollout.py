"""Multi-turn CRAFT episode rollout: three frozen directors and one builder per turn."""
import concurrent.futures
import copy
import random
import sys
from pathlib import Path

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from agents.builder_agent import BuilderAgent
from agents.director_agent import DirectorAgent
from agents.environment import EnhancedGameState, get_oracle_moves
from agents.oracle import enumerate_correct_actions, FLAG_OK

from reward import compute_builder_reward

# used only for prompt building and parsing
_BUILDER_HELPER = BuilderAgent(api_key="unused-local-generation-only")


def build_frozen_directors(
    structure_index, run_id, director_model_name="gpt-4.1-mini", api_key=None,
    director_mode="api", shared_director_model=None, shared_director_tokenizer=None,
):
    """D1-D3 director agents. director_mode="local" shares one local model across all three."""
    use_api = director_mode != "local"
    return {
        did: DirectorAgent(
            director_id=did,
            use_api=use_api,
            api_key=api_key if use_api else None,
            model_name=director_model_name,
            local_model=None if use_api else shared_director_model,
            local_tokenizer=None if use_api else shared_director_tokenizer,
            structure_index=structure_index,
            run=run_id,
        )
        for did in ["D1", "D2", "D3"]
    }


def director_discussion_text(director_responses):
    return "\n".join(
        f"{did}: {r['public_message']}"
        for did, r in director_responses.items()
        if r["public_message"] != "No message provided"
    )


def sample_oracle_candidates(game_state, n=5, rng=None):
    """Like get_oracle_moves, but returns full oracle entries (including overall_progress)."""
    all_correct = enumerate_correct_actions(game_state)
    ok_entries = [e for e in all_correct if e["flag"] == FLAG_OK]
    if rng is not None and len(ok_entries) > n:
        ok_entries = rng.sample(ok_entries, n)
    elif len(ok_entries) > n:
        ok_entries = ok_entries[:n]
    return ok_entries


def candidate_moves_only(candidates):
    return [c["move"] for c in candidates]


def _oracle_match(move_dict, oracle_moves):
    """Matches on action/position/layer only. None for CLARIFY."""
    if move_dict.get("action") not in ("place", "remove"):
        return None
    if not oracle_moves:
        return False
    return any(
        move_dict.get("action") == m["action"]
        and move_dict.get("position") == m["position"]
        and move_dict.get("layer") == m["layer"]
        for m in oracle_moves
    )


_COMMAND_PREFIXES = ("PLACE:", "REMOVE:", "CLARIFY:")


def _cells(move):
    """Unordered set of cells a move covers, so span direction doesn't matter."""
    return frozenset(c for c in (move.get("position"), move.get("span_to")) if c)


def _same_move(a, b):
    if a.get("action") != b.get("action") or a.get("layer") != b.get("layer"):
        return False
    if _cells(a) != _cells(b):
        return False
    # REMOVE carries no block code; PLACE must match colour/size too
    return a.get("action") == "remove" or a.get("block") == b.get("block")


def _strict_match(move_dict, moves):
    """Like _oracle_match but also requires the same block and span."""
    if move_dict.get("action") not in ("place", "remove"):
        return None
    return any(_same_move(move_dict, m) for m in moves)


def seeded_part_type(seed):
    """Deterministic starting board for an eval episode."""
    return random.Random(seed).choice(EnhancedGameState.PARTIAL_OPTIONS)


DIRECTOR_RETRIES = 4


def _director_failed(response):
    return str(response.get("internal_thinking", "")).startswith("Error in generation")


def _call_director(agent, **kwargs):
    """Retry with backoff, since generate_response hides API errors behind a placeholder message.

    Returns (response, retries, failed)."""
    import time
    for attempt in range(DIRECTOR_RETRIES + 1):
        response = agent.generate_response(**kwargs)
        if not _director_failed(response):
            return response, attempt, False
        if attempt < DIRECTOR_RETRIES:
            wait = 2 ** attempt * 5
            print(f"  [director {agent.director_id}] error, retrying in {wait}s: {response['internal_thinking'][:200]}")
            time.sleep(wait)
    return response, DIRECTOR_RETRIES, True


def _extract_command_line(decoded_text, pick="first"):
    """First (or last) line starting with PLACE:/REMOVE:/CLARIFY:, else the first line.

    Use pick="last" for chain-of-thought output, where reasoning may quote moves before the answer."""
    lines = decoded_text.split("\n")
    for line in (reversed(lines) if pick == "last" else lines):
        stripped = line.strip().strip("[]").strip("`* ")
        if stripped.startswith(_COMMAND_PREFIXES):
            return stripped
    return decoded_text.strip().split("\n")[0].strip()


def _is_parse_failure(move_dict):
    """parse_builder_response returns unparseable output as a clarify; detect that case."""
    if move_dict.get("action") != "clarify":
        return False
    text = move_dict.get("clarification", "") or ""
    return text.startswith("Could not parse response") or text.startswith("Parse error")


def run_builder_episode(
    structure_data,
    generate_fn,
    structure_index=0,
    run_id=0,
    oracle_n=5,
    max_turns=15,
    director_model_name="gpt-4.1-mini",
    director_api_key=None,
    director_mode="api",
    shared_director_model=None,
    shared_director_tokenizer=None,
    seed=None,
    part_type=None,
    oracle_in_prompt=True,
    command_pick="first",
    retry_directors=False,
):
    """Run one episode.

    generate_fn(prompt_text, oracle_moves) -> (input_ids, new_tokens, decoded_text), where
    input_ids/new_tokens are 1D CPU tensors. oracle_moves is passed so generate_fn can pick
    the matching system prompt.

    Eval-only options:
        part_type: fixed starting board (see seeded_part_type).
        oracle_in_prompt: False still samples oracle candidates (keeping the RNG stream
            unchanged) but hides them from the builder.
        command_pick: "last" for chain-of-thought outputs.
        retry_directors: retry director API errors.

    Returns per-turn lists prompt_ids, completion_ids, rewards, reward_infos, plus
    part_type, initial_progress, final_progress.
    """
    target_structure = structure_data["structure"]
    target_spans = {int(k): v for k, v in structure_data["spans"].items()}

    game_state = EnhancedGameState(
        target_structure=copy.deepcopy(target_structure),
        target_spans=target_spans,
        partComplete=True,
        partType=part_type,
    )
    # wall starts alias the target's lists into current_structure
    game_state.current_structure = copy.deepcopy(game_state.current_structure)
    tracker = game_state.progress_tracker
    initial_progress = tracker.calculate_progress(game_state.current_structure)["overall_progress"]

    director_agents = build_frozen_directors(
        structure_index, run_id, director_model_name=director_model_name, api_key=director_api_key,
        director_mode=director_mode, shared_director_model=shared_director_model,
        shared_director_tokenizer=shared_director_tokenizer,
    )
    target_director_views = game_state.get_target_director_views()

    rng = random.Random(seed if seed is not None else (structure_index * 7919 + run_id))

    prompt_ids, completion_ids, rewards, reward_infos = [], [], [], []
    conversation_history = []

    for turn in range(max_turns):
        game_state.increment_turn()

        full_board_state = game_state.get_director_views()
        public_conversation = "\n".join(conversation_history)
        director_order = ["D1", "D2", "D3"]
        rng.shuffle(director_order)

        director_responses = {}
        director_kwargs = dict(
            current_view=full_board_state,
            conversation_history=public_conversation,
            available_blocks=game_state.available_blocks,
        )
        director_retries, director_failures = 0, 0
        if retry_directors:
            def call(did):
                return _call_director(director_agents[did], target_view=target_director_views[did], **director_kwargs)
            if director_mode == "api":
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(director_order)) as pool:
                    results = dict(zip(director_order, pool.map(call, director_order)))
            else:
                results = {did: call(did) for did in director_order}
            for did in director_order:
                director_responses[did], n_retry, failed = results[did]
                director_retries += n_retry
                director_failures += int(failed)
        elif director_mode == "api":
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(director_order)) as pool:
                futures = {
                    did: pool.submit(
                        director_agents[did].generate_response,
                        current_view=full_board_state,
                        target_view=target_director_views[did],
                        conversation_history=public_conversation,
                        available_blocks=game_state.available_blocks,
                    )
                    for did in director_order
                }
                director_responses = {did: f.result() for did, f in futures.items()}
        else:
            for did in director_order:
                director_responses[did] = director_agents[did].generate_response(
                    current_view=full_board_state,
                    target_view=target_director_views[did],
                    conversation_history=public_conversation,
                    available_blocks=game_state.available_blocks,
                )

        for did in director_order:
            conversation_history.append(f"{did}: {director_responses[did]['public_message']}")

        discussion = director_discussion_text(director_responses)

        oracle_moves = get_oracle_moves(game_state, n=oracle_n, rng=rng)
        shown_moves = oracle_moves if oracle_in_prompt else []
        all_correct = [e["move"] for e in enumerate_correct_actions(game_state) if e["flag"] == FLAG_OK]

        prompt_text = _BUILDER_HELPER.create_builder_prompt(
            director_discussion=discussion,
            current_state=game_state.current_structure,
            available_blocks=game_state.available_blocks,
            oracle_moves=shown_moves,
        )
        input_ids, new_tokens, decoded_text = generate_fn(prompt_text, shown_moves)
        print(f"  [builder raw] {decoded_text!r}")
        command_line = _extract_command_line(decoded_text, pick=command_pick)
        print(f"  [builder cmd] {command_line!r}")
        move = _BUILDER_HELPER.parse_builder_response(command_line)

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
                # overallState only means "everything placed so far is correct", not completion
                completed = game_state.is_complete()
            else:
                progress_delta, completed = 0.0, False
                move_invalid = True
                print(f"  [execute_move failed] {progress_data.get('error', '<no reason given>')}")
            conversation_history.append(f"Builder: {move.get('confirmation', '')}")

        turns_remaining = max_turns - (turn + 1)
        reward, reward_info = compute_builder_reward(
            move_dict=move,
            progress_delta=progress_delta,
            matched_oracle=matched_oracle,
            completed=completed,
            turns_remaining_at_completion=turns_remaining,
            parse_failure=parse_failure,
            move_invalid=move_invalid,
        )
        reward_info["turn"] = turn
        # eval-only metrics
        reward_info["matched_oracle_strict"] = _strict_match(move, oracle_moves) if oracle_in_prompt else None
        reward_info["correct_move"] = _strict_match(move, all_correct)
        reward_info["director_retries"] = director_retries
        reward_info["director_failures"] = director_failures
        reward_info["completion_tokens"] = int(len(new_tokens)) if hasattr(new_tokens, "__len__") else None
        reward_info["progress"] = tracker.calculate_progress(game_state.current_structure)["overall_progress"]
        print(
            f"  [rollout] structure={structure_index} turn={turn + 1}/{max_turns} "
            f"action={move.get('action')} reward={reward:.3f} "
            f"progress_delta={progress_delta:.3f} completed={completed}"
        )

        prompt_ids.append(input_ids)
        completion_ids.append(new_tokens)
        rewards.append(reward)
        reward_infos.append(reward_info)

        if completed:
            break

    final_progress = tracker.calculate_progress(game_state.current_structure)["overall_progress"]
    if reward_infos:
        reward_infos[-1]["final_progress"] = final_progress

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "rewards": rewards,
        "reward_infos": reward_infos,
        "part_type": game_state.partType,
        "initial_progress": initial_progress,
        "final_progress": final_progress,
    }
