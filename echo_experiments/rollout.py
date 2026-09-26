"""
Shared multi-turn rollout logic for CRAFT-ECHO experiments.

Runs one live episode of the actual CRAFT game -- 3 frozen director turns
plus one trainable builder turn per turn, repeated until the structure is
complete or a turn cap is hit -- and returns per-turn (prompt, completion,
reward) lists in the flat shape CRAFTEchoTrainer / advantages.py's
compute_multiturn_advantages expects. 
    
"""
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

# dummy key to initialize local builder
_BUILDER_HELPER = BuilderAgent(api_key="unused-local-generation-only")


def build_frozen_directors(
    structure_index, run_id, director_model_name="gpt-4.1-mini", api_key=None,
    director_mode="api", shared_director_model=None, shared_director_tokenizer=None,
):
    """Three DirectorAgents, one per role -- never trained, only called under
    torch.no_grad() from the rollout loop (see trainer.py's generate_fn).

    director_mode="local" reuses one shared local model/tokenizer (loaded once
    by train.py) across all three roles instead of making OpenAI API calls --
    see agents/director_agent.py's _generate_with_local_model and
    local_model_utils.load_local_director_pipeline."""
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
    """
    Same sampling behavior as agents.environment.get_oracle_moves, but keeps
    each candidate's full entry (overall_progress, flag, source) instead of
    stripping down to just the move dict. get_oracle_moves discards
    overall_progress, which the within-turn eval needs for scoring; the live
    training rollout below doesn't need it and just uses get_oracle_moves
    directly.
    """
    all_correct = enumerate_correct_actions(game_state)
    ok_entries = [e for e in all_correct if e["flag"] == FLAG_OK]
    if rng is not None and len(ok_entries) > n:
        ok_entries = rng.sample(ok_entries, n)
    elif len(ok_entries) > n:
        ok_entries = ok_entries[:n]
    return ok_entries


def candidate_moves_only(candidates):
    """Full oracle entries (with overall_progress) -> plain move dicts, the
    shape BuilderAgent.create_builder_prompt/format_oracle_moves_for_prompt
    expects."""
    return [c["move"] for c in candidates]


def _oracle_match(move_dict, oracle_moves):
    """Same check run_craft.py logs as builder_followed_oracle. Returns None
    for CLARIFY (not applicable), True/False for place/remove."""
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


def _extract_command_line(decoded_text):
    """The builder prompt tells the model to think step by step before
    committing to a move, so the PLACE:/REMOVE:/CLARIFY: line often isn't
    the first line of the completion -- scan every line for one that starts
    with a known prefix instead of assuming line 1 is the command, which
    was sending every turn with any leading reasoning straight to
    parse_builder_response's fallback (disguised as a CLARIFY parse
    failure). Falls back to the old first-line behavior so a genuinely
    unparseable completion still surfaces as a parse failure."""
    for line in decoded_text.split("\n"):
        stripped = line.strip().strip("[]").strip("`* ")
        if stripped.startswith(_COMMAND_PREFIXES):
            return stripped
    return decoded_text.strip().split("\n")[0].strip()


def _is_parse_failure(move_dict):
    """parse_builder_response's own fallback disguises unparseable output as
    action == 'clarify' -- distinguish that from a genuine clarification."""
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
):
    """
    Runs one live episode.

    generate_fn(prompt_text, oracle_moves) -> (input_ids, new_tokens, decoded_text)
        is the only model-dependent piece: input_ids/new_tokens are 1D CPU
        long tensors (prompt tokens, completion tokens), decoded_text is the
        detokenized completion. oracle_moves (this turn's candidate list, same
        object passed to create_builder_prompt above) is handed back so
        generate_fn can pick agents.builder_agent's BUILDER_SYSTEM_PROMPT_ORACLE
        vs BUILDER_SYSTEM_PROMPT_BASE exactly the way BuilderAgent.generate_move
        does -- keeping every caller's system-prompt choice consistent with
        run_craft.py's own builder rather than each reinventing one. CRAFTEchoTrainer
        supplies a generate_fn that calls the trainable policy under
        unwrap_model_for_generation.

    Returns a dict with parallel lists (one entry per turn actually taken):
        prompt_ids, completion_ids, rewards, reward_infos
    """
    target_structure = structure_data["structure"]
    target_spans = {int(k): v for k, v in structure_data["spans"].items()}

    game_state = EnhancedGameState(
        target_structure=copy.deepcopy(target_structure),
        target_spans=target_spans,
        partComplete=True,
    )

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

        # ---- frozen directors --------------------------------------------
        full_board_state = game_state.get_director_views()
        public_conversation = "\n".join(conversation_history)
        director_order = ["D1", "D2", "D3"]
        rng.shuffle(director_order)

        director_responses = {}
        if director_mode == "api":

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

        # ---- oracle candidates --------------------------------------------
        oracle_moves = get_oracle_moves(game_state, n=oracle_n, rng=rng)

        # ---- trainable builder ---------------------------------------------
        prompt_text = _BUILDER_HELPER.create_builder_prompt(
            director_discussion=discussion,
            current_state=game_state.current_structure,
            available_blocks=game_state.available_blocks,
            oracle_moves=oracle_moves,
        )
        input_ids, new_tokens, decoded_text = generate_fn(prompt_text, oracle_moves)
        print(f"  [builder raw] {decoded_text!r}")
        command_line = _extract_command_line(decoded_text)
        print(f"  [builder cmd] {command_line!r}")
        move = _BUILDER_HELPER.parse_builder_response(command_line)

        matched_oracle = _oracle_match(move, oracle_moves)
        parse_failure = _is_parse_failure(move)

        # ---- execute ---------------------------------------------------------
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
                # overallState ("everything placed SO FAR is correct") is not
                # completion -- run_craft.py only ever logs it as
                # overall_structure_correctness and never uses it to end a
                # game. The real check (also run_craft.py's own) is whether
                # overall_progress has crossed a real threshold -- a single
                # correct early block would otherwise satisfy overallState
                # vacuously (nothing else has been placed to be wrong yet)
                # and falsely trigger the completion bonus.
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

    # Terminal overall_progress for this episode -- the actual objective
    # (final progress, ideally 1.0) as opposed to progress_delta, which is
    # the dense per-turn shaping signal used for training. Logged separately
    # so training progress and the true end goal can be tracked independently
    # of each other. Attached to the last turn's reward_info since this is an
    # episode-level (not per-turn) quantity and reward_infos is a per-turn list.
    if reward_infos:
        history = game_state.progress_tracker.progress_history
        final_progress = history[-1]["metrics"]["overall_progress"] if history else 0.0
        reward_infos[-1]["final_progress"] = final_progress

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "rewards": rewards,
        "reward_infos": reward_infos,
    }
