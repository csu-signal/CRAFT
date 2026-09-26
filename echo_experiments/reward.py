"""
Builder reward for CRAFT-ECHO experiments.

completion_bonus/efficiency_weight were originally 1.0/0.1 -- checked against
a live 183-step echo run and found corr(reward, completed_rate) = 0.97, with
mean reward -0.046 in steps where zero episodes completed vs +0.028 where at
least one did. completed_rate sits at ~1-3%, so this bonus (20-100x a typical
single-turn progress_delta of ~0.01-0.05) was dominating the training signal
whenever it fired instead of rewarding smooth incremental progress. Lowered
to 0.3/0.02 so finishing still earns real extra credit without swamping the
rest of the signal in whichever small per-step batch happens to contain it.
"""


def compute_builder_reward(
    move_dict,
    progress_delta,
    matched_oracle,
    completed,
    turns_remaining_at_completion=0,
    parse_failure=False,
    move_invalid=False,
    off_list_penalty=-0.1,
    invalid_move_penalty=-0.15,
    completion_bonus=0.3,
    efficiency_weight=0.02,
    clarify_penalty=-0.05,
    positive_progress_weight=1.0,
    negative_progress_weight=1.0,
):
    """
    move_dict: parsed builder output (agents.builder_agent.BuilderAgent.parse_builder_response)
    progress_delta: EnhancedGameState.execute_move's progress_data["progress_delta"]
        (0.0 when the move failed or wasn't executed, e.g. CLARIFY) -- raw,
        unshaped value; the shaping in this function is applied on top of it
        and reflected only in "training_reward", not in this logged field.
    matched_oracle: True/False if the move was place/remove, None if CLARIFY
        (see rollout._oracle_match)
    completed: EnhancedGameState.execute_move's overallState return value
    turns_remaining_at_completion: max_turns - (turn + 1), only meaningful if completed
    parse_failure: True if parse_builder_response fell back to its
        "Could not parse response" / "Parse error" path (disguised as CLARIFY)
    move_invalid: True if a parsed place/remove failed EnhancedGameState.execute_move's
        own validation (rollout.py's `success` flag) -- distinct from matched_oracle,
        which only checks whether the move's fields happen to equal a sampled
        candidate, regardless of whether it would actually execute.
    clarify_penalty: flat reward for a genuine (non-parse-failure) CLARIFY --
        negative but less harsh than off_list_penalty, since CLARIFY is still
        a valid, compliant action, just possibly unwarranted.
    invalid_move_penalty: flat reward for a place/remove that broke the game's
        own rules -- harsher than off_list_penalty (see module docstring).
    positive_progress_weight / negative_progress_weight: scaling applied to
        progress_delta before summing with completion/efficiency bonuses.
        Left at symmetric 1.0/1.0 by default -- see module docstring for why
        (potential-based shaping / telescoping to final progress). Exposed
        as kwargs in case asymmetric exploration incentives are worth
        revisiting later, but that's now a deliberate opt-in, not the default.
    """
    reward_info = {
        "action": move_dict.get("action"),
        "progress_delta": progress_delta,
        "matched_oracle": matched_oracle,
        "parse_failure": parse_failure,
        "move_invalid": move_invalid,
        "completed": completed,
        "off_list_penalty": 0.0,
        "invalid_move_penalty": 0.0,
        "clarify_penalty": 0.0,
        "completion_bonus": 0.0,
        "efficiency_bonus": 0.0,
        "training_reward": 0.0,
    }

    if parse_failure:
        # Genuinely non-compliant output — worse than a deliberate CLARIFY.
        reward_info["off_list_penalty"] = off_list_penalty
        reward_info["training_reward"] = off_list_penalty
        return off_list_penalty, reward_info

    if move_dict.get("action") == "clarify":
        reward_info["clarify_penalty"] = clarify_penalty
        reward_info["training_reward"] = clarify_penalty
        return clarify_penalty, reward_info

    if move_invalid:
        # Parsed fine, but broke the game's own placement rules (bad layer,
        # unknown block, illegal span, ...) -- a harder failure than simply
        # picking a valid move that wasn't in the sampled oracle candidates.
        reward_info["invalid_move_penalty"] = invalid_move_penalty
        reward_info["training_reward"] = invalid_move_penalty
        return invalid_move_penalty, reward_info

    if matched_oracle is False:
        # Parsed as place/remove but doesn't match any oracle-verified candidate.
        reward_info["off_list_penalty"] = off_list_penalty
        reward_info["training_reward"] = off_list_penalty
        return off_list_penalty, reward_info

    shaping_weight = positive_progress_weight if progress_delta >= 0 else negative_progress_weight
    reward = progress_delta * shaping_weight
    if completed:
        reward_info["completion_bonus"] = completion_bonus
        reward_info["efficiency_bonus"] = efficiency_weight * turns_remaining_at_completion
        reward += reward_info["completion_bonus"] + reward_info["efficiency_bonus"]

    reward_info["training_reward"] = reward
    return reward, reward_info


class BuilderRewardFunction:
    def __init__(self, **reward_kwargs):
        self.reward_kwargs = reward_kwargs
        self.__name__ = "builder_reward"
        self.last_reward_infos = []

    def __call__(self, prompts, completions, **kwargs):
        raise NotImplementedError(
            "BuilderRewardFunction is not called through TRL's reward_funcs "
            "CRAFTEchoTrainer computes rewards inline per turn "
            "via compute_builder_reward."
        )
