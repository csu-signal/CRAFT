"""
Builder reward for CRAFT-ECHO experiments.
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
        (0.0 when the move failed or wasn't executed, e.g. CLARIFY)
    matched_oracle: True/False if the move was place/remove, None if CLARIFY
        (see rollout._oracle_match)
    completed: EnhancedGameState.execute_move's overallState return value
    turns_remaining_at_completion: max_turns - (turn + 1), only meaningful if completed
    parse_failure: True if parse_builder_response fell back to its
        "Could not parse response" / "Parse error" path
    move_invalid: True if a parsed place/remove failed EnhancedGameState.execute_move's
        own validation (rollout.py's `success` flag) .
    clarify_penalty: flat reward for a CLARIFY.
    invalid_move_penalty: flat reward for a place/remove that broke the game's
        own rules
    positive_progress_weight / negative_progress_weight: scaling applied to
        progress_delta before summing with completion/efficiency bonuses.
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
        reward_info["off_list_penalty"] = off_list_penalty
        reward_info["training_reward"] = off_list_penalty
        return off_list_penalty, reward_info

    if move_dict.get("action") == "clarify":
        reward_info["clarify_penalty"] = clarify_penalty
        reward_info["training_reward"] = clarify_penalty
        return clarify_penalty, reward_info

    if move_invalid:.
        reward_info["invalid_move_penalty"] = invalid_move_penalty
        reward_info["training_reward"] = invalid_move_penalty
        return invalid_move_penalty, reward_info

    if matched_oracle is False:
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
