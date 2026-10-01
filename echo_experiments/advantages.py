"""
ECHO advantage computation and baselines.

ECHO ("echo" mode): per-turn cross-generation normalization.
  Normalizes each turn's reward across G parallel rollouts of the same
  structure, giving a turn-level advantage that compares completions at the
  same turn position.

Baselines included:
  episode_return  — REINFORCE: sum rewards per episode, same value broadcast to all turns
  rloo_per_turn   — leave-one-out per turn

"""

import torch

MAX_TURNS = 20


def _echo_advantages(rewards, group_sizes, B, G, device):
    """
    ECHO: normalize within (structure, turn_position) group across G generations.

    rewards:     (total_turns,)
    padded:      (B*G, MAX_TURNS)  — pad each episode to MAX_TURNS
    reshaped:    (B, G, MAX_TURNS) — B structures × G gens × T turns
    normalized across G dim per turn position, then unpadded back to (total_turns,)
    """
    padded   = torch.zeros(B * G, MAX_TURNS, device=device)
    flat_idx = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        padded[ep_idx, :ep_size] = rewards[flat_idx:flat_idx + ep_size]
        flat_idx += ep_size

    padded   = padded.view(B, G, MAX_TURNS)
    mean_r   = padded.mean(dim=1, keepdim=True)
    std_r    = padded.std(dim=1, keepdim=True).clamp_min(1e-4)
    padded   = ((padded - mean_r) / std_r).view(B * G, MAX_TURNS)

    advantages = torch.zeros_like(rewards)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        advantages[flat_idx:flat_idx + ep_size] = padded[ep_idx, :ep_size]
        flat_idx += ep_size

    return advantages


def _episode_return_advantages(rewards, group_sizes, B, G, device):
    """
    REINFORCE: sum rewards per episode, normalize across G episodes per structure.
    All turns in an episode share the same advantage value.

    rewards:     (total_turns,)
    ep_returns:  (B*G,)  — one scalar per episode
    ep_returns:  (B, G)  — normalize across G
    broadcast back to (total_turns,)
    """
    ep_returns = torch.zeros(B * G, device=device)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        ep_returns[ep_idx] = rewards[flat_idx:flat_idx + ep_size].sum()
        flat_idx += ep_size

    ep_returns = ep_returns.view(B, G)
    mean_r     = ep_returns.mean(dim=1, keepdim=True)
    std_r      = ep_returns.std(dim=1, keepdim=True).clamp_min(1e-4)
    ep_adv     = ((ep_returns - mean_r) / std_r).view(B * G)

    advantages = torch.zeros_like(rewards)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        advantages[flat_idx:flat_idx + ep_size] = ep_adv[ep_idx]
        flat_idx += ep_size

    return advantages


def _rloo_per_turn_advantages(rewards, group_sizes, B, G, device):
    """
    RLOO leave-one-out per turn position.

    For each (structure, turn_position), the baseline for generation g is the
    mean of all other G-1 generations at that same position.

    rewards:    (total_turns,)
    padded:     (B*G, MAX_TURNS)
    reshaped:   (B, G, MAX_TURNS)
    loo_mean:   (B, G, MAX_TURNS) — mean excluding self per (b, t)
    """
    padded   = torch.zeros(B * G, MAX_TURNS, device=device)
    flat_idx = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        padded[ep_idx, :ep_size] = rewards[flat_idx:flat_idx + ep_size]
        flat_idx += ep_size

    padded   = padded.view(B, G, MAX_TURNS)
    total    = padded.sum(dim=1, keepdim=True)
    loo_mean = (total - padded) / max(G - 1, 1)
    std_r    = padded.std(dim=1, keepdim=True).clamp_min(1e-4)
    padded   = ((padded - loo_mean) / std_r).view(B * G, MAX_TURNS)

    advantages = torch.zeros_like(rewards)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        advantages[flat_idx:flat_idx + ep_size] = padded[ep_idx, :ep_size]
        flat_idx += ep_size

    return advantages


def compute_multiturn_advantages(rewards, group_sizes, B, G, mode, device, **kwargs):
    """
    Dispatcher for all advantage modes.

    Args:
        rewards     : (total_turns,) flat reward tensor
        group_sizes : list of ints — turns per episode, len = B*G
        B           : number of structures in this batch
        G           : number of generations per structure
        mode        : "echo" | "episode_return" | "rloo_per_turn"
        device      : torch device string

    Returns:
        advantages  : (total_turns,) tensor
    """
    if mode == "echo":
        return _echo_advantages(rewards, group_sizes, B, G, device)
    elif mode == "episode_return":
        return _episode_return_advantages(rewards, group_sizes, B, G, device)
    elif mode == "rloo_per_turn":
        return _rloo_per_turn_advantages(rewards, group_sizes, B, G, device)
    else:
        raise ValueError(f"Unknown advantage mode: {mode!r}. "
                         f"Choose from: echo, episode_return, rloo_per_turn")
