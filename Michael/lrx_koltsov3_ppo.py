#!/usr/bin/env python3
"""Starter PPO implementation for Koltsov3 permutation traversal.

This script is a standalone baseline intended to begin replacing the RL section
of the LRX workflow with PPO while using Koltsov3 generators.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    torch = None
    nn = None


# -----------------------------------------------------------------------------
# Koltsov3 utilities (NumPy)
# -----------------------------------------------------------------------------

def get_koltsov3_generators(n: int, k: int = 0) -> np.ndarray:
    """Return (3, n) Koltsov3 generator permutations [I, K, S]."""
    if n < 3:
        raise ValueError("n must be at least 3 for Koltsov3 generators")
    if not (0 <= k <= n - 3):
        raise ValueError(f"k must be in [0, {n - 3}] for n={n}")

    I = np.arange(n, dtype=np.int64)
    K = np.arange(n, dtype=np.int64)
    S = np.arange(n, dtype=np.int64)

    for i in range(0, n - 1, 2):
        I[i], I[i + 1] = I[i + 1], I[i]
    for i in range(1, n - 1, 2):
        K[i], K[i + 1] = K[i + 1], K[i]
    S[k], S[k + 2] = S[k + 2], S[k]

    return np.stack([I, K, S], axis=0)


def apply_generator(state: np.ndarray, generator: np.ndarray) -> np.ndarray:
    """Apply one permutation generator to a state permutation."""
    return state[generator]


def distance_to_identity(state: np.ndarray) -> int:
    """Simple shaping distance: number of misplaced positions."""
    return int(np.sum(state != np.arange(state.shape[0], dtype=state.dtype)))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimation (GAE-Lambda)."""
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv, returns


def compute_potential_shaping_reward(
    prev_potential: float,
    next_potential: float,
    *,
    step_penalty: float,
    success_bonus: float,
    done: bool,
) -> float:
    """Potential-shaped reward using a fixed heuristic as the shaping potential."""
    reward = float(prev_potential - next_potential - step_penalty)
    if done:
        reward += success_bonus
    return reward


def koltsov3_feature_dim(n: int) -> int:
    """Return the engineered Koltsov3 feature dimension for length ``n``."""
    if n < 3:
        raise ValueError("n must be at least 3 for Koltsov3 features")
    return 6 * n + 35


def _columnize_np(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    return array.astype(np.float32, copy=False)


def extract_koltsov3_features_np(
    states_array: np.ndarray,
    n: int,
    k: int = 0,
) -> np.ndarray:
    """Extract the engineered Koltsov3 feature vector used by the repo baselines."""
    states_array = np.asarray(states_array, dtype=np.int64)
    if states_array.ndim == 1:
        states_array = states_array[None, :]
    if states_array.ndim != 2 or states_array.shape[1] != n:
        raise ValueError(f"states_array must have shape (batch, {n})")
    if not (0 <= k <= n - 3):
        raise ValueError(f"k must be in [0, {n - 3}] for n={n}")

    ident = np.arange(n, dtype=np.int64)
    feature_blocks: list[np.ndarray] = []

    disp = np.abs(states_array - ident).astype(np.float32)
    feature_blocks.append(disp)
    feature_blocks.extend(
        [
            _columnize_np(disp.sum(axis=1)),
            _columnize_np(disp.max(axis=1)),
            _columnize_np(disp.mean(axis=1)),
            _columnize_np(disp.std(axis=1)),
            _columnize_np(np.median(disp, axis=1)),
            _columnize_np((disp == 0).sum(axis=1)),
            _columnize_np((disp == 1).sum(axis=1)),
            _columnize_np((disp >= 2).sum(axis=1)),
            _columnize_np((disp >= 4).sum(axis=1)),
        ]
    )

    pm = ((states_array & 1) != (ident & 1)).astype(np.float32)
    feature_blocks.append(pm)
    feature_blocks.extend(
        [
            _columnize_np(pm.sum(axis=1)),
            _columnize_np(pm.mean(axis=1)),
            _columnize_np((pm * disp).sum(axis=1)),
        ]
    )

    sorted_adj = (states_array[:, :-1] < states_array[:, 1:]).astype(np.float32)
    feature_blocks.append(sorted_adj)
    feature_blocks.append(_columnize_np(sorted_adj.sum(axis=1)))

    adj_diff = np.abs(np.diff(states_array, axis=1)).astype(np.float32)
    feature_blocks.append(adj_diff)
    feature_blocks.extend(
        [
            _columnize_np(adj_diff.sum(axis=1)),
            _columnize_np(adj_diff.max(axis=1)),
        ]
    )
    pair_correct = (
        (states_array[:, :-1] == ident[:-1])
        & (states_array[:, 1:] == ident[1:])
    ).astype(np.float32)
    feature_blocks.append(_columnize_np(pair_correct.sum(axis=1)))

    invs = np.zeros(states_array.shape[0], dtype=np.float32)
    for i in range(n):
        invs += (states_array[:, i : i + 1] > states_array[:, i + 1 :]).sum(axis=1)
    feature_blocks.extend(
        [
            _columnize_np(invs),
            _columnize_np(invs / (n * (n - 1) / 2.0)),
            _columnize_np((invs.astype(np.int64) & 1).astype(np.float32)),
        ]
    )

    desc = (states_array[:, :-1] > states_array[:, 1:]).sum(axis=1).astype(np.float32)
    feature_blocks.append(_columnize_np(desc))

    i_unsorted = np.zeros(states_array.shape[0], dtype=np.float32)
    i_correct = np.zeros(states_array.shape[0], dtype=np.float32)
    for idx in range(0, n - 1, 2):
        i_unsorted += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        i_correct += (
            (states_array[:, idx] == idx)
            & (states_array[:, idx + 1] == idx + 1)
        ).astype(np.float32)
    feature_blocks.extend([_columnize_np(i_unsorted), _columnize_np(i_correct)])

    k_unsorted = np.zeros(states_array.shape[0], dtype=np.float32)
    k_correct = np.zeros(states_array.shape[0], dtype=np.float32)
    for idx in range(1, n - 1, 2):
        k_unsorted += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        k_correct += (
            (states_array[:, idx] == idx)
            & (states_array[:, idx + 1] == idx + 1)
        ).astype(np.float32)
    feature_blocks.extend([_columnize_np(k_unsorted), _columnize_np(k_correct)])

    feature_blocks.extend(
        [
            _columnize_np((states_array[:, k] < states_array[:, k + 2]).astype(np.float32)),
            _columnize_np(np.abs(states_array[:, k] - states_array[:, k + 2]).astype(np.float32)),
            _columnize_np(
                np.abs(states_array[:, k] - k)
                + np.abs(states_array[:, k + 2] - (k + 2))
            ),
            _columnize_np(
                (
                    (states_array[:, k] == k + 2)
                    & (states_array[:, k + 2] == k)
                ).astype(np.float32)
            ),
        ]
    )

    lb_disp = disp.max(axis=1) / 2.0
    lb_parity = pm.sum(axis=1) / 2.0
    feature_blocks.extend(
        [
            _columnize_np(lb_disp),
            _columnize_np(lb_parity),
            _columnize_np(np.maximum(lb_disp, lb_parity)),
        ]
    )

    inv_perm = np.argsort(states_array, axis=1).astype(np.float32)
    feature_blocks.append(inv_perm)
    feature_blocks.extend(
        [
            _columnize_np(np.abs(inv_perm[:, 0] - 0.0)),
            _columnize_np(np.abs(inv_perm[:, n - 1] - float(n - 1))),
        ]
    )

    at_home = states_array == ident
    run_lens = np.zeros(states_array.shape[0], dtype=np.float32)
    current = np.zeros(states_array.shape[0], dtype=np.int64)
    for col in range(n):
        current = np.where(at_home[:, col], current + 1, 0)
        run_lens = np.maximum(run_lens, current.astype(np.float32))
    feature_blocks.append(_columnize_np(run_lens))

    signed_disp = states_array.astype(np.float32) - ident.astype(np.float32)
    feature_blocks.extend(
        [
            _columnize_np(np.maximum(signed_disp, 0.0).sum(axis=1)),
            _columnize_np(np.maximum(-signed_disp, 0.0).sum(axis=1)),
            _columnize_np(signed_disp.sum(axis=1)),
        ]
    )

    feature_blocks.append(states_array.astype(np.float32))

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    expected_dim = koltsov3_feature_dim(n)
    if features.shape[1] != expected_dim:
        raise RuntimeError(
            f"feature encoder produced {features.shape[1]} dims, expected {expected_dim}"
        )
    return features


def _columnize_torch(array: torch.Tensor) -> torch.Tensor:
    if array.ndim == 1:
        array = array.unsqueeze(1)
    return array.to(torch.float32)


def extract_koltsov3_features_torch(
    states: torch.Tensor,
    n: int,
    k: int = 0,
) -> torch.Tensor:
    """Torch version of the engineered Koltsov3 feature extractor."""
    _require_torch()

    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.ndim != 2 or states.shape[1] != n:
        raise ValueError(f"states must have shape (batch, {n})")
    if not (0 <= k <= n - 3):
        raise ValueError(f"k must be in [0, {n - 3}] for n={n}")

    states = states.to(dtype=torch.long)
    ident = torch.arange(n, dtype=torch.long, device=states.device).unsqueeze(0)
    feature_blocks: list[torch.Tensor] = []

    disp = (states - ident).abs().to(torch.float32)
    feature_blocks.append(disp)
    feature_blocks.extend(
        [
            disp.sum(dim=1, keepdim=True),
            disp.max(dim=1, keepdim=True).values,
            disp.mean(dim=1, keepdim=True),
            disp.std(dim=1, keepdim=True, unbiased=False),
            disp.median(dim=1, keepdim=True).values,
            (disp == 0).sum(dim=1, keepdim=True).to(torch.float32),
            (disp == 1).sum(dim=1, keepdim=True).to(torch.float32),
            (disp >= 2).sum(dim=1, keepdim=True).to(torch.float32),
            (disp >= 4).sum(dim=1, keepdim=True).to(torch.float32),
        ]
    )

    pm = ((states & 1) != (ident & 1)).to(torch.float32)
    feature_blocks.append(pm)
    feature_blocks.extend(
        [
            pm.sum(dim=1, keepdim=True),
            pm.mean(dim=1, keepdim=True),
            (pm * disp).sum(dim=1, keepdim=True),
        ]
    )

    sorted_adj = (states[:, :-1] < states[:, 1:]).to(torch.float32)
    feature_blocks.append(sorted_adj)
    feature_blocks.append(sorted_adj.sum(dim=1, keepdim=True))

    adj_diff = (states[:, 1:] - states[:, :-1]).abs().to(torch.float32)
    feature_blocks.append(adj_diff)
    feature_blocks.extend(
        [
            adj_diff.sum(dim=1, keepdim=True),
            adj_diff.max(dim=1, keepdim=True).values,
        ]
    )
    pair_correct = (
        (states[:, :-1] == ident[:, :-1]) & (states[:, 1:] == ident[:, 1:])
    ).to(torch.float32)
    feature_blocks.append(pair_correct.sum(dim=1, keepdim=True))

    invs = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    for i in range(n):
        invs += (states[:, i : i + 1] > states[:, i + 1 :]).sum(dim=1).to(torch.float32)
    feature_blocks.extend(
        [
            _columnize_torch(invs),
            _columnize_torch(invs / (n * (n - 1) / 2.0)),
            _columnize_torch((invs.to(torch.long) & 1).to(torch.float32)),
        ]
    )

    desc = (states[:, :-1] > states[:, 1:]).sum(dim=1).to(torch.float32)
    feature_blocks.append(_columnize_torch(desc))

    i_unsorted = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    i_correct = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    for idx in range(0, n - 1, 2):
        i_unsorted += (states[:, idx] > states[:, idx + 1]).to(torch.float32)
        i_correct += (
            (states[:, idx] == idx) & (states[:, idx + 1] == idx + 1)
        ).to(torch.float32)
    feature_blocks.extend([_columnize_torch(i_unsorted), _columnize_torch(i_correct)])

    k_unsorted = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    k_correct = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    for idx in range(1, n - 1, 2):
        k_unsorted += (states[:, idx] > states[:, idx + 1]).to(torch.float32)
        k_correct += (
            (states[:, idx] == idx) & (states[:, idx + 1] == idx + 1)
        ).to(torch.float32)
    feature_blocks.extend([_columnize_torch(k_unsorted), _columnize_torch(k_correct)])

    feature_blocks.extend(
        [
            _columnize_torch((states[:, k] < states[:, k + 2]).to(torch.float32)),
            _columnize_torch((states[:, k] - states[:, k + 2]).abs().to(torch.float32)),
            _columnize_torch(
                ((states[:, k] - k).abs() + (states[:, k + 2] - (k + 2)).abs()).to(torch.float32)
            ),
            _columnize_torch(
                ((states[:, k] == k + 2) & (states[:, k + 2] == k)).to(torch.float32)
            ),
        ]
    )

    lb_disp = disp.max(dim=1).values / 2.0
    lb_parity = pm.sum(dim=1) / 2.0
    feature_blocks.extend(
        [
            _columnize_torch(lb_disp),
            _columnize_torch(lb_parity),
            _columnize_torch(torch.maximum(lb_disp, lb_parity)),
        ]
    )

    inv_perm = torch.argsort(states, dim=1).to(torch.float32)
    feature_blocks.append(inv_perm)
    feature_blocks.extend(
        [
            _columnize_torch((inv_perm[:, 0] - 0.0).abs()),
            _columnize_torch((inv_perm[:, n - 1] - float(n - 1)).abs()),
        ]
    )

    at_home = states.eq(ident)
    run_lens = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    current = torch.zeros(states.shape[0], dtype=torch.int64, device=states.device)
    for col in range(n):
        current = torch.where(at_home[:, col], current + 1, torch.zeros_like(current))
        run_lens = torch.maximum(run_lens, current.to(torch.float32))
    feature_blocks.append(_columnize_torch(run_lens))

    signed_disp = states.to(torch.float32) - ident.to(torch.float32)
    feature_blocks.extend(
        [
            _columnize_torch(torch.maximum(signed_disp, torch.zeros_like(signed_disp)).sum(dim=1)),
            _columnize_torch(torch.maximum(-signed_disp, torch.zeros_like(signed_disp)).sum(dim=1)),
            _columnize_torch(signed_disp.sum(dim=1)),
        ]
    )

    feature_blocks.append(states.to(torch.float32))

    features = torch.cat(feature_blocks, dim=1)
    expected_dim = koltsov3_feature_dim(n)
    if features.shape[1] != expected_dim:
        raise RuntimeError(
            f"feature encoder produced {features.shape[1]} dims, expected {expected_dim}"
        )
    return features


# -----------------------------------------------------------------------------
# PPO model + training loop (PyTorch)
# -----------------------------------------------------------------------------


@dataclass
class PPOConfig:
    n: int = 16
    k: int = 0
    total_steps: int = 50_000
    rollout_steps: int = 1024
    max_episode_steps: Optional[int] = None
    max_scramble_steps: Optional[int] = None
    hidden_dim: int = 512
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    value_coef: float = 0.5
    heuristic_coef: float = 1.0
    entropy_coef: float = 0.01
    clip_coef: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 256
    grad_clip_norm: float = 0.5
    target_kl: Optional[float] = None
    anneal_learning_rate: bool = False
    heuristic_teacher_refresh_updates: int = 0
    offpolicy_heuristic_refresh_updates: int = 0
    offpolicy_heuristic_refresh_epochs: int = 1
    success_bonus: float = 5.0
    step_penalty: float = 1.0
    warmup_epochs: int = 0
    warmup_walks: int = 256
    warmup_walk_length: Optional[int] = None
    warmup_batch_size: int = 512
    warmup_history_size: int = 32
    bellman_epochs: int = 0
    bellman_batch_size: int = 512
    policy_warmstart_epochs: int = 0
    policy_warmstart_batch_size: int = 512
    policy_head_lr: Optional[float] = None
    seed: int = 42
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    log_every_updates: int = 10
    eval_every_updates: int = 0
    eval_beam_width: int = 64
    eval_step_limit: Optional[int] = None
    eval_history_size: int = 32
    eval_policy_alpha: float = 1.0
    eval_apply_x_trick: bool = True

    def __post_init__(self) -> None:
        if self.max_episode_steps is None:
            self.max_episode_steps = 4 * self.n
        if self.max_scramble_steps is None:
            self.max_scramble_steps = 3 * self.n
        if self.warmup_walk_length is None:
            self.warmup_walk_length = 8 * self.n
        if self.policy_head_lr is None:
            self.policy_head_lr = self.policy_lr
        if self.eval_step_limit is None:
            self.eval_step_limit = 2 * max(1, self.n * (self.n - 1) // 2)
        if self.rollout_steps % self.minibatch_size != 0:
            raise ValueError(
                "rollout_steps must be divisible by minibatch_size for stable minibatches"
            )
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when provided")
        if self.heuristic_teacher_refresh_updates < 0:
            raise ValueError("heuristic_teacher_refresh_updates must be non-negative")
        if self.offpolicy_heuristic_refresh_updates < 0:
            raise ValueError("offpolicy_heuristic_refresh_updates must be non-negative")
        if self.offpolicy_heuristic_refresh_epochs < 0:
            raise ValueError("offpolicy_heuristic_refresh_epochs must be non-negative")


if nn is not None:
    class ActorCritic(nn.Module):
        """Engineered Koltsov3 feature encoder + shared trunk + policy/value heads."""

        def __init__(self, n: int, hidden_dim: int, k: int = 0):
            super().__init__()
            self.n = n
            self.k = k
            input_dim = koltsov3_feature_dim(n)
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_dim, 3)
            self.value_head = nn.Linear(hidden_dim, 1)
            self.critic_head = nn.Linear(hidden_dim, 1)

        def _encode(self, obs: torch.Tensor) -> torch.Tensor:
            return extract_koltsov3_features_torch(obs, n=self.n, k=self.k)

        def _forward_heads(
            self,
            obs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = self._encode(obs)
            h = self.trunk(x)
            logits = self.policy_head(h)
            heuristic_values = self.value_head(h).squeeze(-1)
            critic_values = self.critic_head(h).squeeze(-1)
            return logits, heuristic_values, critic_values

        def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits, heuristic_values, _ = self._forward_heads(obs)
            return logits, heuristic_values

        def actor_critic(
            self,
            obs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self._forward_heads(obs)
else:
    class ActorCritic:  # pragma: no cover - used only when torch is unavailable
        def __init__(self, *_args, **_kwargs):
            _require_torch()


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for PPO training. Install dependencies first."
        )


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sync_reference_model(reference_model: ActorCritic, source_model: ActorCritic) -> None:
    """Copy the current model weights into a frozen evaluation-only reference model."""
    reference_model.load_state_dict(source_model.state_dict())
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)


def build_ppo_optimizer(model: ActorCritic, config: PPOConfig) -> torch.optim.Optimizer:
    """Use separate learning rates for the policy head and shared/value parameters."""
    return torch.optim.Adam(
        [
            {
                "params": list(model.trunk.parameters())
                + list(model.value_head.parameters())
                + list(model.critic_head.parameters()),
                "lr": config.value_lr,
            },
            {
                "params": model.policy_head.parameters(),
                "lr": config.policy_lr,
            },
        ]
    )


def anneal_ppo_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    *,
    update_idx: int,
    total_updates: int,
) -> None:
    """Linearly anneal PPO learning rates across updates when requested."""
    if not config.anneal_learning_rate:
        return

    progress_remaining = max(
        0.0,
        1.0 - (update_idx - 1) / max(1, total_updates),
    )
    optimizer.param_groups[0]["lr"] = config.value_lr * progress_remaining
    optimizer.param_groups[1]["lr"] = config.policy_lr * progress_remaining


def generate_nonbacktracking_walk_dataset(
    generators: np.ndarray,
    num_walks: int,
    walk_length: int,
    *,
    rng: np.random.Generator,
    history_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate first-visit walk-depth targets from non-backtracking walks."""
    if num_walks <= 0:
        raise ValueError("num_walks must be positive")
    if walk_length <= 0:
        raise ValueError("walk_length must be positive")
    if history_size < 0:
        raise ValueError("history_size must be non-negative")

    n = generators.shape[1]
    identity = np.arange(n, dtype=np.int64)
    states = [identity.copy()]
    steps = [0.0]

    for _ in range(num_walks):
        state = identity.copy()
        seen_in_walk = {tuple(state.tolist())}
        history: list[tuple[int, ...]] = [tuple(state.tolist())]

        for step in range(1, walk_length + 1):
            action_order = rng.permutation(generators.shape[0])
            next_state = None
            next_key = None
            for action in action_order:
                candidate = apply_generator(state, generators[int(action)])
                candidate_key = tuple(candidate.tolist())
                if history_size == 0 or candidate_key not in history:
                    next_state = candidate
                    next_key = candidate_key
                    break

            if next_state is None:
                action = int(action_order[0])
                next_state = apply_generator(state, generators[action])
                next_key = tuple(next_state.tolist())

            state = next_state
            history.append(next_key)
            if history_size > 0 and len(history) > history_size:
                history.pop(0)

            if next_key not in seen_in_walk:
                seen_in_walk.add(next_key)
                states.append(state.copy())
                steps.append(float(step))

    return np.stack(states, axis=0), np.asarray(steps, dtype=np.float32)


def generate_fixed_depth_walk_states(
    generators: np.ndarray,
    num_states: int,
    walk_length: int,
    *,
    seed: int,
    history_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate unique final states from deterministic non-backtracking walks."""
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if walk_length <= 0:
        raise ValueError("walk_length must be positive")
    if history_size < 0:
        raise ValueError("history_size must be non-negative")

    rng = np.random.default_rng(seed)
    n = generators.shape[1]
    identity = np.arange(n, dtype=np.int64)

    states: list[np.ndarray] = []
    witness_lengths: list[float] = []
    seen = set()
    attempts = 0
    max_attempts = max(64, num_states * 32)

    while len(states) < num_states and attempts < max_attempts:
        state = identity.copy()
        history: list[tuple[int, ...]] = [tuple(state.tolist())]

        for _ in range(walk_length):
            action_order = rng.permutation(generators.shape[0])
            next_state = None
            for action in action_order:
                candidate = apply_generator(state, generators[int(action)])
                candidate_key = tuple(candidate.tolist())
                if history_size == 0 or candidate_key not in history[-history_size:]:
                    next_state = candidate
                    break

            if next_state is None:
                next_state = apply_generator(state, generators[int(action_order[0])])

            state = next_state
            history.append(tuple(state.tolist()))

        state_key = tuple(state.tolist())
        if state_key not in seen:
            seen.add(state_key)
            states.append(state.copy())
            witness_lengths.append(float(walk_length))
        attempts += 1

    if len(states) < num_states:
        raise RuntimeError(
            f"only generated {len(states)} unique states out of requested {num_states}"
        )

    return np.stack(states, axis=0), np.asarray(witness_lengths, dtype=np.float32)


def compute_bellman_clipped_targets(
    model,
    states: np.ndarray,
    generators: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Compute clipped Bellman targets ``1 + min_a V(s')`` for a state batch."""
    _require_torch()

    if states.shape[0] != upper_bounds.shape[0]:
        raise ValueError("states and upper_bounds must have the same batch dimension")

    n = states.shape[1]
    identity = np.arange(n, dtype=np.int64)
    targets = np.zeros(states.shape[0], dtype=np.float32)
    was_training = getattr(model, "training", False)
    model.eval()

    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        batch_states = states[start:end]
        batch_upper = upper_bounds[start:end]
        neighbor_states = np.stack([batch_states[:, gen] for gen in generators], axis=1)
        neighbor_t = torch.tensor(
            neighbor_states.reshape(-1, n),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            _, neighbor_values = model(neighbor_t)

        bellman = 1.0 + neighbor_values.view(-1, generators.shape[0]).min(dim=1).values
        clipped = np.clip(bellman.cpu().numpy().astype(np.float32), 0.0, batch_upper)
        clipped[np.all(batch_states == identity, axis=1)] = 0.0
        targets[start:end] = clipped

    model.train(was_training)
    return targets


def predict_heuristic_values(
    model,
    states: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Predict heuristic values for a batch of states using the search value head."""
    _require_torch()

    states = np.asarray(states, dtype=np.int64)
    if states.ndim == 1:
        states = states[None, :]

    values = np.zeros(states.shape[0], dtype=np.float32)
    was_training = getattr(model, "training", False)
    model.eval()

    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        obs_t = torch.tensor(states[start:end], dtype=torch.long, device=device)
        with torch.no_grad():
            _, batch_values = model(obs_t)
        values[start:end] = batch_values.cpu().numpy().astype(np.float32)

    model.train(was_training)
    return values


def compute_reference_heuristic_targets(
    reference_model,
    states: np.ndarray,
    generators: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Compute fixed Bellman-style targets from the frozen reference heuristic."""
    upper_bounds = np.maximum(
        predict_heuristic_values(
            reference_model,
            states,
            device=device,
            batch_size=batch_size,
        ),
        0.0,
    )
    return compute_bellman_clipped_targets(
        reference_model,
        states,
        generators,
        upper_bounds,
        device=device,
        batch_size=batch_size,
    )


def successor_values_to_soft_action_targets(
    successor_values: np.ndarray,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """Convert per-action successor values into uniform soft labels over tied best moves."""
    best_values = successor_values.min(axis=1, keepdims=True)
    best_mask = np.isclose(successor_values, best_values, atol=atol)
    targets = best_mask.astype(np.float32)
    targets /= targets.sum(axis=1, keepdims=True)
    return targets


def compute_policy_imitation_targets(
    model,
    states: np.ndarray,
    generators: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Label each state with a soft target over value-best successor actions."""
    _require_torch()

    n = states.shape[1]
    successor_values = np.zeros((states.shape[0], generators.shape[0]), dtype=np.float32)
    was_training = getattr(model, "training", False)
    model.eval()

    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        batch_states = states[start:end]
        neighbor_states = np.stack([batch_states[:, gen] for gen in generators], axis=1)
        neighbor_t = torch.tensor(
            neighbor_states.reshape(-1, n),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            _, neighbor_values = model(neighbor_t)
        successor_values[start:end] = neighbor_values.view(-1, generators.shape[0]).cpu().numpy()

    model.train(was_training)
    return successor_values_to_soft_action_targets(successor_values)


def _run_value_regression_epochs(
    model: ActorCritic,
    states: np.ndarray,
    targets: np.ndarray,
    *,
    device: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    grad_clip_norm: float,
    log_prefix: str,
) -> None:
    if epochs <= 0:
        return

    for epoch_idx in range(1, epochs + 1):
        permutation = np.random.permutation(states.shape[0])
        batch_losses = []

        for start in range(0, states.shape[0], batch_size):
            end = start + batch_size
            batch_idx = permutation[start:end]
            obs_t = torch.tensor(states[batch_idx], dtype=torch.long, device=device)
            target_t = torch.tensor(targets[batch_idx], dtype=torch.float32, device=device)

            _, predicted_values = model(obs_t)
            value_loss = nn.functional.mse_loss(predicted_values, target_t)

            optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_losses.append(float(value_loss.item()))

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        if epoch_idx == 1 or epoch_idx == epochs or epoch_idx % max(1, epochs // 5) == 0:
            print(
                f"{log_prefix}_epoch={epoch_idx:03d}/{epochs:03d} "
                f"value_loss={mean_loss:.6f}"
            )


def _run_policy_warmstart_epochs(
    model: ActorCritic,
    states: np.ndarray,
    targets: np.ndarray,
    *,
    device: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    grad_clip_norm: float,
    log_prefix: str,
) -> None:
    if epochs <= 0 or states.shape[0] == 0:
        return

    for epoch_idx in range(1, epochs + 1):
        permutation = np.random.permutation(states.shape[0])
        batch_losses = []

        for start in range(0, states.shape[0], batch_size):
            end = start + batch_size
            batch_idx = permutation[start:end]
            obs_t = torch.tensor(states[batch_idx], dtype=torch.long, device=device)
            target_t = torch.tensor(targets[batch_idx], dtype=torch.float32, device=device)

            logits, _ = model(obs_t)
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = -(target_t * log_probs).sum(dim=1).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(model.policy_head.parameters(), grad_clip_norm)
            optimizer.step()

            batch_losses.append(float(policy_loss.item()))

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        if epoch_idx == 1 or epoch_idx == epochs or epoch_idx % max(1, epochs // 5) == 0:
            print(
                f"{log_prefix}_epoch={epoch_idx:03d}/{epochs:03d} "
                f"policy_loss={mean_loss:.6f}"
            )


@dataclass
class BeamSearchResult:
    path_found: bool
    path: Optional[list[int]]
    path_length: int
    steps_taken: int
    states_scored: int
    runtime_sec: float


@dataclass(frozen=True)
class HardStateEvalSummary:
    success_rate: float
    median_path_ratio: float
    reference_kind: str
    median_path_length: float
    median_runtime_sec: float
    solved_count: int
    total_count: int
    source: str
    beam_width: int
    step_limit: int
    policy_alpha: float
    apply_x_trick: bool

    def comparison_key(self) -> tuple[float, float, float]:
        path_quality = (
            -self.median_path_ratio if np.isfinite(self.median_path_ratio) else float("-inf")
        )
        runtime_quality = (
            -self.median_runtime_sec if np.isfinite(self.median_runtime_sec) else float("-inf")
        )
        return (self.success_rate, path_quality, runtime_quality)


def valid_koltsov3_actions(state: np.ndarray, *, apply_x_trick: bool = False) -> tuple[int, ...]:
    """Return the allowed actions for a state under the optional X-trick."""
    if apply_x_trick and state[0] < state[1]:
        return (1, 2)
    return (0, 1, 2)


def score_policy_guided_candidates(
    candidate_values: np.ndarray,
    candidate_log_probs: np.ndarray,
    *,
    policy_alpha: float,
) -> np.ndarray:
    """Score beam candidates using value plus a policy prior term."""
    return candidate_values - policy_alpha * candidate_log_probs


def beam_search_with_policy_prior(
    start_state: np.ndarray,
    model,
    generators: np.ndarray,
    *,
    beam_width: int,
    step_limit: int,
    policy_alpha: float,
    device: str = "auto",
    deduplicate: bool = True,
    history_size: int = 32,
    apply_x_trick: bool = False,
) -> BeamSearchResult:
    """Beam search that ranks candidates by value plus PPO policy prior."""
    _require_torch()

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if step_limit < 0:
        raise ValueError("step_limit must be non-negative")
    if history_size < 0:
        raise ValueError("history_size must be non-negative")

    device = _select_device(device)
    start_state = np.asarray(start_state, dtype=np.int64)
    identity = np.arange(start_state.shape[0], dtype=np.int64)
    t0 = time.perf_counter()

    if np.array_equal(start_state, identity):
        return BeamSearchResult(
            path_found=True,
            path=[],
            path_length=0,
            steps_taken=0,
            states_scored=0,
            runtime_sec=time.perf_counter() - t0,
        )

    beam_states = start_state[None, :]
    beam_paths: list[list[int]] = [[]]
    visited = {tuple(start_state.tolist())} if deduplicate else None
    history: list[tuple[int, ...]] = [tuple(start_state.tolist())] if history_size > 0 else []
    total_states_scored = 0
    was_training = getattr(model, "training", False)
    model.eval()

    for step in range(step_limit):
        obs_t = torch.tensor(beam_states, dtype=torch.long, device=device)
        with torch.no_grad():
            parent_logits, _ = model(obs_t)
        parent_log_probs = torch.log_softmax(parent_logits, dim=-1).cpu().numpy()

        candidate_states: list[np.ndarray] = []
        candidate_paths: list[list[int]] = []
        candidate_log_probs: list[float] = []

        for parent_idx, state in enumerate(beam_states):
            for action in valid_koltsov3_actions(state, apply_x_trick=apply_x_trick):
                candidate_states.append(apply_generator(state, generators[action]))
                candidate_paths.append(beam_paths[parent_idx] + [action])
                candidate_log_probs.append(float(parent_log_probs[parent_idx, action]))

        if not candidate_states:
            break

        candidate_states_array = np.stack(candidate_states, axis=0)
        candidate_log_probs_array = np.asarray(candidate_log_probs, dtype=np.float32)
        cand_t = torch.tensor(candidate_states_array, dtype=torch.long, device=device)
        with torch.no_grad():
            _, candidate_values_t = model(cand_t)
        candidate_values = candidate_values_t.cpu().numpy().astype(np.float32)
        candidate_scores = score_policy_guided_candidates(
            candidate_values,
            candidate_log_probs_array,
            policy_alpha=policy_alpha,
        )
        total_states_scored += candidate_states_array.shape[0]

        if visited is not None:
            for idx, state in enumerate(candidate_states_array):
                if tuple(state.tolist()) in visited:
                    candidate_scores[idx] = np.inf

        if history:
            recent = set(history)
            for idx, state in enumerate(candidate_states_array):
                state_key = tuple(state.tolist())
                if state_key != tuple(identity.tolist()) and state_key in recent:
                    candidate_scores[idx] = np.inf

        top_indices = np.argsort(candidate_scores)[:beam_width]
        beam_states = candidate_states_array[top_indices]
        beam_paths = [candidate_paths[idx] for idx in top_indices]

        for path, state in zip(beam_paths, beam_states):
            if np.array_equal(state, identity):
                model.train(was_training)
                return BeamSearchResult(
                    path_found=True,
                    path=path,
                    path_length=len(path),
                    steps_taken=step + 1,
                    states_scored=total_states_scored,
                    runtime_sec=time.perf_counter() - t0,
                )

        if visited is not None:
            for state in beam_states:
                visited.add(tuple(state.tolist()))

        if history_size > 0:
            history.extend(tuple(state.tolist()) for state in beam_states)
            if len(history) > history_size:
                history = history[-history_size:]

    model.train(was_training)
    return BeamSearchResult(
        path_found=False,
        path=None,
        path_length=0,
        steps_taken=step_limit,
        states_scored=total_states_scored,
        runtime_sec=time.perf_counter() - t0,
    )


def get_conjectured_longest_element(n: int) -> np.ndarray:
    """Return the paper's conjectured longest element ``(1 0 n-1 n-2 ... 3 2)``."""
    if n < 3:
        raise ValueError("n must be at least 3")
    tail = np.arange(n - 1, 1, -1, dtype=np.int64)
    return np.concatenate([np.array([1, 0], dtype=np.int64), tail])


def _default_bfs_results_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Junaid" / "lrx-traversal" / "lrx-traversal" / "bfs_results"


def _default_hard_benchmark_dir() -> Path:
    repo_root = Path(__file__).resolve().parent
    return repo_root / "benchmarks"


def build_hard_state_benchmark(
    n: int,
    *,
    benchmark_dir: Optional[Path | str] = None,
    bfs_results_dir: Optional[Path | str] = None,
) -> tuple[list[np.ndarray], dict[tuple[int, ...], float], str, str]:
    """Load exact longest elements when available, else fall back to the conjectured one."""
    benchmark_root = Path(benchmark_dir) if benchmark_dir is not None else _default_hard_benchmark_dir()
    benchmark_path = benchmark_root / f"koltsov3_n{n:02d}_hard_states.json"
    if benchmark_path.exists():
        data = json.loads(benchmark_path.read_text())
        states = [np.asarray(state, dtype=np.int64) for state in data["states"]]
        reference_lengths = {
            tuple(state.tolist()): float(length)
            for state, length in zip(states, data["reference_lengths"])
        }
        return states, reference_lengths, data.get("source", benchmark_path.stem), data.get(
            "reference_kind", "witness_length"
        )

    bfs_dir = Path(bfs_results_dir) if bfs_results_dir is not None else _default_bfs_results_dir()
    bfs_path = bfs_dir / f"koltsov3_bfs_n{n:02d}.json"

    if bfs_path.exists():
        data = json.loads(bfs_path.read_text())
        states = [np.asarray(state, dtype=np.int64) for state in data["longest_elements"]]
        optimal_length = float(data["diameter"])
        optimal_lengths = {tuple(state.tolist()): optimal_length for state in states}
        return states, optimal_lengths, f"bfs_longest_n{n:02d}", "optimal_length"

    conjectured = get_conjectured_longest_element(n)
    constructive_length = float(n * (n - 1) // 2)
    return (
        [conjectured],
        {tuple(conjectured.tolist()): constructive_length},
        "conjectured_longest_constructive",
        "constructive_length",
    )


def evaluate_hard_state_benchmark(
    model,
    config: PPOConfig,
    generators: np.ndarray,
    *,
    device: str,
) -> HardStateEvalSummary:
    """Evaluate the current search policy on the fixed hard-state benchmark."""
    states, reference_lengths, source, reference_kind = build_hard_state_benchmark(config.n)

    solved_path_ratios: list[float] = []
    solved_path_lengths: list[float] = []
    solved_runtimes: list[float] = []
    solved_count = 0

    for state in states:
        result = beam_search_with_policy_prior(
            state,
            model,
            generators,
            beam_width=config.eval_beam_width,
            step_limit=config.eval_step_limit,
            policy_alpha=config.eval_policy_alpha,
            device=device,
            history_size=config.eval_history_size,
            apply_x_trick=config.eval_apply_x_trick,
        )
        if result.path_found:
            solved_count += 1
            solved_runtimes.append(result.runtime_sec)
            solved_path_lengths.append(float(result.path_length))
            reference_length = reference_lengths.get(tuple(state.tolist()))
            if reference_length is not None and reference_length > 0:
                solved_path_ratios.append(result.path_length / float(reference_length))

    success_rate = solved_count / len(states)
    median_path_ratio = (
        float(np.median(solved_path_ratios)) if solved_path_ratios else float("inf")
    )
    median_path_length = (
        float(np.median(solved_path_lengths)) if solved_path_lengths else float("inf")
    )
    median_runtime = (
        float(np.median(solved_runtimes)) if solved_runtimes else float("inf")
    )
    return HardStateEvalSummary(
        success_rate=success_rate,
        median_path_ratio=median_path_ratio,
        reference_kind=reference_kind,
        median_path_length=median_path_length,
        median_runtime_sec=median_runtime,
        solved_count=solved_count,
        total_count=len(states),
        source=source,
        beam_width=config.eval_beam_width,
        step_limit=config.eval_step_limit,
        policy_alpha=config.eval_policy_alpha,
        apply_x_trick=config.eval_apply_x_trick,
    )


def save_checkpoint(
    path: str,
    *,
    model: ActorCritic,
    config: PPOConfig,
    update_idx: Optional[int] = None,
    hard_eval: Optional[HardStateEvalSummary] = None,
    reference_model: Optional[ActorCritic] = None,
) -> None:
    """Persist a training checkpoint with optional hard-state evaluation metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": config.__dict__,
    }
    if update_idx is not None:
        payload["update_idx"] = update_idx
    if hard_eval is not None:
        payload["hard_eval"] = asdict(hard_eval)
    if reference_model is not None:
        payload["reference_state_dict"] = reference_model.state_dict()
    torch.save(payload, checkpoint_path)


def run_value_bootstrap(
    model: ActorCritic,
    generators: np.ndarray,
    config: PPOConfig,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Warm-start the shared trunk/value head before PPO fine-tuning."""
    if config.warmup_epochs <= 0 and config.bellman_epochs <= 0:
        return None

    rng = np.random.default_rng(config.seed)
    walk_states, walk_depths = generate_nonbacktracking_walk_dataset(
        generators,
        num_walks=config.warmup_walks,
        walk_length=config.warmup_walk_length,
        rng=rng,
        history_size=config.warmup_history_size,
    )
    print(
        f"value_bootstrap_dataset size={walk_states.shape[0]} "
        f"walk_length={config.warmup_walk_length} history={config.warmup_history_size}"
    )

    value_optimizer = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.value_head.parameters()),
        lr=config.value_lr,
    )

    _run_value_regression_epochs(
        model,
        walk_states,
        walk_depths,
        device=device,
        optimizer=value_optimizer,
        epochs=config.warmup_epochs,
        batch_size=config.warmup_batch_size,
        grad_clip_norm=config.grad_clip_norm,
        log_prefix="warmup",
    )

    for bellman_epoch in range(1, config.bellman_epochs + 1):
        bellman_targets = compute_bellman_clipped_targets(
            model,
            walk_states,
            generators,
            walk_depths,
            device=device,
            batch_size=config.bellman_batch_size,
        )
        _run_value_regression_epochs(
            model,
            walk_states,
            bellman_targets,
            device=device,
            optimizer=value_optimizer,
            epochs=1,
            batch_size=config.warmup_batch_size,
            grad_clip_norm=config.grad_clip_norm,
            log_prefix=f"bellman[{bellman_epoch:03d}/{config.bellman_epochs:03d}]",
        )

    return walk_states, walk_depths


def run_policy_warmstart(
    model: ActorCritic,
    states: np.ndarray,
    generators: np.ndarray,
    config: PPOConfig,
    *,
    device: str,
) -> None:
    """Warm-start the policy head from Bellman-best successor actions."""
    if config.policy_warmstart_epochs <= 0:
        return

    identity_mask = np.all(states == np.arange(states.shape[1], dtype=np.int64), axis=1)
    train_states = states[~identity_mask]
    if train_states.shape[0] == 0:
        return

    action_targets = compute_policy_imitation_targets(
        model,
        train_states,
        generators,
        device=device,
        batch_size=config.policy_warmstart_batch_size,
    )
    policy_optimizer = torch.optim.Adam(
        model.policy_head.parameters(),
        lr=config.policy_head_lr,
    )
    _run_policy_warmstart_epochs(
        model,
        train_states,
        action_targets,
        device=device,
        optimizer=policy_optimizer,
        epochs=config.policy_warmstart_epochs,
        batch_size=config.policy_warmstart_batch_size,
        grad_clip_norm=config.grad_clip_norm,
        log_prefix="policy_warmstart",
    )


def run_offpolicy_heuristic_refresh(
    model: ActorCritic,
    heuristic_teacher_model: ActorCritic,
    generators: np.ndarray,
    config: PPOConfig,
    *,
    device: str,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    update_idx: int,
    total_updates: int,
) -> None:
    """Run DQN-style Bellman refinement on the same deep walk distribution used in warm-up."""
    if config.offpolicy_heuristic_refresh_epochs <= 0:
        return

    refresh_states, _ = generate_nonbacktracking_walk_dataset(
        generators,
        num_walks=config.warmup_walks,
        walk_length=config.warmup_walk_length,
        rng=rng,
        history_size=config.warmup_history_size,
    )
    refresh_targets = compute_reference_heuristic_targets(
        heuristic_teacher_model,
        refresh_states,
        generators,
        device=device,
        batch_size=config.bellman_batch_size,
    )
    _run_value_regression_epochs(
        model,
        refresh_states,
        refresh_targets,
        device=device,
        optimizer=optimizer,
        epochs=config.offpolicy_heuristic_refresh_epochs,
        batch_size=config.warmup_batch_size,
        grad_clip_norm=config.grad_clip_norm,
        log_prefix=(
            f"offpolicy_refresh[{update_idx:04d}/{total_updates:04d}]"
        ),
    )


def train_ppo(config: PPOConfig) -> ActorCritic:
    """Train a PPO policy on Koltsov3 traversal with potential shaping and hard-state eval."""
    _require_torch()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = _select_device(config.device)
    generators = get_koltsov3_generators(config.n, config.k)
    identity = np.arange(config.n, dtype=np.int64)

    model = ActorCritic(config.n, config.hidden_dim, k=config.k).to(device)
    bootstrap_dataset = run_value_bootstrap(model, generators, config, device=device)
    if bootstrap_dataset is None and config.policy_warmstart_epochs > 0:
        rng = np.random.default_rng(config.seed)
        bootstrap_dataset = generate_nonbacktracking_walk_dataset(
            generators,
            num_walks=config.warmup_walks,
            walk_length=config.warmup_walk_length,
            rng=rng,
            history_size=config.warmup_history_size,
        )
    if bootstrap_dataset is not None:
        bootstrap_states, _ = bootstrap_dataset
        run_policy_warmstart(model, bootstrap_states, generators, config, device=device)
    reward_reference_model = ActorCritic(config.n, config.hidden_dim, k=config.k).to(device)
    heuristic_teacher_model = ActorCritic(config.n, config.hidden_dim, k=config.k).to(device)
    sync_reference_model(reward_reference_model, model)
    sync_reference_model(heuristic_teacher_model, model)

    optimizer = build_ppo_optimizer(model, config)
    auxiliary_value_optimizer = None
    if config.offpolicy_heuristic_refresh_updates > 0:
        auxiliary_value_optimizer = torch.optim.Adam(
            list(model.trunk.parameters()) + list(model.value_head.parameters()),
            lr=config.value_lr,
        )
    offpolicy_rng = np.random.default_rng(config.seed + 10_003)

    state = identity.copy()
    episode_step = 0
    updates = max(1, config.total_steps // config.rollout_steps)
    best_hard_eval: Optional[HardStateEvalSummary] = None

    for update_idx in range(1, updates + 1):
        anneal_ppo_learning_rates(
            optimizer,
            config,
            update_idx=update_idx,
            total_updates=updates,
        )

        obs_buf = np.zeros((config.rollout_steps, config.n), dtype=np.int64)
        act_buf = np.zeros(config.rollout_steps, dtype=np.int64)
        logp_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        rew_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        done_buf = np.zeros(config.rollout_steps, dtype=bool)
        val_buf = np.zeros(config.rollout_steps, dtype=np.float32)

        for t in range(config.rollout_steps):
            obs_buf[t] = state
            obs_t = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits_t, _, critic_t = model.actor_critic(obs_t)
                _, ref_prev_value_t = reward_reference_model(obs_t)
                dist_t = torch.distributions.Categorical(logits=logits_t)
                action_t = dist_t.sample()
                logp_t = dist_t.log_prob(action_t)

            action = int(action_t.item())
            next_state = apply_generator(state, generators[action])
            done = bool(np.array_equal(next_state, identity))

            next_obs_t = torch.tensor(next_state, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                _, ref_next_value_t = reward_reference_model(next_obs_t)

            prev_potential = float(ref_prev_value_t.item())
            next_potential = 0.0 if done else float(ref_next_value_t.item())
            reward = compute_potential_shaping_reward(
                prev_potential,
                next_potential,
                step_penalty=config.step_penalty,
                success_bonus=config.success_bonus,
                done=done,
            )

            episode_step += 1
            truncated = episode_step >= config.max_episode_steps

            if done or truncated:
                scramble_steps = np.random.randint(1, config.max_scramble_steps + 1)
                new_state = identity.copy()
                for _ in range(scramble_steps):
                    a = np.random.randint(0, 3)
                    new_state = apply_generator(new_state, generators[a])
                state = new_state
                episode_step = 0
            else:
                state = next_state

            act_buf[t] = action
            logp_buf[t] = float(logp_t.item())
            rew_buf[t] = reward
            done_buf[t] = bool(done or truncated)
            val_buf[t] = float(critic_t.item())

        with torch.no_grad():
            next_obs_t = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
            _, _, next_value_t = model.actor_critic(next_obs_t)
            last_value = float(next_value_t.item())

        adv_buf, ret_buf = compute_gae(
            rewards=rew_buf,
            values=val_buf,
            dones=done_buf,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        obs_t = torch.tensor(obs_buf, dtype=torch.long, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)
        heuristic_target_buf = compute_reference_heuristic_targets(
            heuristic_teacher_model,
            obs_buf,
            generators,
            device=device,
            batch_size=config.bellman_batch_size,
        )
        heuristic_target_t = torch.tensor(
            heuristic_target_buf,
            dtype=torch.float32,
            device=device,
        )

        n_steps = config.rollout_steps
        batch_policy_losses: list[float] = []
        batch_critic_losses: list[float] = []
        batch_heuristic_losses: list[float] = []
        batch_entropies: list[float] = []
        batch_approx_kls: list[float] = []
        batch_clipfracs: list[float] = []
        ppo_epochs_ran = 0
        early_stop_kl: Optional[float] = None

        for epoch_idx in range(1, config.update_epochs + 1):
            ppo_epochs_ran = epoch_idx
            step_indices = np.random.permutation(n_steps)
            for start in range(0, n_steps, config.minibatch_size):
                mb_indices = step_indices[start : start + config.minibatch_size]
                mb_obs = obs_t[mb_indices]
                mb_act = act_t[mb_indices]
                mb_old_logp = old_logp_t[mb_indices]
                mb_adv = adv_t[mb_indices]
                mb_ret = ret_t[mb_indices]
                mb_heuristic_target = heuristic_target_t[mb_indices]

                logits, heuristic_values, critic_values = model.actor_critic(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(log_ratio)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                critic_loss = nn.functional.mse_loss(critic_values, mb_ret)
                heuristic_loss = nn.functional.mse_loss(heuristic_values, mb_heuristic_target)
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config.clip_coef).to(torch.float32).mean()

                batch_policy_losses.append(float(policy_loss.item()))
                batch_critic_losses.append(float(critic_loss.item()))
                batch_heuristic_losses.append(float(heuristic_loss.item()))
                batch_entropies.append(float(entropy.item()))
                batch_approx_kls.append(float(approx_kl.item()))
                batch_clipfracs.append(float(clipfrac.item()))

                if config.target_kl is not None and float(approx_kl.item()) > config.target_kl:
                    early_stop_kl = float(approx_kl.item())
                    break

                loss = (
                    policy_loss
                    + config.value_coef * critic_loss
                    + config.heuristic_coef * heuristic_loss
                    - config.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

            if early_stop_kl is not None:
                break

        if update_idx % config.log_every_updates == 0:
            ret_var = np.var(ret_buf)
            explained_var = (
                0.0 if ret_var < 1e-8 else 1.0 - np.var(ret_buf - val_buf) / ret_var
            )
            print(
                f"update={update_idx:04d}/{updates:04d} "
                f"mean_reward={rew_buf.mean():+.4f} "
                f"mean_return={ret_buf.mean():+.4f} "
                f"explained_var={explained_var:+.4f} "
                f"policy_loss={np.mean(batch_policy_losses):+.4f} "
                f"critic_loss={np.mean(batch_critic_losses):.4f} "
                f"heuristic_loss={np.mean(batch_heuristic_losses):.4f} "
                f"entropy={np.mean(batch_entropies):.4f} "
                f"approx_kl={np.mean(batch_approx_kls):.4f} "
                f"clipfrac={np.mean(batch_clipfracs):.4f} "
                f"ppo_epochs={ppo_epochs_ran:02d}/{config.update_epochs:02d}"
            )
            if early_stop_kl is not None:
                print(
                    f"ppo_early_stop update={update_idx:04d}/{updates:04d} "
                    f"approx_kl={early_stop_kl:.4f} target_kl={config.target_kl:.4f}"
                )

        teacher_refreshed = False
        if (
            config.heuristic_teacher_refresh_updates > 0
            and update_idx % config.heuristic_teacher_refresh_updates == 0
            and update_idx != updates
        ):
            sync_reference_model(heuristic_teacher_model, model)
            teacher_refreshed = True
            print(
                f"heuristic_teacher_refresh update={update_idx:04d}/{updates:04d} "
                f"interval={config.heuristic_teacher_refresh_updates}"
            )

        if (
            config.offpolicy_heuristic_refresh_updates > 0
            and update_idx % config.offpolicy_heuristic_refresh_updates == 0
            and auxiliary_value_optimizer is not None
        ):
            if not teacher_refreshed:
                sync_reference_model(heuristic_teacher_model, model)
                print(
                    f"heuristic_teacher_refresh update={update_idx:04d}/{updates:04d} "
                    f"interval=offpolicy"
                )
            print(
                f"offpolicy_refresh_start update={update_idx:04d}/{updates:04d} "
                f"walks={config.warmup_walks} walk_length={config.warmup_walk_length}"
            )
            run_offpolicy_heuristic_refresh(
                model,
                heuristic_teacher_model,
                generators,
                config,
                device=device,
                rng=offpolicy_rng,
                optimizer=auxiliary_value_optimizer,
                update_idx=update_idx,
                total_updates=updates,
            )

        should_eval = (
            config.eval_every_updates > 0
            and (update_idx % config.eval_every_updates == 0 or update_idx == updates)
        )
        if should_eval:
            hard_eval = evaluate_hard_state_benchmark(
                model,
                config,
                generators,
                device=device,
            )
            print(
                f"hard_eval update={update_idx:04d}/{updates:04d} "
                f"source={hard_eval.source} "
                f"reference={hard_eval.reference_kind} "
                f"success={hard_eval.solved_count}/{hard_eval.total_count} "
                f"success_rate={hard_eval.success_rate:.3f} "
                f"median_path_ratio={hard_eval.median_path_ratio:.3f} "
                f"median_path_len={hard_eval.median_path_length:.1f} "
                f"median_runtime={hard_eval.median_runtime_sec:.3f}"
            )
            if (
                config.checkpoint_path
                and (best_hard_eval is None or hard_eval.comparison_key() > best_hard_eval.comparison_key())
            ):
                save_checkpoint(
                    config.checkpoint_path,
                    model=model,
                    config=config,
                    update_idx=update_idx,
                    hard_eval=hard_eval,
                    reference_model=reward_reference_model,
                )
                best_hard_eval = hard_eval
                print(f"Saved best hard-eval checkpoint to {config.checkpoint_path}")

    if config.checkpoint_path and config.eval_every_updates <= 0:
        save_checkpoint(
            config.checkpoint_path,
            model=model,
            config=config,
            update_idx=updates,
            reference_model=reward_reference_model,
        )
        print(f"Saved checkpoint to {config.checkpoint_path}")

    model.eval()
    return model


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Koltsov3 PPO starter implementation")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--max-scramble-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--heuristic-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--anneal-learning-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--heuristic-teacher-refresh-updates", type=int, default=0)
    parser.add_argument("--offpolicy-heuristic-refresh-updates", type=int, default=0)
    parser.add_argument("--offpolicy-heuristic-refresh-epochs", type=int, default=1)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-walks", type=int, default=256)
    parser.add_argument("--warmup-walk-length", type=int, default=None)
    parser.add_argument("--warmup-batch-size", type=int, default=512)
    parser.add_argument("--warmup-history-size", type=int, default=32)
    parser.add_argument("--bellman-epochs", type=int, default=0)
    parser.add_argument("--bellman-batch-size", type=int, default=512)
    parser.add_argument("--policy-warmstart-epochs", type=int, default=0)
    parser.add_argument("--policy-warmstart-batch-size", type=int, default=512)
    parser.add_argument("--policy-head-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--log-every-updates", type=int, default=10)
    parser.add_argument("--eval-every-updates", type=int, default=0)
    parser.add_argument("--eval-beam-width", type=int, default=64)
    parser.add_argument("--eval-step-limit", type=int, default=None)
    parser.add_argument("--eval-history-size", type=int, default=32)
    parser.add_argument("--eval-policy-alpha", type=float, default=1.0)
    parser.add_argument(
        "--eval-apply-x-trick",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    return PPOConfig(
        n=args.n,
        k=args.k,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        max_episode_steps=args.max_episode_steps,
        max_scramble_steps=args.max_scramble_steps,
        hidden_dim=args.hidden_dim,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        value_coef=args.value_coef,
        heuristic_coef=args.heuristic_coef,
        entropy_coef=args.entropy_coef,
        clip_coef=args.clip_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        grad_clip_norm=args.grad_clip_norm,
        target_kl=args.target_kl,
        anneal_learning_rate=args.anneal_learning_rate,
        heuristic_teacher_refresh_updates=args.heuristic_teacher_refresh_updates,
        offpolicy_heuristic_refresh_updates=args.offpolicy_heuristic_refresh_updates,
        offpolicy_heuristic_refresh_epochs=args.offpolicy_heuristic_refresh_epochs,
        success_bonus=args.success_bonus,
        step_penalty=args.step_penalty,
        warmup_epochs=args.warmup_epochs,
        warmup_walks=args.warmup_walks,
        warmup_walk_length=args.warmup_walk_length,
        warmup_batch_size=args.warmup_batch_size,
        warmup_history_size=args.warmup_history_size,
        bellman_epochs=args.bellman_epochs,
        bellman_batch_size=args.bellman_batch_size,
        policy_warmstart_epochs=args.policy_warmstart_epochs,
        policy_warmstart_batch_size=args.policy_warmstart_batch_size,
        policy_head_lr=args.policy_head_lr,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        log_every_updates=args.log_every_updates,
        eval_every_updates=args.eval_every_updates,
        eval_beam_width=args.eval_beam_width,
        eval_step_limit=args.eval_step_limit,
        eval_history_size=args.eval_history_size,
        eval_policy_alpha=args.eval_policy_alpha,
        eval_apply_x_trick=args.eval_apply_x_trick,
    )


def main() -> None:
    cfg = parse_args()
    train_ppo(cfg)


if __name__ == "__main__":
    main()
