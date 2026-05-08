#!/usr/bin/env python3
"""Starter PPO implementation for Koltsov3 permutation traversal.

This script is a standalone baseline intended to begin replacing the RL section
of the LRX workflow with PPO while using Koltsov3 generators.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module as ActorCriticBase
else:
    Tensor = Any

try:
    import torch as _torch
    import torch.nn as _nn
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    _torch = None
    _nn = None

if not TYPE_CHECKING:
    ActorCriticBase = _nn.Module if _nn is not None else object


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
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_coef: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 256
    grad_clip_norm: float = 0.5
    success_bonus: float = 5.0
    seed: int = 42
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    log_every_updates: int = 10

    def __post_init__(self) -> None:
        if self.max_episode_steps is None:
            self.max_episode_steps = 4 * self.n
        if self.max_scramble_steps is None:
            self.max_scramble_steps = 3 * self.n
        if self.rollout_steps % self.minibatch_size != 0:
            raise ValueError(
                "rollout_steps must be divisible by minibatch_size for stable minibatches"
            )


class ActorCritic(ActorCriticBase):
    """One-hot permutation encoder + shared trunk + policy/value heads."""

    def __init__(self, n: int, hidden_dim: int):
        _, nn = _require_torch()
        super().__init__()
        self.n = n
        input_dim = n * n
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode(self, obs: Tensor) -> Tensor:
        torch, nn = _require_torch()
        x = nn.functional.one_hot(obs, num_classes=self.n).to(torch.float32)
        return x.flatten(start_dim=-2)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        x = self._encode(obs)
        h = self.trunk(x)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values


def _require_torch() -> tuple[Any, Any]:
    if _torch is None or _nn is None:
        raise ImportError(
            "PyTorch is required for PPO training. Install dependencies first."
        )
    return _torch, _nn


def _select_device(device: str) -> str:
    torch, _ = _require_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train_ppo(config: PPOConfig) -> ActorCritic:
    """Train a PPO policy on Koltsov3 traversal with dense distance shaping."""
    torch, nn = _require_torch()
    max_episode_steps = config.max_episode_steps
    max_scramble_steps = config.max_scramble_steps
    if max_episode_steps is None or max_scramble_steps is None:
        raise ValueError("PPOConfig step limits must be initialized before training")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = _select_device(config.device)
    generators = get_koltsov3_generators(config.n, config.k)
    identity = np.arange(config.n, dtype=np.int64)

    model = ActorCritic(config.n, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.policy_lr)

    state = identity.copy()
    episode_step = 0
    updates = max(1, config.total_steps // config.rollout_steps)

    for update_idx in range(1, updates + 1):
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
                logits_t, value_t = model(obs_t)
                dist_t = torch.distributions.Categorical(logits=logits_t)
                action_t = dist_t.sample()
                logp_t = dist_t.log_prob(action_t)

            action = int(action_t.item())
            prev_dist = distance_to_identity(state)
            next_state = apply_generator(state, generators[action])
            next_dist = distance_to_identity(next_state)

            reward = float(prev_dist - next_dist)
            done = next_dist == 0

            episode_step += 1
            truncated = episode_step >= max_episode_steps
            if done:
                reward += config.success_bonus

            if done or truncated:
                scramble_steps = np.random.randint(1, max_scramble_steps + 1)
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
            val_buf[t] = float(value_t.item())

        with torch.no_grad():
            next_obs_t = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
            _, next_value_t = model(next_obs_t)
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

        n_steps = config.rollout_steps
        for _ in range(config.update_epochs):
            step_indices = np.random.permutation(n_steps)
            for start in range(0, n_steps, config.minibatch_size):
                mb_indices = step_indices[start : start + config.minibatch_size]
                mb_obs = obs_t[mb_indices]
                mb_act = act_t[mb_indices]
                mb_old_logp = old_logp_t[mb_indices]
                mb_adv = adv_t[mb_indices]
                mb_ret = ret_t[mb_indices]

                logits, values = model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = nn.functional.mse_loss(values, mb_ret)

                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

        if update_idx % config.log_every_updates == 0:
            ret_var = np.var(ret_buf)
            explained_var = (
                0.0 if ret_var < 1e-8 else 1.0 - np.var(ret_buf - val_buf) / ret_var
            )
            print(
                f"update={update_idx:04d}/{updates:04d} "
                f"mean_reward={rew_buf.mean():+.4f} "
                f"mean_return={ret_buf.mean():+.4f} "
                f"explained_var={explained_var:+.4f}"
            )

    if config.checkpoint_path:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config.__dict__,
            },
            config.checkpoint_path,
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
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--log-every-updates", type=int, default=10)
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
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        clip_coef=args.clip_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        grad_clip_norm=args.grad_clip_norm,
        success_bonus=args.success_bonus,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        log_every_updates=args.log_every_updates,
    )


def main() -> None:
    cfg = parse_args()
    train_ppo(cfg)


if __name__ == "__main__":
    main()
