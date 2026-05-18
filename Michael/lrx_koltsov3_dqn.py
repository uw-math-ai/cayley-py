#!/usr/bin/env python3
"""Modified DQN-style Koltsov3 value baseline with matched benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Michael.lrx_koltsov3_ppo import (
    ActorCritic,
    _run_value_regression_epochs,
    compute_bellman_clipped_targets,
    evaluate_hard_state_benchmark,
    generate_nonbacktracking_walk_dataset,
    get_koltsov3_generators,
    predict_heuristic_values,
)


@dataclass
class DQNConfig:
    n: int = 16
    k: int = 0
    hidden_dim: int = 512
    value_lr: float = 1e-3
    grad_clip_norm: float = 0.5
    warmup_epochs: int = 30
    warmup_walks: int = 4096
    warmup_walk_length: Optional[int] = None
    warmup_batch_size: int = 1024
    warmup_history_size: int = 32
    dqn_epochs: int = 40
    dqn_batch_size: int = 1024
    regenerate_walks_each_epoch: bool = True
    seed: int = 42
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    log_every_epochs: int = 4
    eval_every_epochs: int = 4
    eval_beam_width: int = 256
    eval_step_limit: Optional[int] = None
    eval_history_size: int = 32
    eval_policy_alpha: float = 0.0
    eval_apply_x_trick: bool = True

    def __post_init__(self) -> None:
        if self.warmup_walk_length is None:
            self.warmup_walk_length = 8 * self.n
        if self.eval_step_limit is None:
            self.eval_step_limit = 2 * max(1, self.n * (self.n - 1) // 2)
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.dqn_epochs < 0:
            raise ValueError("dqn_epochs must be non-negative")
        if self.warmup_walks <= 0:
            raise ValueError("warmup_walks must be positive")
        if self.warmup_walk_length <= 0:
            raise ValueError("warmup_walk_length must be positive")
        if self.warmup_batch_size <= 0:
            raise ValueError("warmup_batch_size must be positive")
        if self.dqn_batch_size <= 0:
            raise ValueError("dqn_batch_size must be positive")
        if self.value_lr <= 0.0:
            raise ValueError("value_lr must be positive")


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _make_training_dataset(
    config: DQNConfig,
    generators: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    return generate_nonbacktracking_walk_dataset(
        generators,
        num_walks=config.warmup_walks,
        walk_length=config.warmup_walk_length,
        rng=rng,
        history_size=config.warmup_history_size,
    )


def _save_dqn_checkpoint(
    path: str,
    *,
    model: ActorCritic,
    config: DQNConfig,
    epoch_idx: Optional[int] = None,
    hard_eval: Optional[object] = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "algorithm": "dqn",
        "state_dict": model.state_dict(),
        "config": asdict(config),
    }
    if epoch_idx is not None:
        payload["epoch_idx"] = epoch_idx
    if hard_eval is not None:
        payload["hard_eval"] = asdict(hard_eval)
    torch.save(payload, checkpoint_path)


def train_dqn(config: DQNConfig) -> ActorCritic:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = _select_device(config.device)
    generators = get_koltsov3_generators(config.n, config.k)
    rng = np.random.default_rng(config.seed)

    model = ActorCritic(config.n, config.hidden_dim, k=config.k).to(device)
    for parameter in model.policy_head.parameters():
        parameter.requires_grad_(False)
    for parameter in model.critic_head.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.value_head.parameters()),
        lr=config.value_lr,
    )

    walk_states, walk_depths = _make_training_dataset(config, generators, rng)
    print(
        f"dqn_dataset size={walk_states.shape[0]} "
        f"walk_length={config.warmup_walk_length} history={config.warmup_history_size}"
    )

    _run_value_regression_epochs(
        model,
        walk_states,
        walk_depths,
        device=device,
        optimizer=optimizer,
        epochs=config.warmup_epochs,
        batch_size=config.warmup_batch_size,
        grad_clip_norm=config.grad_clip_norm,
        log_prefix="dqn_warmup",
    )

    best_hard_eval = None

    for epoch_idx in range(1, config.dqn_epochs + 1):
        if epoch_idx > 1 and config.regenerate_walks_each_epoch:
            walk_states, walk_depths = _make_training_dataset(config, generators, rng)

        bellman_targets = compute_bellman_clipped_targets(
            model,
            walk_states,
            generators,
            walk_depths,
            device=device,
            batch_size=config.dqn_batch_size,
        )
        _run_value_regression_epochs(
            model,
            walk_states,
            bellman_targets,
            device=device,
            optimizer=optimizer,
            epochs=1,
            batch_size=config.dqn_batch_size,
            grad_clip_norm=config.grad_clip_norm,
            log_prefix=f"dqn[{epoch_idx:03d}/{config.dqn_epochs:03d}]",
        )

        should_log = config.log_every_epochs > 0 and (
            epoch_idx % config.log_every_epochs == 0 or epoch_idx == config.dqn_epochs
        )
        if should_log:
            predicted_values = predict_heuristic_values(
                model,
                walk_states,
                device=device,
                batch_size=config.dqn_batch_size,
            )
            dataset_mse = float(np.mean((predicted_values - bellman_targets) ** 2))
            print(
                f"dqn_epoch={epoch_idx:03d}/{config.dqn_epochs:03d} "
                f"target_mean={bellman_targets.mean():.4f} "
                f"pred_mean={predicted_values.mean():.4f} "
                f"dataset_mse={dataset_mse:.6f}"
            )

        should_eval = (
            config.eval_every_epochs > 0
            and (epoch_idx % config.eval_every_epochs == 0 or epoch_idx == config.dqn_epochs)
        )
        if should_eval:
            hard_eval = evaluate_hard_state_benchmark(
                model,
                config,
                generators,
                device=device,
            )
            print(
                f"hard_eval epoch={epoch_idx:03d}/{config.dqn_epochs:03d} "
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
                _save_dqn_checkpoint(
                    config.checkpoint_path,
                    model=model,
                    config=config,
                    epoch_idx=epoch_idx,
                    hard_eval=hard_eval,
                )
                best_hard_eval = hard_eval
                print(f"Saved best hard-eval checkpoint to {config.checkpoint_path}")

    if config.checkpoint_path and config.eval_every_epochs <= 0:
        _save_dqn_checkpoint(
            config.checkpoint_path,
            model=model,
            config=config,
            epoch_idx=config.dqn_epochs,
        )
        print(f"Saved checkpoint to {config.checkpoint_path}")

    model.eval()
    return model


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description="Koltsov3 modified DQN baseline")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--warmup-walks", type=int, default=4096)
    parser.add_argument("--warmup-walk-length", type=int, default=None)
    parser.add_argument("--warmup-batch-size", type=int, default=1024)
    parser.add_argument("--warmup-history-size", type=int, default=32)
    parser.add_argument("--dqn-epochs", type=int, default=40)
    parser.add_argument("--dqn-batch-size", type=int, default=1024)
    parser.add_argument(
        "--regenerate-walks-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--log-every-epochs", type=int, default=4)
    parser.add_argument("--eval-every-epochs", type=int, default=4)
    parser.add_argument("--eval-beam-width", type=int, default=256)
    parser.add_argument("--eval-step-limit", type=int, default=None)
    parser.add_argument("--eval-history-size", type=int, default=32)
    parser.add_argument("--eval-policy-alpha", type=float, default=0.0)
    parser.add_argument(
        "--eval-apply-x-trick",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return DQNConfig(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    train_dqn(config)


if __name__ == "__main__":
    main()