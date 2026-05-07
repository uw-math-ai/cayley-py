#!/usr/bin/env python3
"""General Koltsov3 random-walk MLP sweep script for Hyak/Slurm.

This is a cleaned-up, generalized version of koltsov3_sweep.py. It preserves
core experiment choices from the original script:
  - Koltsov3 generator construction on permutations 0..n-1
  - random-walk labels equal to normalized random-walk depth
  - fixed validation/test random walks for each configuration
  - regenerated training random walks at each epoch
  - one MLP trained for each sweep configuration

Major additions:
  - Cartesian-product sweeps over n, width, random-walk count, walk-length
    multiplier, random-walk type, steps-back-to-ban, epochs, lr, batch size,
    validation/test sample counts, and seed
  - summary and epoch-level CSV outputs
  - compact but useful diagnostics about data uniqueness, labels, model size,
    RMSE gaps, best epochs, and timing
  - optional W&B logging
  - optional exact BFS metadata for small n only

Example smoke test:
    python koltsov3_general_sweep.py \
      --n-values 5 \
      --widths 8 \
      --n-random-walks-values 20 \
      --walk-length-multipliers 4 \
      --random-walk-types non-backtracking-beam \
      --steps-back-to-ban-values 2 \
      --n-epochs-values 2 \
      --lr-values 0.001 \
      --batch-size-values 8 \
      --n-val-samples-values 10 \
      --n-test-samples-values 10 \
      --seed-values 0 \
      --output-dir smoke_test \
      --use-wandb false
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Required on Slurm/batch nodes where no display is available.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy import stats
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:
    wandb = None


# -----------------------------
# Argument parsing helpers
# -----------------------------


def parse_int_list(value: str) -> List[int]:
    if not value:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("List cannot be empty.")
    return parsed


def parse_float_list(value: str) -> List[float]:
    if not value:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of floats.")
    try:
        parsed = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("List cannot be empty.")
    return parsed


def parse_str_list(value: str) -> List[str]:
    parsed = [x.strip() for x in value.split(",") if x.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("List cannot be empty.")
    return parsed


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# -----------------------------
# Reproducibility/device helpers
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_arg)


# -----------------------------
# Koltsov3/random-walk logic
# -----------------------------


def get_koltsov3_moves(n: int, k: int = 0) -> List[np.ndarray]:
    """Koltsov3 generators on {0, 1, ..., n-1}.

    This preserves the generator logic from the original script:
      I swaps adjacent pairs starting at 0: (0 1), (2 3), ...
      K swaps adjacent pairs starting at 1: (1 2), (3 4), ...
      S swaps k and k+2
    """
    if n < 3:
        raise ValueError("Koltsov3 with S=(k,k+2) requires n >= 3.")
    if not (0 <= k <= n - 3):
        raise ValueError(f"k must satisfy 0 <= k <= n-3, got k={k} for n={n}.")

    I = np.arange(n)
    K = np.arange(n)
    S = np.arange(n)

    for i in range(0, n - 1, 2):
        I[i], I[i + 1] = I[i + 1], I[i]

    for i in range(1, n - 1, 2):
        K[i], K[i + 1] = K[i + 1], K[i]

    S[k], S[k + 2] = S[k + 2], S[k]
    return [I, K, S]


def get_random_walk_length(n: int, walk_length_multiplier: int) -> int:
    return walk_length_multiplier * n


def build_problem(
    n: int,
    koltsov3_k: int,
    device: torch.device,
) -> Tuple[List[np.ndarray], torch.Tensor, torch.dtype]:
    generators = get_koltsov3_moves(n, k=koltsov3_k)
    identity_state = torch.arange(n, device=device, dtype=torch.int64)
    dtype_state = torch.int64
    return generators, identity_state, dtype_state


def normalize_generators(generators: Sequence[np.ndarray] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(generators, torch.Tensor):
        return generators.to(device=device, dtype=torch.long)
    return torch.tensor(np.array(generators), dtype=torch.long, device=device)


def sample_allowed_moves(allowed: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sample one allowed generator per row.

    This keeps the original behavior easy to inspect. It is not the fastest
    possible implementation, so very large random-walk counts may become slow.
    """
    n_rows = allowed.shape[0]
    move_ids = torch.zeros(n_rows, dtype=torch.long, device=device)
    for i in range(n_rows):
        choices = torch.where(allowed[i])[0]
        move_ids[i] = choices[torch.randint(len(choices), (1,), device=device)]
    return move_ids


def random_walks(
    generators: Sequence[np.ndarray] | torch.Tensor,
    n_random_walk_length: int,
    n_random_walks_to_generate: int,
    n_random_walks_steps_back_to_ban: int,
    random_walks_type: str,
    state_rw_start: torch.Tensor,
    dtype_state: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random-walk states and normalized depth labels.

    Returns all intermediate states, not only final states. For walk length L,
    labels are 1/L, 2/L, ..., 1.0 for each generated walk.
    """
    moves = normalize_generators(generators, device)
    n_generators = moves.shape[0]

    states = state_rw_start.unsqueeze(0).repeat(n_random_walks_to_generate, 1).to(dtype_state)
    all_states: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    if random_walks_type == "simple":
        for step in range(1, n_random_walk_length + 1):
            move_ids = torch.randint(0, n_generators, (n_random_walks_to_generate,), device=device)
            states = torch.gather(states, 1, moves[move_ids])
            all_states.append(states.clone())
            all_y.append(torch.full((n_random_walks_to_generate,), step / n_random_walk_length, device=device))

    elif random_walks_type == "non-backtracking-beam":
        move_history: List[torch.Tensor] = []
        for step in range(1, n_random_walk_length + 1):
            allowed = torch.ones((n_random_walks_to_generate, n_generators), dtype=torch.bool, device=device)

            if len(move_history) > 0 and n_random_walks_steps_back_to_ban > 0:
                for prev_moves in move_history[-n_random_walks_steps_back_to_ban:]:
                    allowed.scatter_(1, prev_moves.unsqueeze(1), False)

            # If every generator was banned for a row, fall back to allowing all.
            fallback_rows = ~allowed.any(dim=1)
            if fallback_rows.any():
                allowed[fallback_rows, :] = True

            move_ids = sample_allowed_moves(allowed, device)
            move_history.append(move_ids.clone())
            states = torch.gather(states, 1, moves[move_ids])
            all_states.append(states.clone())
            all_y.append(torch.full((n_random_walks_to_generate,), step / n_random_walk_length, device=device))

    else:
        raise ValueError(f"Unknown random_walks_type: {random_walks_type}")

    return torch.cat(all_states, dim=0), torch.cat(all_y, dim=0).float()


# -----------------------------
# Optional exact BFS metadata
# -----------------------------


def compose_state_with_generator(state: Tuple[int, ...], generator: np.ndarray) -> Tuple[int, ...]:
    """Apply the same index-gather action used by torch.gather in random_walks."""
    return tuple(state[int(i)] for i in generator)


def compute_exact_bfs_metadata(
    n: int,
    generators: Sequence[np.ndarray],
    max_states: int,
) -> Dict[str, Any]:
    """Compute exact BFS diameter/layer sizes from the identity, if small enough.

    This can grow as n!, so it is disabled by default. The max_states guard is
    intended to prevent accidental large exact BFS runs on Hyak.
    """
    total_possible_states = math.factorial(n)
    if total_possible_states > max_states:
        return {
            "bfs_computed": False,
            "bfs_skip_reason": f"n!={total_possible_states} exceeds max_bfs_states={max_states}",
            "diameter": np.nan,
            "last_layer_count": np.nan,
            "layer_sizes": "",
        }

    start = tuple(range(n))
    visited = {start: 0}
    q: deque[Tuple[int, ...]] = deque([start])
    layer_counts: Dict[int, int] = {0: 1}

    while q:
        state = q.popleft()
        depth = visited[state]
        for gen in generators:
            nxt = compose_state_with_generator(state, gen)
            if nxt not in visited:
                visited[nxt] = depth + 1
                layer_counts[depth + 1] = layer_counts.get(depth + 1, 0) + 1
                q.append(nxt)

    diameter = max(layer_counts)
    layer_sizes = [layer_counts[i] for i in range(diameter + 1)]
    return {
        "bfs_computed": True,
        "bfs_skip_reason": "",
        "diameter": diameter,
        "last_layer_count": layer_sizes[-1],
        "layer_sizes": json.dumps(layer_sizes),
    }


# -----------------------------
# Model/evaluation helpers
# -----------------------------


class Net(nn.Module):
    def __init__(self, input_size: int, hidden_dims: Sequence[int], num_classes_for_one_hot: int):
        super().__init__()
        self.num_classes_for_one_hot = num_classes_for_one_hot
        in_features = input_size * num_classes_for_one_hot
        layers: List[nn.Module] = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes_for_one_hot).float().flatten(start_dim=-2)
        return self.layers(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        spearman = np.nan
    else:
        spearman = stats.spearmanr(y_true, y_pred).statistic
    return {"rmse": float(rmse), "r2": float(r2), "spearman": float(spearman) if not np.isnan(spearman) else np.nan}


def predict_in_batches(model: nn.Module, X: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size)
    pred_list: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            pred_list.append(model(batch_X).detach().cpu())
    return torch.cat(pred_list).numpy().ravel()


def unique_state_count(X: torch.Tensor) -> int:
    return int(torch.unique(X.detach().cpu(), dim=0).shape[0])


def label_stats(*ys: torch.Tensor) -> Dict[str, float]:
    y_all = torch.cat([y.detach().cpu().float().flatten() for y in ys])
    return {
        "label_min": float(y_all.min().item()),
        "label_max": float(y_all.max().item()),
        "label_mean": float(y_all.mean().item()),
        "label_std": float(y_all.std(unbiased=False).item()),
    }


@dataclass(frozen=True)
class SweepConfig:
    n: int
    width: int
    n_random_walks_to_generate: int
    walk_length_multiplier: int
    random_walks_type: str
    n_random_walks_steps_back_to_ban: int
    n_epochs: int
    lr: float
    batch_size: int
    n_val_samples: int
    n_test_samples: int
    seed: int

    @property
    def run_name(self) -> str:
        return (
            f"n{self.n}_w{self.width}_rw{self.n_random_walks_to_generate}_"
            f"wlm{self.walk_length_multiplier}_{self.random_walks_type}_"
            f"ban{self.n_random_walks_steps_back_to_ban}_ep{self.n_epochs}_"
            f"lr{self.lr:g}_bs{self.batch_size}_seed{self.seed}"
        )


def iter_sweep_configs(args: argparse.Namespace) -> Iterable[SweepConfig]:
    for values in itertools.product(
        args.n_values,
        args.widths,
        args.n_random_walks_values,
        args.walk_length_multipliers,
        args.random_walk_types,
        args.steps_back_to_ban_values,
        args.n_epochs_values,
        args.lr_values,
        args.batch_size_values,
        args.n_val_samples_values,
        args.n_test_samples_values,
        args.seed_values,
    ):
        yield SweepConfig(*values)


def train_one_config(
    sweep_cfg: SweepConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    sweep_group_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(sweep_cfg.seed)

    n = sweep_cfg.n
    walk_length = get_random_walk_length(n, sweep_cfg.walk_length_multiplier)
    generators, identity_state, dtype_state = build_problem(n=n, koltsov3_k=args.koltsov3_k, device=device)

    print(f"\n----- Starting {sweep_cfg.run_name} -----", flush=True)
    print(f"walk_length={walk_length}, device={device}", flush=True)

    # Validation/test data are fixed within this configuration. Training data are
    # regenerated each epoch below, matching the original script's logic.
    X_val, y_val = random_walks(
        generators,
        n_random_walk_length=walk_length,
        n_random_walks_to_generate=sweep_cfg.n_val_samples,
        n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
        random_walks_type=sweep_cfg.random_walks_type,
        state_rw_start=identity_state,
        dtype_state=dtype_state,
        device=device,
    )
    X_test, y_test = random_walks(
        generators,
        n_random_walk_length=walk_length,
        n_random_walks_to_generate=sweep_cfg.n_test_samples,
        n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
        random_walks_type=sweep_cfg.random_walks_type,
        state_rw_start=identity_state,
        dtype_state=dtype_state,
        device=device,
    )

    model = Net(input_size=n, hidden_dims=[sweep_cfg.width], num_classes_for_one_hot=n).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=sweep_cfg.lr)
    num_parameters = count_parameters(model)

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --use-wandb true was requested.")
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=sweep_group_name,
            name=sweep_cfg.run_name,
            job_type="train",
            tags=["koltsov3", "random-walk", "mlp", "general-sweep"],
            config={
                **asdict(sweep_cfg),
                "walk_length": walk_length,
                "koltsov3_k": args.koltsov3_k,
                "device_arg": args.device,
                "num_parameters": num_parameters,
                "layer_sizes": json.dumps([sweep_cfg.width]),
            },
            # This prevents accidental continuation/overwriting of previous runs.
            resume="never",
            reinit="finish_previous",
        )

    epoch_rows: List[Dict[str, Any]] = []
    val_rmse_by_epoch: List[float] = []
    val_spearman_by_epoch: List[float] = []
    train_loss_by_epoch: List[float] = []
    X_train_last: Optional[torch.Tensor] = None
    y_train_last: Optional[torch.Tensor] = None

    fit_start = time.time()
    for epoch in range(sweep_cfg.n_epochs):
        X_train, y_train = random_walks(
            generators,
            n_random_walk_length=walk_length,
            n_random_walks_to_generate=sweep_cfg.n_random_walks_to_generate,
            n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
            random_walks_type=sweep_cfg.random_walks_type,
            state_rw_start=identity_state,
            dtype_state=dtype_state,
            device=device,
        )

        indices = torch.randperm(X_train.shape[0], device=X_train.device)
        X_train = X_train[indices]
        y_train = y_train[indices]
        X_train_last = X_train
        y_train_last = y_train

        model.train()
        train_loss = 0.0
        n_batches = 0
        for start in range(0, X_train.shape[0], sweep_cfg.batch_size):
            end = min(start + sweep_cfg.batch_size, X_train.shape[0])
            batch_X = X_train[start:end]
            batch_y = y_train[start:end]

            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_loss_by_epoch.append(float(train_loss))

        val_pred = predict_in_batches(model, X_val, sweep_cfg.batch_size, device)
        val_metrics = evaluate_predictions(y_val.detach().cpu().numpy(), val_pred)
        val_rmse_by_epoch.append(val_metrics["rmse"])
        val_spearman_by_epoch.append(val_metrics["spearman"])

        epoch_row = {
            "config_id": sweep_cfg.run_name,
            **asdict(sweep_cfg),
            "epoch": epoch,
            "walk_length": walk_length,
            "train_loss": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_spearman": val_metrics["spearman"],
            "num_parameters": num_parameters,
            "layer_sizes": json.dumps([sweep_cfg.width]),
        }
        epoch_rows.append(epoch_row)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/rmse": val_metrics["rmse"],
                    "val/r2": val_metrics["r2"],
                    "val/spearman": val_metrics["spearman"],
                },
                step=epoch,
            )

        print(
            f"epoch {epoch + 1}/{sweep_cfg.n_epochs}: "
            f"train_loss={train_loss:.5f}, "
            f"val_rmse={val_metrics['rmse']:.5f}, "
            f"val_spearman={val_metrics['spearman']:.5f}",
            flush=True,
        )

    fit_time_sec = time.time() - fit_start
    assert X_train_last is not None and y_train_last is not None

    predict_start = time.time()
    train_pred = predict_in_batches(model, X_train_last, sweep_cfg.batch_size, device)
    val_pred = predict_in_batches(model, X_val, sweep_cfg.batch_size, device)
    test_pred = predict_in_batches(model, X_test, sweep_cfg.batch_size, device)
    predict_time_sec = time.time() - predict_start

    train_metrics = evaluate_predictions(y_train_last.detach().cpu().numpy(), train_pred)
    val_metrics = evaluate_predictions(y_val.detach().cpu().numpy(), val_pred)
    test_metrics = evaluate_predictions(y_test.detach().cpu().numpy(), test_pred)

    n_train_states = int(X_train_last.shape[0])
    n_val_states = int(X_val.shape[0])
    n_test_states = int(X_test.shape[0])
    num_unique_train_states = unique_state_count(X_train_last)
    num_unique_val_states = unique_state_count(X_val)
    num_unique_test_states = unique_state_count(X_test)

    label_summary = label_stats(y_train_last, y_val, y_test)

    finite_val_spearman = np.array(val_spearman_by_epoch, dtype=float)
    if np.all(np.isnan(finite_val_spearman)):
        best_epoch_by_val_spearman = np.nan
    else:
        best_epoch_by_val_spearman = int(np.nanargmax(finite_val_spearman))

    bfs_metadata: Dict[str, Any] = {
        "bfs_computed": False,
        "bfs_skip_reason": "disabled",
        "diameter": np.nan,
        "last_layer_count": np.nan,
        "layer_sizes": "",
    }
    if args.compute_bfs_metadata:
        print("Computing exact BFS metadata. This is only safe for small n.", flush=True)
        bfs_metadata = compute_exact_bfs_metadata(n, generators, max_states=args.max_bfs_states)

    summary_row = {
        "config_id": sweep_cfg.run_name,
        **asdict(sweep_cfg),
        "walk_length": walk_length,
        "graph_family": "koltsov3",
        "koltsov3_k": args.koltsov3_k,
        "device": str(device),
        "final_train_loss": train_loss_by_epoch[-1],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "train_spearman": train_metrics["spearman"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_spearman": val_metrics["spearman"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_spearman": test_metrics["spearman"],
        "n_train_states": n_train_states,
        "n_val_states": n_val_states,
        "n_test_states": n_test_states,
        "num_unique_train_states": num_unique_train_states,
        "num_unique_val_states": num_unique_val_states,
        "num_unique_test_states": num_unique_test_states,
        "unique_train_fraction": num_unique_train_states / max(n_train_states, 1),
        "unique_val_fraction": num_unique_val_states / max(n_val_states, 1),
        "unique_test_fraction": num_unique_test_states / max(n_test_states, 1),
        **label_summary,
        "num_parameters": num_parameters,
        "layer_sizes": json.dumps([sweep_cfg.width]),
        "train_val_rmse_gap": val_metrics["rmse"] - train_metrics["rmse"],
        "train_test_rmse_gap": test_metrics["rmse"] - train_metrics["rmse"],
        "val_test_rmse_gap": test_metrics["rmse"] - val_metrics["rmse"],
        "best_epoch_by_val_rmse": int(np.argmin(val_rmse_by_epoch)),
        "best_epoch_by_val_spearman": best_epoch_by_val_spearman,
        "best_val_rmse_during_training": float(np.min(val_rmse_by_epoch)),
        "best_val_spearman_during_training": float(np.nanmax(finite_val_spearman)) if not np.all(np.isnan(finite_val_spearman)) else np.nan,
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
        **bfs_metadata,
    }

    if wandb_run is not None:
        wandb_run.log({f"final/{k}": v for k, v in summary_row.items() if isinstance(v, (int, float, str, bool))})
        wandb_run.finish()

    print(f"Finished {sweep_cfg.run_name}", flush=True)
    return summary_row, epoch_rows


# -----------------------------
# Output helpers
# -----------------------------


def save_plots(df_summary: pd.DataFrame, df_epochs: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Compact overview: one point per configuration. The x-axis is config index
    # because the sweep may include many dimensions beyond width.
    df_plot = df_summary.reset_index(drop=True).copy()
    df_plot["config_index"] = np.arange(len(df_plot))

    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["config_index"], df_plot["train_rmse"], marker="o", label="train")
    plt.plot(df_plot["config_index"], df_plot["val_rmse"], marker="o", label="val")
    plt.plot(df_plot["config_index"], df_plot["test_rmse"], marker="o", label="test")
    plt.title("RMSE by completed configuration")
    plt.xlabel("configuration index")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "rmse_by_config.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["config_index"], df_plot["train_spearman"], marker="o", label="train")
    plt.plot(df_plot["config_index"], df_plot["val_spearman"], marker="o", label="val")
    plt.plot(df_plot["config_index"], df_plot["test_spearman"], marker="o", label="test")
    plt.title("Spearman by completed configuration")
    plt.xlabel("configuration index")
    plt.ylabel("Spearman")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "spearman_by_config.png", dpi=200)
    plt.close()

    # Width trend plots for cases where width is one of the varied dimensions.
    group_cols = [
        "n",
        "n_random_walks_to_generate",
        "walk_length_multiplier",
        "random_walks_type",
        "n_random_walks_steps_back_to_ban",
        "n_epochs",
        "lr",
        "batch_size",
        "n_val_samples",
        "n_test_samples",
        "seed",
    ]
    for group_key, df_g in df_summary.groupby(group_cols, dropna=False):
        if df_g["width"].nunique() < 2:
            continue
        df_g = df_g.sort_values("width")
        n = df_g["n"].iloc[0]
        suffix = f"n{n}_rw{df_g['n_random_walks_to_generate'].iloc[0]}_wlm{df_g['walk_length_multiplier'].iloc[0]}_ban{df_g['n_random_walks_steps_back_to_ban'].iloc[0]}_seed{df_g['seed'].iloc[0]}"

        plt.figure(figsize=(8, 5))
        plt.plot(df_g["width"], df_g["train_rmse"], marker="o", label="train")
        plt.plot(df_g["width"], df_g["val_rmse"], marker="o", label="val")
        plt.plot(df_g["width"], df_g["test_rmse"], marker="o", label="test")
        plt.xscale("log", base=2)
        plt.title(f"RMSE vs width ({suffix})")
        plt.xlabel("width")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"rmse_vs_width_{suffix}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df_g["width"], df_g["train_spearman"], marker="o", label="train")
        plt.plot(df_g["width"], df_g["val_spearman"], marker="o", label="val")
        plt.plot(df_g["width"], df_g["test_spearman"], marker="o", label="test")
        plt.xscale("log", base=2)
        plt.title(f"Spearman vs width ({suffix})")
        plt.xlabel("width")
        plt.ylabel("Spearman")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"spearman_vs_width_{suffix}.png", dpi=200)
        plt.close()

    # Epoch trends for a limited number of configs to avoid producing thousands
    # of tiny plot files during a large sweep.
    max_epoch_plots = 25
    for idx, (config_id, df_cfg) in enumerate(df_epochs.groupby("config_id")):
        if idx >= max_epoch_plots:
            break
        df_cfg = df_cfg.sort_values("epoch")
        safe_id = str(config_id).replace("/", "_").replace(" ", "_")[:160]

        plt.figure(figsize=(8, 5))
        plt.plot(df_cfg["epoch"], df_cfg["val_rmse"], marker="o")
        plt.title(f"Validation RMSE by epoch\n{safe_id}")
        plt.xlabel("epoch")
        plt.ylabel("validation RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"val_rmse_by_epoch_{safe_id}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df_cfg["epoch"], df_cfg["val_spearman"], marker="o")
        plt.title(f"Validation Spearman by epoch\n{safe_id}")
        plt.xlabel("epoch")
        plt.ylabel("validation Spearman")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"val_spearman_by_epoch_{safe_id}.png", dpi=200)
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generalized Koltsov3 random-walk MLP sweeps.")

    parser.add_argument("--n-values", type=parse_int_list, default=[5], help="Comma-separated n values, e.g. 5,6,7,8")
    parser.add_argument("--widths", type=parse_int_list, default=[32], help="Comma-separated hidden widths, e.g. 32,64,128")
    parser.add_argument("--n-random-walks-values", type=parse_int_list, default=[500], help="Comma-separated training random-walk counts generated each epoch")
    parser.add_argument("--walk-length-multipliers", type=parse_int_list, default=[8], help="Comma-separated multipliers; walk_length = multiplier * n")
    parser.add_argument("--random-walk-types", type=parse_str_list, default=["non-backtracking-beam"], help="Comma-separated types: simple,non-backtracking-beam")
    parser.add_argument("--steps-back-to-ban-values", type=parse_int_list, default=[2], help="Comma-separated previous move counts to ban")
    parser.add_argument("--n-epochs-values", type=parse_int_list, default=[25], help="Comma-separated epoch counts")
    parser.add_argument("--lr-values", type=parse_float_list, default=[1e-3], help="Comma-separated Adam learning rates")
    parser.add_argument("--batch-size-values", type=parse_int_list, default=[64], help="Comma-separated batch sizes")
    parser.add_argument("--n-val-samples-values", type=parse_int_list, default=[300], help="Comma-separated validation random-walk counts")
    parser.add_argument("--n-test-samples-values", type=parse_int_list, default=[300], help="Comma-separated test random-walk counts")
    parser.add_argument("--seed-values", type=parse_int_list, default=[0], help="Comma-separated seeds")

    parser.add_argument("--output-dir", type=Path, default=Path("koltsov3_general_sweep_results"), help="Output directory for CSVs and plots")
    parser.add_argument("--koltsov3-k", type=int, default=0, help="k in the Koltsov3 S=(k,k+2) generator")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Use auto, cpu, or cuda")
    parser.add_argument("--save-plots", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--compute-bfs-metadata", type=str2bool, nargs="?", const=True, default=False, help="Compute exact BFS diameter/layers when n! <= max_bfs_states")
    parser.add_argument("--max-bfs-states", type=int, default=50000, help="Safety cap for optional exact BFS metadata")

    parser.add_argument("--use-wandb", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--wandb-entity", type=str, default="CayleyPy")
    parser.add_argument("--wandb-project", type=str, default="cayley-py")
    parser.add_argument("--wandb-group", type=str, default=None, help="Group name for the whole sweep; defaults to timestamped name")
    parser.add_argument("--wandb-login", type=str2bool, nargs="?", const=True, default=False, help="Call wandb.login() before starting")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    valid_walk_types = {"simple", "non-backtracking-beam"}
    invalid = sorted(set(args.random_walk_types) - valid_walk_types)
    if invalid:
        raise ValueError(f"Invalid random walk type(s): {invalid}. Valid options: {sorted(valid_walk_types)}")

    n_configs = 1
    for values in [
        args.n_values,
        args.widths,
        args.n_random_walks_values,
        args.walk_length_multipliers,
        args.random_walk_types,
        args.steps_back_to_ban_values,
        args.n_epochs_values,
        args.lr_values,
        args.batch_size_values,
        args.n_val_samples_values,
        args.n_test_samples_values,
        args.seed_values,
    ]:
        n_configs *= len(values)
    args.n_total_configs = n_configs

    if n_configs > 200:
        print(
            f"WARNING: this sweep has {n_configs} configurations. "
            "That may be expensive on Hyak. Consider a smaller smoke test first.",
            flush=True,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    device = choose_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    sweep_group_name = args.wandb_group or f"koltsov3_general_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"device: {device}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"total configurations: {args.n_total_configs}", flush=True)
    print(f"sweep_group_name: {sweep_group_name}", flush=True)

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run with --use-wandb false.")
        if args.wandb_login:
            wandb.login()

    summary_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []

    all_configs = list(iter_sweep_configs(args))
    for i, sweep_cfg in enumerate(all_configs, start=1):
        print(f"\n===== Configuration {i}/{len(all_configs)} =====", flush=True)
        summary_row, epoch_rows_for_config = train_one_config(
            sweep_cfg=sweep_cfg,
            args=args,
            device=device,
            output_dir=output_dir,
            sweep_group_name=sweep_group_name,
        )
        summary_rows.append(summary_row)
        epoch_rows.extend(epoch_rows_for_config)

        # Incremental writes make long Slurm jobs safer if a later config fails.
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary_results_partial.csv", index=False)
        pd.DataFrame(epoch_rows).to_csv(output_dir / "epoch_results_partial.csv", index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_epochs = pd.DataFrame(epoch_rows)

    summary_csv = output_dir / "summary_results.csv"
    epoch_csv = output_dir / "epoch_results.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_epochs.to_csv(epoch_csv, index=False)

    # Save the exact command/config arguments for reproducibility.
    with open(output_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    if args.save_plots:
        save_plots(df_summary, df_epochs, output_dir)

    print(f"\nSaved final summary CSV to {summary_csv}", flush=True)
    print(f"Saved epoch-level CSV to {epoch_csv}", flush=True)
    print(f"Saved plots to {output_dir / 'plots'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
