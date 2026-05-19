#!/usr/bin/env python3
"""Koltsov3 random-walk gradient boosting (XGBoost) pipeline for Hyak/Slurm.

Ports Chervov & Soibelman's Algorithm 4 (Modified DQN) to XGBoost. Three phases
chained per configuration:

  Phase 1 (warm-up). Diffusion-distance training: generate random walks from
    the identity, label states by normalized random-walk depth, fit XGBoost.
    This is the only phase the original koltsov3_gb_sweep.py ran.

  Phase 2 (MDQN refinement, optional). For each MDQN epoch, regenerate walks,
    expand each visited state's neighbors, compute Bellman targets
        d(s) = 1 + min_{t in N(s)} f_theta(t),
    clip d(s) into [0, k] using the state's first-visit step k, and refit
    XGBoost on (s, d(s)). Enabled when any --n-epochs-dqn-values > 0.

  Phase 3 (guided beam search, optional). Use the final model as a heuristic in
    a torch beam search from a scrambled state back to the identity. Reports
    whether and at what step the identity was reached. Enabled by
    --run-beam-search true.

Preserved from the original sweep script:
  - Koltsov3 generator construction on permutations 0..n-1
  - random-walk training data with optional first-visit dedup
  - fixed validation/test random walks per configuration
  - hand-crafted feature set from Junaid's LightGBM work (extract_features())
  - summary CSV, per-iteration CSV, plots, optional W&B
  - optional exact BFS metadata for small n only
  - W&B key loaded from a gitignored .env file (WANDB_API_KEY)

Sweep axes:
  - Phase 1 / data: n, n_random_walks, walk_length_multiplier, walk_type,
    steps_back_to_ban, n_val_samples, n_test_samples, seed
  - Phase 1 / XGBoost: n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, min_child_weight, reg_lambda, reg_alpha
  - Phase 2 / MDQN: n_epochs_dqn (0 disables), dqn_n_random_walks, dqn_clip
  - Phase 3 / beam: beam_width, n_steps_limit_mult (limit = mult * n**2),
    beam_steps_back_to_ban, n_scrambles

MDQN requires --dedup-strategy first-visit so that each state has a well-defined
"first-visit step k" to clip against (line 7 of Algorithm 4).

Example smoke test (Phase 1 only):
    python koltsov3_gb_pipeline.py \
      --n-values 5 --n-random-walks-values 50 \
      --walk-length-multipliers 4 --random-walk-types simple \
      --steps-back-to-ban-values 0 \
      --n-estimators-values 20 --max-depth-values 4 \
      --learning-rate-values 0.1 --subsample-values 0.8 \
      --colsample-bytree-values 0.8 --min-child-weight-values 5 \
      --reg-lambda-values 1.0 --reg-alpha-values 0.0 \
      --n-val-samples-values 20 --n-test-samples-values 20 \
      --seed-values 0 --output-dir smoke_test --use-wandb false

Example full Algorithm 4 (Phase 1 + Phase 2 + Phase 3):
    python koltsov3_gb_pipeline.py \
      ... (Phase 1 args as above) \
      --dedup-strategy first-visit \
      --n-epochs-dqn-values 50 --dqn-n-random-walks-values 50 \
      --run-beam-search true --beam-width-values 1024 \
      --n-steps-limit-mult-values 4 --beam-steps-back-to-ban-values 8 \
      --n-scrambles-values 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
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
import xgboost as xgb
from scipy import stats
from sklearn.metrics import r2_score, root_mean_squared_error

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

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


def parse_walks_per_n(value: str) -> Dict[int, int]:
    if not value:
        raise argparse.ArgumentTypeError("Expected entries like '5:500,16:25000'.")
    mapping: Dict[int, int] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(
                f"Invalid walks-per-n entry '{chunk}'; expected 'n:walks'."
            )
        n_str, walks_str = chunk.split(":", 1)
        try:
            n_key = int(n_str.strip())
            walks = int(walks_str.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid walks-per-n entry '{chunk}'.") from exc
        if n_key <= 0 or walks <= 0:
            raise argparse.ArgumentTypeError(
                f"walks-per-n entry '{chunk}' must have positive n and walks."
            )
        if n_key in mapping:
            raise argparse.ArgumentTypeError(f"Duplicate n={n_key} in --walks-per-n.")
        mapping[n_key] = walks
    if not mapping:
        raise argparse.ArgumentTypeError("--walks-per-n cannot be empty.")
    return mapping


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_bool_list(value: str) -> List[bool]:
    if not value:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of bools.")
    parsed = [str2bool(x.strip()) for x in value.split(",") if x.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("List cannot be empty.")
    return parsed


# -----------------------------
# Reproducibility/device helpers
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wandb_key() -> Optional[str]:
    """Load WANDB_API_KEY from the environment or a gitignored .env file.

    Search order for the .env file:
      1. RITHIKESH_ENV_PATH if set
      2. <this file>/../.env  (i.e. Rithikesh/.env)
      3. current working directory / .env
    """
    if load_dotenv is not None:
        candidates = []
        env_override = os.environ.get("RITHIKESH_ENV_PATH")
        if env_override:
            candidates.append(Path(env_override))
        candidates.append(Path(__file__).resolve().parent.parent / ".env")
        candidates.append(Path.cwd() / ".env")
        for candidate in candidates:
            if candidate.is_file():
                load_dotenv(candidate)
                break
    return os.environ.get("WANDB_API_KEY")


# -----------------------------
# Koltsov3/random-walk logic (preserved from the MLP sweep script)
# -----------------------------


def get_koltsov3_moves(n: int, k: int = 0) -> List[np.ndarray]:
    """Koltsov3 generators on {0, 1, ..., n-1}.

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


def get_random_walk_length(n: int, walk_length_multiplier: int, mode: str) -> int:
    """Decide walk length from n and the chosen mode.

    Modes:
      - 'multiplier' (default): walk_length = walk_length_multiplier * n. Lets
        the multiplier axis drive walk length, matching Merav's MLP sweep.
      - 'min_npairs_or_16n': walk_length = min(n*(n-1)//2, 16*n). Danny's
        n(n-1)/2 formula (the "Antonina Dolgorukova" value used in his
        CNN_FINAL_RESULTS notebook), capped at 16n so n=64 doesn't blow up
        to 2016-step walks. In this mode walk_length_multiplier is ignored
        for the length computation, but is still tracked in run names so
        sweeps that vary the axis stay distinguishable.
    """
    if mode == "multiplier":
        return walk_length_multiplier * n
    if mode == "min_npairs_or_16n":
        return min(n * (n - 1) // 2, 16 * n)
    raise ValueError(f"Unknown walk_length_mode: {mode}")


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


def get_neighbors(states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    """Apply every generator to every state.

    Returns a 3-D tensor of shape (n_states, n_moves, state_size). Used by both
    the MDQN Bellman update (min over neighbors) and beam-search expansion. To
    get a 2-D candidate matrix, call .flatten(end_dim=1) on the result.
    """
    n_states = states.size(0)
    n_moves = moves.size(0)
    state_size = states.size(1)
    expanded_states = states.unsqueeze(1).expand(n_states, n_moves, state_size)
    expanded_moves = moves.unsqueeze(0).expand(n_states, n_moves, state_size)
    return torch.gather(expanded_states, 2, expanded_moves)


def sample_allowed_moves(allowed: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sample one allowed generator per row."""
    n_rows = allowed.shape[0]
    move_ids = torch.zeros(n_rows, dtype=torch.long, device=device)
    for i in range(n_rows):
        choices = torch.where(allowed[i])[0]
        move_ids[i] = choices[torch.randint(len(choices), (1,), device=device)]
    return move_ids


def first_visit_dedup(
    states: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep one row per unique state, labeled by the minimum (earliest) visit.

    Implements the paper's diffusion-distance semantics: a state s visited
    multiple times at steps k1 < k2 < ... gets label k1. Without this, the
    same state appears in training with conflicting labels and the model
    converges to predicting the per-state mean (~0.5 under our normalization).
    """
    n = states.shape[0]
    if n == 0:
        return states, labels
    # torch.unique(dim=0) returns sorted unique rows + an inverse index that
    # maps each input row to its group id. We then use scatter_reduce to take
    # the min label per group.
    unique_states, inverse_idx = torch.unique(states, dim=0, return_inverse=True)
    n_groups = unique_states.shape[0]
    min_labels = torch.full(
        (n_groups,), float("inf"), dtype=labels.dtype, device=labels.device
    )
    min_labels.scatter_reduce_(0, inverse_idx, labels, reduce="amin", include_self=True)
    return unique_states, min_labels


def random_walks(
    generators: Sequence[np.ndarray] | torch.Tensor,
    n_random_walk_length: int,
    n_random_walks_to_generate: int,
    n_random_walks_steps_back_to_ban: int,
    random_walks_type: str,
    state_rw_start: torch.Tensor,
    dtype_state: torch.dtype,
    device: torch.device,
    dedup_strategy: str = "none",
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

    states_out = torch.cat(all_states, dim=0)
    labels_out = torch.cat(all_y, dim=0).float()
    if dedup_strategy == "first-visit":
        states_out, labels_out = first_visit_dedup(states_out, labels_out)
    elif dedup_strategy != "none":
        raise ValueError(f"Unknown dedup_strategy: {dedup_strategy}")
    return states_out, labels_out


# -----------------------------
# Feature extraction (ported from Junaid's LightGBM run scripts)
# -----------------------------


def extract_features(states_array: np.ndarray, n: int, k: int = 0) -> pd.DataFrame:
    """Hand-crafted features for a batch of Koltsov3 permutation states.

    states_array: integer array of shape (N, n), each row a permutation of 0..n-1.
    Returns a DataFrame with one row per state and ~13 groups of features.

    This is the same feature set Junaid used for LightGBM on this problem;
    it is reused here so the gradient boosting model has informative inputs
    instead of raw permutation positions.
    """
    states_array = np.asarray(states_array)
    N = states_array.shape[0]
    f: Dict[str, np.ndarray] = {}
    ident = np.arange(n)

    # 1. Displacement: how far each value is from its identity position.
    disp = np.abs(states_array - ident)
    for i in range(n):
        f[f"d_{i}"] = disp[:, i]
    f["d_sum"] = disp.sum(1)
    f["d_max"] = disp.max(1)
    f["d_mean"] = disp.mean(1)
    f["d_std"] = disp.std(1)
    f["d_med"] = np.median(disp, 1)
    f["d_eq0"] = (disp == 0).sum(1)
    f["d_eq1"] = (disp == 1).sum(1)
    f["d_ge2"] = (disp >= 2).sum(1)
    f["d_ge4"] = (disp >= 4).sum(1)

    # 2. Parity mismatch: value at an even/odd index that has the wrong parity.
    pm = ((states_array & 1) != (ident & 1)).astype(np.float32)
    for i in range(n):
        f[f"pm_{i}"] = pm[:, i]
    f["pm_sum"] = pm.sum(1)
    f["pm_pct"] = pm.mean(1)
    f["pm_x_disp"] = (pm * disp).sum(1)

    # 3. Adjacent pairs: sortedness and adjacent differences.
    sorted_adj = (states_array[:, :-1] < states_array[:, 1:]).astype(np.float32)
    for i in range(n - 1):
        f[f"s_{i}{i + 1}"] = sorted_adj[:, i]
    f["s_sum"] = sorted_adj.sum(1)
    adj_diff = np.abs(np.diff(states_array, axis=1))
    for i in range(n - 1):
        f[f"ad_{i}"] = adj_diff[:, i]
    f["ad_sum"] = adj_diff.sum(1)
    f["ad_max"] = adj_diff.max(1)
    pair_correct = (
        (states_array[:, :-1] == ident[:-1]) & (states_array[:, 1:] == ident[1:])
    ).astype(np.float32)
    f["pair_correct"] = pair_correct.sum(1)

    # 4. Inversions.
    invs = np.zeros(N, dtype=np.float32)
    for i in range(n):
        invs += (states_array[:, i : i + 1] > states_array[:, i + 1 :]).sum(axis=1)
    f["inv"] = invs
    f["inv_frac"] = invs / (n * (n - 1) / 2)
    f["inv_parity"] = (invs.astype(np.int64) & 1).astype(np.float32)

    # 5. Descents.
    descs = np.zeros(N, dtype=np.float32)
    for i in range(n - 1):
        descs += (states_array[:, i] > states_array[:, i + 1]).astype(np.float32)
    f["desc"] = descs

    # 6. I-generator features: positions the I generator acts on.
    ieu = np.zeros(N, dtype=np.float32)
    iec = np.zeros(N, dtype=np.float32)
    for idx in range(0, n - 1, 2):
        ieu += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        iec += ((states_array[:, idx] == idx) & (states_array[:, idx + 1] == idx + 1)).astype(np.float32)
    f["I_unsorted"] = ieu
    f["I_correct"] = iec

    # 7. K-generator features: positions the K generator acts on.
    keu = np.zeros(N, dtype=np.float32)
    kec = np.zeros(N, dtype=np.float32)
    for idx in range(1, n - 1, 2):
        keu += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        kec += ((states_array[:, idx] == idx) & (states_array[:, idx + 1] == idx + 1)).astype(np.float32)
    f["K_unsorted"] = keu
    f["K_correct"] = kec

    # 8. S-generator features: the (k, k+2) swap.
    if k + 2 < n:
        f["S_sorted"] = (states_array[:, k] < states_array[:, k + 2]).astype(np.float32)
        f["S_diff"] = np.abs(states_array[:, k].astype(np.float32) - states_array[:, k + 2].astype(np.float32))
        f["S_disp"] = np.abs(states_array[:, k] - k) + np.abs(states_array[:, k + 2] - (k + 2))
        f["S_need_swap"] = ((states_array[:, k] == k + 2) & (states_array[:, k + 2] == k)).astype(np.float32)

    # 9. Theoretical lower bounds on distance.
    f["lb_disp"] = disp.max(1) / 2.0
    f["lb_parity"] = pm.sum(1) / 2.0
    f["lb_comb"] = np.maximum(f["lb_disp"], f["lb_parity"])

    # 10. Inverse permutation: where each value currently sits.
    inv_perm = np.argsort(states_array, axis=1)
    for v in range(n):
        f[f"pos_of_{v}"] = inv_perm[:, v].astype(np.float32)
    f["where_0"] = np.abs(inv_perm[:, 0] - 0).astype(np.float32)
    f["where_n1"] = np.abs(inv_perm[:, n - 1] - (n - 1)).astype(np.float32)

    # 11. Longest run of already-correct positions (vectorized).
    at_home = states_array == ident
    run_lens = np.zeros(N, dtype=np.float32)
    current = np.zeros(N, dtype=np.int64)
    for col in range(n):
        col_home = at_home[:, col]
        current = np.where(col_home, current + 1, 0)
        run_lens = np.maximum(run_lens, current)
    f["run_correct"] = run_lens

    # 12. Signed displacement / skew.
    sd = states_array.astype(np.float32) - ident.astype(np.float32)
    f["disp_pos"] = np.maximum(sd, 0).sum(1)
    f["disp_neg"] = np.maximum(-sd, 0).sum(1)
    f["disp_net"] = sd.sum(1)

    # 13. Raw positions.
    for i in range(n):
        f[f"p_{i}"] = states_array[:, i].astype(np.float32)

    return pd.DataFrame(f)


def states_tensor_to_features(states: torch.Tensor, n: int, k: int) -> pd.DataFrame:
    """Bridge: torch state tensor from random_walks() -> feature DataFrame."""
    states_np = states.detach().cpu().numpy().astype(np.int32)
    return extract_features(states_np, n=n, k=k)


# -----------------------------
# Optional exact BFS metadata (preserved from the MLP sweep script)
# -----------------------------


def compose_state_with_generator(state: Tuple[int, ...], generator: np.ndarray) -> Tuple[int, ...]:
    """Apply the same index-gather action used by torch.gather in random_walks."""
    return tuple(state[int(i)] for i in generator)


def compute_exact_bfs_metadata(
    n: int,
    generators: Sequence[np.ndarray],
    max_states: int,
    return_distance_map: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[Tuple[int, ...], int]]]:
    """Compute exact BFS diameter/layer sizes from the identity, if small enough.

    Returns (metadata, distance_map). When return_distance_map=True and BFS
    succeeded, distance_map[state_tuple] = shortest-path-distance from identity
    to that state. When BFS was skipped or return_distance_map=False, the
    second element is None.
    """
    total_possible_states = math.factorial(n)
    if total_possible_states > max_states:
        meta = {
            "bfs_computed": False,
            "bfs_skip_reason": f"n!={total_possible_states} exceeds max_bfs_states={max_states}",
            "diameter": np.nan,
            "last_layer_count": np.nan,
            "layer_sizes": "",
        }
        return meta, None

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
    meta = {
        "bfs_computed": True,
        "bfs_skip_reason": "",
        "diameter": diameter,
        "last_layer_count": layer_sizes[-1],
        "layer_sizes": json.dumps(layer_sizes),
    }
    return meta, visited if return_distance_map else None


# -----------------------------
# Evaluation helpers (preserved from the MLP sweep script)
# -----------------------------


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


def unique_state_count(states: torch.Tensor) -> int:
    return int(torch.unique(states.detach().cpu(), dim=0).shape[0])


def label_stats(*ys: torch.Tensor) -> Dict[str, float]:
    y_all = torch.cat([y.detach().cpu().float().flatten() for y in ys])
    return {
        "label_min": float(y_all.min().item()),
        "label_max": float(y_all.max().item()),
        "label_mean": float(y_all.mean().item()),
        "label_std": float(y_all.std(unbiased=False).item()),
    }


# -----------------------------
# Sweep configuration
# -----------------------------


@dataclass(frozen=True)
class SweepConfig:
    n: int
    n_random_walks_to_generate: int
    walk_length_multiplier: int
    random_walks_type: str
    n_random_walks_steps_back_to_ban: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_lambda: float
    reg_alpha: float
    n_val_samples: int
    n_test_samples: int
    seed: int
    # Phase 2 (MDQN). n_epochs_dqn == 0 disables. dqn_n_random_walks is the
    # number of walks regenerated per MDQN epoch. dqn_clip toggles the
    # min(k, max(0, d)) clip from Algorithm 4 line 7.
    n_epochs_dqn: int
    dqn_n_random_walks: int
    dqn_clip: bool
    # Phase 3 (beam search). Enabled per-config when args.run_beam_search is on.
    # n_steps_limit = n_steps_limit_mult * n**2.
    # scramble_depth_mult: scramble steps = mult * n_generators**3 (notebook
    # uses mult=100 i.e. n_gens**3 * 100; the deep scramble matters because
    # it sets how far the start state is from identity and therefore how hard
    # the beam search has to work. Make this an axis so it can be swept.)
    beam_width: int
    n_steps_limit_mult: int
    beam_steps_back_to_ban: int
    n_scrambles: int
    beam_scramble_depth_mult: int

    @property
    def run_name(self) -> str:
        base = (
            f"n{self.n}_rw{self.n_random_walks_to_generate}_"
            f"wlm{self.walk_length_multiplier}_{self.random_walks_type}_"
            f"ban{self.n_random_walks_steps_back_to_ban}_"
            f"ne{self.n_estimators}_md{self.max_depth}_lr{self.learning_rate:g}_"
            f"ss{self.subsample:g}_cs{self.colsample_bytree:g}_"
            f"mcw{self.min_child_weight:g}_l2{self.reg_lambda:g}_l1{self.reg_alpha:g}_"
            f"seed{self.seed}"
        )
        if self.n_epochs_dqn > 0:
            base += f"_dqn{self.n_epochs_dqn}rw{self.dqn_n_random_walks}clip{int(self.dqn_clip)}"
        # Beam fields are always in the run name (even when beam search is off
        # they're carried along as no-op axes) -- keeps run_names unique when a
        # user does sweep over beam params with phases switched on/off.
        base += (
            f"_bw{self.beam_width}_lim{self.n_steps_limit_mult}_"
            f"bban{self.beam_steps_back_to_ban}_sc{self.n_scrambles}_"
            f"scrm{self.beam_scramble_depth_mult}"
        )
        return base


def walks_for_n(args: argparse.Namespace, n: int) -> List[int]:
    """Per-n walk counts. If --walks-per-n maps n, it overrides --n-random-walks-values."""
    override = getattr(args, "walks_per_n", None)
    if override and n in override:
        return [override[n]]
    return list(args.n_random_walks_values)


def iter_sweep_configs(args: argparse.Namespace) -> Iterable[SweepConfig]:
    for n in args.n_values:
        for values in itertools.product(
            walks_for_n(args, n),
            args.walk_length_multipliers,
            args.random_walk_types,
            args.steps_back_to_ban_values,
            args.n_estimators_values,
            args.max_depth_values,
            args.learning_rate_values,
            args.subsample_values,
            args.colsample_bytree_values,
            args.min_child_weight_values,
            args.reg_lambda_values,
            args.reg_alpha_values,
            args.n_val_samples_values,
            args.n_test_samples_values,
            args.seed_values,
            args.n_epochs_dqn_values,
            args.dqn_n_random_walks_values,
            args.dqn_clip_values,
            args.beam_width_values,
            args.n_steps_limit_mult_values,
            args.beam_steps_back_to_ban_values,
            args.n_scrambles_values,
            args.beam_scramble_depth_mult_values,
        ):
            yield SweepConfig(n, *values)


# -----------------------------
# MDQN refinement (Algorithm 4, Part 2)
# -----------------------------


def predict_features_batched(
    model: xgb.Booster,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    best_ntree_limit: int,
    batch_size: int = 200_000,
) -> np.ndarray:
    """Predict in chunks to keep DMatrix allocation bounded."""
    n = len(feature_df)
    preds = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        d = xgb.DMatrix(feature_df.iloc[start:end], feature_names=feature_cols)
        preds[start:end] = model.predict(d, iteration_range=(0, best_ntree_limit))
    return preds


def run_mdqn_phase(
    model: xgb.Booster,
    initial_best_ntree_limit: int,
    sweep_cfg: SweepConfig,
    args: argparse.Namespace,
    generators: Sequence[np.ndarray],
    identity_state: torch.Tensor,
    dtype_state: torch.dtype,
    device: torch.device,
    feature_cols: List[str],
    walk_length: int,
    target_scale: float,
    wandb_run: Optional[Any] = None,
    log_step_offset: int = 0,
) -> Tuple[xgb.Booster, int, List[Dict[str, Any]]]:
    """Run Algorithm 4 Part 2: Bellman-update MDQN refinement on top of a warm-started XGBoost.

    Loop body, repeated sweep_cfg.n_epochs_dqn times:
      1. Regenerate dqn_n_random_walks walks, dedup to first-visit (so each
         state has a well-defined first-visit step k).
      2. Expand neighbors of every visited state via get_neighbors().
      3. Predict normalized distances for the neighbor set with the current
         model; recover integer distance estimates by multiplying by target_scale.
      4. Bellman: d(s) = 1 + min_t f_theta(t).
      5. Clip d(s) into [0, k] (optional, controlled by dqn_clip).
      6. Renormalize to [0, 1] and refit XGBoost from scratch on (state_features,
         normalized d(s)).
    """
    moves = normalize_generators(generators, device)
    n_moves = moves.shape[0]
    state_size = identity_state.shape[0]
    history: List[Dict[str, Any]] = []

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": sweep_cfg.max_depth,
        "learning_rate": sweep_cfg.learning_rate,
        "subsample": sweep_cfg.subsample,
        "colsample_bytree": sweep_cfg.colsample_bytree,
        "min_child_weight": sweep_cfg.min_child_weight,
        "reg_lambda": sweep_cfg.reg_lambda,
        "reg_alpha": sweep_cfg.reg_alpha,
        "seed": sweep_cfg.seed,
        "nthread": args.nthread,
    }
    if device.type == "cuda":
        params["device"] = "cuda"

    current_model = model
    current_ntree_limit = initial_best_ntree_limit

    for epoch in range(sweep_cfg.n_epochs_dqn):
        t0 = time.time()
        # 1. Fresh random walks for this MDQN epoch (the M in MDQN: random
        # subset of nodes is regenerated each epoch).
        X_states, y_norm = random_walks(
            generators,
            n_random_walk_length=walk_length,
            n_random_walks_to_generate=sweep_cfg.dqn_n_random_walks,
            n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
            random_walks_type=sweep_cfg.random_walks_type,
            state_rw_start=identity_state,
            dtype_state=dtype_state,
            device=device,
            dedup_strategy="first-visit",
        )
        n_states = int(X_states.shape[0])
        if n_states == 0:
            print(f"  MDQN epoch {epoch}: dedup produced zero states, skipping", flush=True)
            continue

        # k = integer first-visit step (used for the clip in Algorithm 4 line 7).
        k_per_state = (y_norm.detach().cpu().numpy() * target_scale).round().astype(np.float32)

        # 2. Expand neighbors. neighbors: (n_states, n_moves, state_size) ->
        # flatten to (n_states * n_moves, state_size) so we predict in one batch.
        neighbors = get_neighbors(X_states, moves).reshape(n_states * n_moves, state_size)
        neighbor_features = states_tensor_to_features(neighbors, n=sweep_cfg.n, k=args.koltsov3_k)

        # 3. Predict and reshape back to (n_states, n_moves).
        neighbor_preds = predict_features_batched(
            current_model, neighbor_features, feature_cols, current_ntree_limit
        )
        neighbor_preds = neighbor_preds.reshape(n_states, n_moves)
        # The model predicts NORMALIZED distance; rescale to integer step units.
        neighbor_preds_steps = neighbor_preds * target_scale

        # 4. Bellman update: 1 + min over moves.
        d_new_steps = 1.0 + neighbor_preds_steps.min(axis=1)

        # 5. Clip to [0, k]. With clip on, the algorithm trusts the random-walk
        # bound (the state was reached in k steps, so the true distance can't
        # exceed k).
        if sweep_cfg.dqn_clip:
            d_new_steps = np.minimum(k_per_state, np.maximum(0.0, d_new_steps))

        # 6. Renormalize and refit. We renormalize by walk_length to keep
        # outputs on the same [0, 1] scale as Phase 1, so the warm-up model and
        # the MDQN-refined model are interchangeable downstream.
        d_new_norm = d_new_steps / max(walk_length, 1)
        X_features = states_tensor_to_features(X_states, n=sweep_cfg.n, k=args.koltsov3_k)
        dtrain = xgb.DMatrix(X_features, label=d_new_norm, feature_names=feature_cols)

        # Refit from scratch (n_estimators full rounds, no early stopping --
        # MDQN's Bellman targets change every epoch, so a separate val set
        # tracking through epochs would be misleading).
        evals_result: Dict[str, Dict[str, List[float]]] = {}
        current_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=sweep_cfg.n_estimators,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        current_ntree_limit = current_model.num_boosted_rounds()

        train_rmse_last = evals_result.get("train", {}).get("rmse", [np.nan])[-1]
        d_mean = float(np.mean(d_new_steps))
        d_max = float(np.max(d_new_steps))
        elapsed = time.time() - t0

        row = {
            "dqn_epoch": epoch,
            "dqn_states": n_states,
            "dqn_train_rmse": float(train_rmse_last),
            "dqn_d_mean_steps": d_mean,
            "dqn_d_max_steps": d_max,
            "dqn_epoch_sec": elapsed,
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "mdqn/epoch": epoch,
                    "mdqn/train_rmse": row["dqn_train_rmse"],
                    "mdqn/d_mean_steps": d_mean,
                    "mdqn/d_max_steps": d_max,
                    "mdqn/epoch_sec": elapsed,
                },
                step=log_step_offset + epoch,
            )

        if args.verbose_mdqn:
            print(
                f"  MDQN epoch {epoch}: states={n_states}, train_rmse={train_rmse_last:.5f}, "
                f"d_mean_steps={d_mean:.2f}, d_max_steps={d_max:.2f}, time={elapsed:.1f}s",
                flush=True,
            )

    return current_model, current_ntree_limit, history


# -----------------------------
# Beam search evaluation (Algorithm 4, Part 3 -- guided beam search)
# -----------------------------


def scramble_state(
    identity_state: torch.Tensor,
    generators_tensor: torch.Tensor,
    n_scramble_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply n_scramble_steps random generators to the identity to make a start state."""
    state = identity_state.clone()
    n_gens = generators_tensor.shape[0]
    for _ in range(n_scramble_steps):
        idx = int(torch.randint(0, n_gens, (1,), device=device).item())
        state = state[generators_tensor[idx]]
    return state


def run_beam_search_once(
    model: xgb.Booster,
    best_ntree_limit: int,
    feature_cols: List[str],
    state_start: torch.Tensor,
    state_destination: torch.Tensor,
    generators_tensor: torch.Tensor,
    beam_width: int,
    n_steps_limit: int,
    steps_back_to_ban: int,
    target_scale: float,
    args: argparse.Namespace,
    sweep_cfg: SweepConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """One beam-search rollout using XGBoost as the heuristic.

    Mirrors the notebook's torch beam search (cell 82): at every step apply all
    generators to the current beam, dedup, ban states seen on the last
    steps_back_to_ban steps via hashing, score with the model, keep beam_width
    cheapest.

    Path reconstruction. At each step we store, per surviving beam row, the
    index of its parent in the previous beam and the generator index used to
    reach it. On found=True we walk these backpointers from the matching row
    back to state_start, producing the move sequence. The sequence is then
    re-applied to state_start as a correctness check (beam_path_verified).
    """
    state_size = state_destination.shape[0]
    n_generators = generators_tensor.shape[0]

    # Random hash vector (same trick as the notebook): h(s) = sum(s * v) for a
    # random v in int64. Lets us check "state seen recently" with a fast isin.
    max_int = int(2**62)
    vec_hasher = torch.randint(
        -max_int, max_int, size=(state_size,), device=device, dtype=torch.int64
    )

    array_beam = state_start.view(1, state_size).to(device=device, dtype=torch.long)

    if steps_back_to_ban > 0:
        hash_initial = torch.sum(array_beam.to(torch.int64) * vec_hasher, dim=1)
        vec_hashes_current = hash_initial.expand(
            beam_width * n_generators, steps_back_to_ban
        ).clone()
        cyclic_idx = 0

    found = False
    found_step = -1
    found_local_idx = -1  # row index of the matching candidate in the final beam (pre-break)
    last_q_min = float("nan")
    last_q_max = float("nan")
    last_q_median = float("nan")
    early_exit_reason = "step_limit_reached"

    # Backpointer storage. parent_history[t] and gen_history[t] are 1-D tensors
    # of length beam_size_at_step_t telling us, for each row in the step-t beam,
    # which row of the step-(t-1) beam it came from and which generator was
    # applied. parent_history[0] / gen_history[0] correspond to the start state
    # and are filled with sentinel -1.
    parent_history: List[torch.Tensor] = [
        torch.tensor([-1], dtype=torch.long, device=device)
    ]
    gen_history: List[torch.Tensor] = [
        torch.tensor([-1], dtype=torch.long, device=device)
    ]

    for i_step in range(1, n_steps_limit + 1):
        prev_beam_size = array_beam.shape[0]

        # 1. Apply all generators to all beam states; shape (beam, gens, size).
        #    Before reshape, position (i, j) in candidates_3d came from beam
        #    row i via generator j. After flattening row-major to (B*G, size),
        #    flat index k maps to (parent=k//G, gen=k%G).
        candidates = get_neighbors(array_beam, generators_tensor).reshape(-1, state_size)
        cand_parent = (
            torch.arange(candidates.shape[0], device=device) // n_generators
        )
        cand_gen = torch.arange(candidates.shape[0], device=device) % n_generators

        # 2. Deduplicate. (Critical -- without this the beam collapses to copies.)
        # We need a representative pre-dedup index for each unique row so we
        # can carry parent/gen info through. torch.unique(return_inverse=True)
        # gives inverse[i] = group-id of input row i; scatter the first input
        # index per group to obtain "input index of one representative".
        unique_cands, inverse = torch.unique(candidates, dim=0, return_inverse=True)
        n_unique = unique_cands.shape[0]
        rep_input_idx = torch.full((n_unique,), -1, dtype=torch.long, device=device)
        all_input_idx = torch.arange(candidates.shape[0], device=device)
        # For each group, the smallest input index becomes its representative.
        # scatter_reduce 'amin' with full(-1) doesn't work for ints because
        # amin treats -1 as a candidate min; use a forward fill instead.
        # Iterate-free option: sort inverse and take the first input index per
        # unique group via cumulative bincount tricks. Simplest correct:
        # use scatter_reduce on input indices initialized to a sentinel larger
        # than any valid index.
        sentinel = candidates.shape[0]
        rep_input_idx = torch.full((n_unique,), sentinel, dtype=torch.long, device=device)
        rep_input_idx.scatter_reduce_(
            0, inverse, all_input_idx, reduce="amin", include_self=True
        )
        # All groups must have at least one input row, so no sentinel left.
        candidates = unique_cands
        cand_parent = cand_parent[rep_input_idx]
        cand_gen = cand_gen[rep_input_idx]

        # 3. Check for destination.
        eq_dest = torch.all(candidates == state_destination, dim=1)
        if torch.any(eq_dest).item():
            # Record this final layer in the histories *before* breaking, so
            # backpointer reconstruction can include it.
            parent_history.append(cand_parent.clone())
            gen_history.append(cand_gen.clone())
            found = True
            found_step = i_step
            found_local_idx = int(torch.where(eq_dest)[0][0].item())
            break

        # 4. Ban states seen on the last steps_back_to_ban steps.
        if steps_back_to_ban > 0:
            new_hashes = torch.sum(candidates.to(torch.int64) * vec_hasher, dim=1)
            mask_new = ~torch.isin(new_hashes, vec_hashes_current.view(-1), assume_unique=False)
            if int(mask_new.sum().item()) == 0:
                early_exit_reason = "no_new_states"
                break
            candidates = candidates[mask_new]
            cand_parent = cand_parent[mask_new]
            cand_gen = cand_gen[mask_new]
            # Update hash storage (cyclic buffer over the last steps_back_to_ban rounds).
            cyclic_idx = (cyclic_idx + 1) % steps_back_to_ban
            new_hashes_kept = new_hashes[mask_new]
            n_kept = new_hashes_kept.numel()
            vec_hashes_current[:n_kept, cyclic_idx] = new_hashes_kept

        # 5. Score with the model and keep the beam_width cheapest.
        if candidates.shape[0] > beam_width:
            feat = states_tensor_to_features(candidates, n=sweep_cfg.n, k=args.koltsov3_k)
            q = predict_features_batched(model, feat, feature_cols, best_ntree_limit)
            # Tighter q = closer to identity.
            idx_np = np.argsort(q)[:beam_width]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            array_beam = candidates[idx, :]
            cand_parent = cand_parent[idx]
            cand_gen = cand_gen[idx]
            last_q_min = float(q[idx_np].min())
            last_q_max = float(q[idx_np].max())
            last_q_median = float(np.median(q[idx_np]))
        else:
            array_beam = candidates

        parent_history.append(cand_parent.clone())
        gen_history.append(cand_gen.clone())

    # -------- Path reconstruction --------
    path_moves: List[int] = []
    path_verified = False
    if found and found_local_idx >= 0:
        # parent_history / gen_history have len = found_step + 1 (entry 0 is
        # the start state, entry t is step t). Walk back from the matching
        # row in the final layer to step 0.
        idx_cursor = found_local_idx
        for t in range(found_step, 0, -1):
            gen_used = int(gen_history[t][idx_cursor].item())
            path_moves.append(gen_used)
            idx_cursor = int(parent_history[t][idx_cursor].item())
        path_moves.reverse()
        # Verify by re-applying.
        verify_state = state_start.view(state_size).to(device=device, dtype=torch.long).clone()
        for g in path_moves:
            verify_state = verify_state[generators_tensor[g]]
        path_verified = bool(torch.equal(verify_state, state_destination.to(verify_state.dtype)))

    PATH_JSON_CAP = 500
    if len(path_moves) > PATH_JSON_CAP:
        # Keep first 250 and last 250 with a marker so JSON stays readable.
        path_for_json = path_moves[: PATH_JSON_CAP // 2] + [-1] + path_moves[-PATH_JSON_CAP // 2 :]
    else:
        path_for_json = path_moves

    return {
        "beam_found": bool(found),
        "beam_found_step": int(found_step) if found else -1,
        "beam_steps_taken": int(found_step) if found else int(n_steps_limit),
        "beam_exit_reason": "found" if found else early_exit_reason,
        "beam_final_q_min": last_q_min,
        "beam_final_q_max": last_q_max,
        "beam_final_q_median": last_q_median,
        "beam_path_length": len(path_moves),
        "beam_path_verified": path_verified,
        "beam_path_moves_json": json.dumps(path_for_json),
    }


def run_beam_phase(
    model: xgb.Booster,
    best_ntree_limit: int,
    feature_cols: List[str],
    sweep_cfg: SweepConfig,
    args: argparse.Namespace,
    generators: Sequence[np.ndarray],
    identity_state: torch.Tensor,
    walk_length: int,
    target_scale: float,
    device: torch.device,
    wandb_run: Optional[Any] = None,
    log_step_offset: int = 0,
    bfs_distance_map: Optional[Dict[Tuple[int, ...], int]] = None,
) -> Dict[str, Any]:
    """Run n_scrambles beam-search rollouts and aggregate. Returns summary stats.

    If bfs_distance_map is provided (small n where exact BFS fit in memory),
    each rollout also records the true shortest-path distance from state_start
    to identity, plus an optimality ratio beam_steps_taken / optimal_steps.
    """
    generators_tensor = normalize_generators(generators, device)
    n = sweep_cfg.n
    n_steps_limit = sweep_cfg.n_steps_limit_mult * n * n
    # Scramble depth follows the notebook (cell 76 solve_random_state):
    # n_generators**3 * mult, default mult=100 i.e. 27 * 100 = 2700 random
    # moves. This deep scramble is what the paper's beam-search comparison
    # uses; it dominates walk_length and makes the start state genuinely far
    # from identity. To compare directly with teammates' neural-net runs we
    # want the same value here.
    n_gens = generators_tensor.shape[0]
    n_scramble_steps = n_gens ** 3 * sweep_cfg.beam_scramble_depth_mult

    per_run_rows: List[Dict[str, Any]] = []
    for run_idx in range(sweep_cfg.n_scrambles):
        state_start = scramble_state(identity_state, generators_tensor, n_scramble_steps, device)
        t0 = time.time()
        result = run_beam_search_once(
            model=model,
            best_ntree_limit=best_ntree_limit,
            feature_cols=feature_cols,
            state_start=state_start,
            state_destination=identity_state,
            generators_tensor=generators_tensor,
            beam_width=sweep_cfg.beam_width,
            n_steps_limit=n_steps_limit,
            steps_back_to_ban=sweep_cfg.beam_steps_back_to_ban,
            target_scale=target_scale,
            args=args,
            sweep_cfg=sweep_cfg,
            device=device,
        )
        result["beam_run_sec"] = time.time() - t0
        result["beam_run_idx"] = run_idx
        result["beam_scramble_steps"] = n_scramble_steps

        # Optimal-distance lookup. With first-visit BFS from identity, the
        # distance map gives the true shortest-path distance. For start states
        # not in the map (would only happen if the graph were disconnected,
        # which it isn't for Koltsov3), we record -1 / nan to make the gap
        # visible rather than silent.
        if bfs_distance_map is not None:
            start_tuple = tuple(int(x) for x in state_start.detach().cpu().tolist())
            optimal_steps = bfs_distance_map.get(start_tuple, -1)
            result["beam_optimal_steps"] = int(optimal_steps)
            if result["beam_found"] and optimal_steps > 0:
                result["beam_optimality_ratio"] = (
                    result["beam_steps_taken"] / float(optimal_steps)
                )
            else:
                result["beam_optimality_ratio"] = float("nan")
        else:
            result["beam_optimal_steps"] = -1
            result["beam_optimality_ratio"] = float("nan")

        per_run_rows.append(result)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "beam/run_idx": run_idx,
                    "beam/found": int(result["beam_found"]),
                    "beam/steps_taken": result["beam_steps_taken"],
                    "beam/run_sec": result["beam_run_sec"],
                    **(
                        {
                            "beam/optimal_steps": result["beam_optimal_steps"],
                            "beam/optimality_ratio": result["beam_optimality_ratio"],
                        }
                        if result["beam_optimal_steps"] >= 0
                        else {}
                    ),
                },
                step=log_step_offset + run_idx,
            )
        if args.verbose_beam:
            opt_str = ""
            if result["beam_optimal_steps"] >= 0:
                ratio = result["beam_optimality_ratio"]
                ratio_str = "n/a" if (ratio != ratio) else f"{ratio:.2f}"
                opt_str = (
                    f", optimal={result['beam_optimal_steps']}, "
                    f"ratio={ratio_str}"
                )
            print(
                f"  Beam run {run_idx}: found={result['beam_found']}, "
                f"steps={result['beam_steps_taken']}{opt_str}, "
                f"reason={result['beam_exit_reason']}, time={result['beam_run_sec']:.1f}s",
                flush=True,
            )

    found_flags = [r["beam_found"] for r in per_run_rows]
    steps = [r["beam_steps_taken"] for r in per_run_rows]

    # Aggregate optimality stats only over runs where we both have an optimal
    # answer and the beam actually found a solution; otherwise the ratio is
    # nan.
    optimal_runs = [r for r in per_run_rows if r["beam_optimal_steps"] >= 0]
    opt_known_rate = len(optimal_runs) / max(len(per_run_rows), 1)
    found_with_optimal = [
        r for r in optimal_runs if r["beam_found"] and r["beam_optimal_steps"] > 0
    ]
    if found_with_optimal:
        mean_optimal_steps = float(
            np.mean([r["beam_optimal_steps"] for r in found_with_optimal])
        )
        mean_opt_ratio = float(
            np.mean([r["beam_optimality_ratio"] for r in found_with_optimal])
        )
    else:
        mean_optimal_steps = float("nan")
        mean_opt_ratio = float("nan")

    return {
        "beam_n_runs": len(per_run_rows),
        "beam_found_rate": float(np.mean(found_flags)) if per_run_rows else np.nan,
        "beam_mean_steps": float(np.mean(steps)) if per_run_rows else np.nan,
        "beam_median_steps": float(np.median(steps)) if per_run_rows else np.nan,
        "beam_min_steps": int(np.min(steps)) if per_run_rows else -1,
        "beam_max_steps": int(np.max(steps)) if per_run_rows else -1,
        "beam_per_run_json": json.dumps(per_run_rows),
        "beam_n_steps_limit": n_steps_limit,
        "beam_scramble_steps": n_scramble_steps,
        "beam_optimal_known_rate": opt_known_rate,
        "beam_mean_optimal_steps": mean_optimal_steps,
        "beam_mean_optimality_ratio": mean_opt_ratio,
    }


# -----------------------------
# Train one configuration
# -----------------------------


def train_one_config(
    sweep_cfg: SweepConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    sweep_group_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    set_seed(sweep_cfg.seed)

    n = sweep_cfg.n
    walk_length = get_random_walk_length(
        n, sweep_cfg.walk_length_multiplier, args.walk_length_mode
    )
    generators, identity_state, dtype_state = build_problem(n=n, koltsov3_k=args.koltsov3_k, device=device)

    print(f"\n----- Starting {sweep_cfg.run_name} -----", flush=True)
    print(f"walk_length={walk_length}, device={device}", flush=True)

    # Unlike the MLP script (which regenerated training walks every epoch),
    # XGBoost trains once, so training/validation/test data are each generated
    # exactly once for this configuration.
    data_start = time.time()
    X_train_states, y_train_t = random_walks(
        generators,
        n_random_walk_length=walk_length,
        n_random_walks_to_generate=sweep_cfg.n_random_walks_to_generate,
        n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
        random_walks_type=sweep_cfg.random_walks_type,
        state_rw_start=identity_state,
        dtype_state=dtype_state,
        device=device,
        dedup_strategy=args.dedup_strategy,
    )
    X_val_states, y_val_t = random_walks(
        generators,
        n_random_walk_length=walk_length,
        n_random_walks_to_generate=sweep_cfg.n_val_samples,
        n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
        random_walks_type=sweep_cfg.random_walks_type,
        state_rw_start=identity_state,
        dtype_state=dtype_state,
        device=device,
        dedup_strategy=args.dedup_strategy,
    )
    X_test_states, y_test_t = random_walks(
        generators,
        n_random_walk_length=walk_length,
        n_random_walks_to_generate=sweep_cfg.n_test_samples,
        n_random_walks_steps_back_to_ban=sweep_cfg.n_random_walks_steps_back_to_ban,
        random_walks_type=sweep_cfg.random_walks_type,
        state_rw_start=identity_state,
        dtype_state=dtype_state,
        device=device,
        dedup_strategy=args.dedup_strategy,
    )

    # Optional cap on training rows to keep feature matrices manageable for large n.
    n_generated_train = int(X_train_states.shape[0])
    if args.max_train_samples > 0 and n_generated_train > args.max_train_samples:
        keep = torch.randperm(n_generated_train)[: args.max_train_samples]
        X_train_states = X_train_states[keep]
        y_train_t = y_train_t[keep]
        print(
            f"Subsampled training states from {n_generated_train} to {args.max_train_samples}",
            flush=True,
        )

    y_train = y_train_t.detach().cpu().numpy()
    y_val = y_val_t.detach().cpu().numpy()
    y_test = y_test_t.detach().cpu().numpy()

    X_train = states_tensor_to_features(X_train_states, n=n, k=args.koltsov3_k)
    X_val = states_tensor_to_features(X_val_states, n=n, k=args.koltsov3_k)
    X_test = states_tensor_to_features(X_test_states, n=n, k=args.koltsov3_k)
    feature_cols = list(X_train.columns)
    data_time_sec = time.time() - data_start
    print(f"data+features ready in {data_time_sec:.1f}s, {len(feature_cols)} features", flush=True)

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
            tags=["koltsov3", "random-walk", "xgboost", "gradient-boosting", "general-sweep"],
            config={
                **asdict(sweep_cfg),
                "walk_length": walk_length,
                "koltsov3_k": args.koltsov3_k,
                "device_arg": args.device,
                "walk_length_mode": args.walk_length_mode,
                "n_features": len(feature_cols),
                "early_stopping_rounds": args.early_stopping_rounds,
            },
            resume="never",
            reinit="finish_previous",
        )

    # -------- Train XGBoost --------
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": sweep_cfg.max_depth,
        "learning_rate": sweep_cfg.learning_rate,
        "subsample": sweep_cfg.subsample,
        "colsample_bytree": sweep_cfg.colsample_bytree,
        "min_child_weight": sweep_cfg.min_child_weight,
        "reg_lambda": sweep_cfg.reg_lambda,
        "reg_alpha": sweep_cfg.reg_alpha,
        "seed": sweep_cfg.seed,
        "nthread": args.nthread,
    }
    if device.type == "cuda":
        params["device"] = "cuda"

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    fit_start = time.time()
    early_stopping = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=sweep_cfg.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=early_stopping,
        verbose_eval=args.verbose_eval if args.verbose_eval > 0 else False,
    )
    fit_time_sec = time.time() - fit_start

    best_iteration = int(getattr(model, "best_iteration", model.num_boosted_rounds() - 1))
    best_ntree_limit = best_iteration + 1

    # -------- Predict --------
    predict_start = time.time()
    iteration_range = (0, best_ntree_limit)
    train_pred = model.predict(dtrain, iteration_range=iteration_range)
    val_pred = model.predict(dval, iteration_range=iteration_range)
    test_pred = model.predict(dtest, iteration_range=iteration_range)
    predict_time_sec = time.time() - predict_start

    train_metrics = evaluate_predictions(y_train, train_pred)
    val_metrics = evaluate_predictions(y_val, val_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    # -------- Phase 2 (MDQN refinement, optional) --------
    target_scale = float(max(walk_length, 1))
    mdqn_history: List[Dict[str, Any]] = []
    mdqn_time_sec = 0.0
    post_mdqn_train_metrics = train_metrics
    post_mdqn_val_metrics = val_metrics
    post_mdqn_test_metrics = test_metrics
    if sweep_cfg.n_epochs_dqn > 0:
        if args.dedup_strategy != "first-visit":
            raise ValueError(
                "MDQN requires --dedup-strategy first-visit so that each state has a "
                "well-defined first-visit step k for the Algorithm 4 line 7 clip."
            )
        print(
            f"  Phase 2 (MDQN): {sweep_cfg.n_epochs_dqn} epochs, "
            f"{sweep_cfg.dqn_n_random_walks} walks/epoch, clip={sweep_cfg.dqn_clip}",
            flush=True,
        )
        mdqn_start = time.time()
        model, best_ntree_limit, mdqn_history = run_mdqn_phase(
            model=model,
            initial_best_ntree_limit=best_ntree_limit,
            sweep_cfg=sweep_cfg,
            args=args,
            generators=generators,
            identity_state=identity_state,
            dtype_state=dtype_state,
            device=device,
            feature_cols=feature_cols,
            walk_length=walk_length,
            target_scale=target_scale,
            wandb_run=wandb_run,
            log_step_offset=int(sweep_cfg.n_estimators) + 1,
        )
        mdqn_time_sec = time.time() - mdqn_start
        # Re-evaluate on the same val/test sets with the refined model so we can
        # report MDQN's effect. The labels on those sets are still "diffusion
        # distance" (normalized RW depth) -- MDQN may shift the model's
        # predictions away from those labels, so post-MDQN val/test RMSE can
        # legitimately go UP even though heuristic quality goes up. The beam
        # search phase is the real downstream test of heuristic quality.
        post_iter_range = (0, best_ntree_limit)
        post_train_pred = model.predict(dtrain, iteration_range=post_iter_range)
        post_val_pred = model.predict(dval, iteration_range=post_iter_range)
        post_test_pred = model.predict(dtest, iteration_range=post_iter_range)
        post_mdqn_train_metrics = evaluate_predictions(y_train, post_train_pred)
        post_mdqn_val_metrics = evaluate_predictions(y_val, post_val_pred)
        post_mdqn_test_metrics = evaluate_predictions(y_test, post_test_pred)
        print(
            f"  Phase 2 done in {mdqn_time_sec:.1f}s. "
            f"val_rmse: {val_metrics['rmse']:.5f} -> {post_mdqn_val_metrics['rmse']:.5f}, "
            f"test_rmse: {test_metrics['rmse']:.5f} -> {post_mdqn_test_metrics['rmse']:.5f}",
            flush=True,
        )

    # -------- BFS metadata (computed before Phase 3 so the distance map is
    # available for optimal-path comparison) --------
    bfs_metadata: Dict[str, Any] = {
        "bfs_computed": False,
        "bfs_skip_reason": "disabled",
        "diameter": np.nan,
        "last_layer_count": np.nan,
        "layer_sizes": "",
    }
    bfs_distance_map: Optional[Dict[Tuple[int, ...], int]] = None
    if args.compute_bfs_metadata:
        print("Computing exact BFS metadata. This is only safe for small n.", flush=True)
        bfs_metadata, bfs_distance_map = compute_exact_bfs_metadata(
            n, generators, max_states=args.max_bfs_states, return_distance_map=True
        )

    # -------- Phase 3 (guided beam search, optional) --------
    beam_summary: Dict[str, Any] = {
        "beam_n_runs": 0,
        "beam_found_rate": np.nan,
        "beam_mean_steps": np.nan,
        "beam_median_steps": np.nan,
        "beam_min_steps": -1,
        "beam_max_steps": -1,
        "beam_per_run_json": "",
        "beam_n_steps_limit": -1,
        "beam_scramble_steps": -1,
        "beam_optimal_known_rate": np.nan,
        "beam_mean_optimal_steps": np.nan,
        "beam_mean_optimality_ratio": np.nan,
    }
    beam_time_sec = 0.0
    if args.run_beam_search and sweep_cfg.n_scrambles > 0:
        print(
            f"  Phase 3 (beam): width={sweep_cfg.beam_width}, "
            f"n_steps_limit={sweep_cfg.n_steps_limit_mult * n * n}, "
            f"steps_back_to_ban={sweep_cfg.beam_steps_back_to_ban}, "
            f"n_scrambles={sweep_cfg.n_scrambles}, "
            f"bfs_known={bfs_distance_map is not None}",
            flush=True,
        )
        beam_start = time.time()
        beam_summary = run_beam_phase(
            model=model,
            best_ntree_limit=best_ntree_limit,
            feature_cols=feature_cols,
            sweep_cfg=sweep_cfg,
            args=args,
            generators=generators,
            identity_state=identity_state,
            walk_length=walk_length,
            target_scale=target_scale,
            device=device,
            wandb_run=wandb_run,
            log_step_offset=int(sweep_cfg.n_estimators) + int(sweep_cfg.n_epochs_dqn) + 2,
            bfs_distance_map=bfs_distance_map,
        )
        beam_time_sec = time.time() - beam_start
        print(
            f"  Phase 3 done in {beam_time_sec:.1f}s. "
            f"found_rate={beam_summary['beam_found_rate']:.2f}, "
            f"mean_steps={beam_summary['beam_mean_steps']:.1f}, "
            f"mean_opt_ratio={beam_summary['beam_mean_optimality_ratio']}",
            flush=True,
        )

    # -------- Per-iteration history (the "epoch" CSV equivalent) --------
    # XGBoost's evals_result only carries train/val RMSE round-by-round. To get
    # training-curve parity with Merav's MLP W&B view (which shows r2 + Spearman
    # over time as well), recompute val r2/Spearman every val_metric_every
    # rounds via model.predict with a partial iteration_range. The cost is one
    # prediction every N rounds -- cheap relative to training.
    train_rmse_hist = evals_result.get("train", {}).get("rmse", [])
    val_rmse_hist = evals_result.get("val", {}).get("rmse", [])
    metric_every = max(1, args.val_metric_every)
    iteration_rows: List[Dict[str, Any]] = []
    for it, (tr_rmse, va_rmse) in enumerate(zip(train_rmse_hist, val_rmse_hist)):
        is_metric_round = (it + 1) % metric_every == 0 or it == len(train_rmse_hist) - 1
        val_r2_it = np.nan
        val_spearman_it = np.nan
        if is_metric_round:
            val_pred_it = model.predict(dval, iteration_range=(0, it + 1))
            metrics_it = evaluate_predictions(y_val, val_pred_it)
            val_r2_it = metrics_it["r2"]
            val_spearman_it = metrics_it["spearman"]

        row = {
            "config_id": sweep_cfg.run_name,
            **asdict(sweep_cfg),
            "iteration": it,
            "walk_length": walk_length,
            "train_rmse": tr_rmse,
            "val_rmse": va_rmse,
            "val_r2": val_r2_it,
            "val_spearman": val_spearman_it,
            "n_features": len(feature_cols),
        }
        iteration_rows.append(row)
        if wandb_run is not None:
            log_payload: Dict[str, Any] = {
                "iteration": it,
                "train/rmse": tr_rmse,
                "val/rmse": va_rmse,
            }
            if is_metric_round:
                if not np.isnan(val_r2_it):
                    log_payload["val/r2"] = val_r2_it
                if not np.isnan(val_spearman_it):
                    log_payload["val/spearman"] = val_spearman_it
            wandb_run.log(log_payload, step=it)

    # -------- Feature importance (gain) --------
    gain_scores = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        {"feature": list(gain_scores.keys()), "gain": list(gain_scores.values())}
    ).sort_values("gain", ascending=False)
    top_features = importance_df.head(args.top_features_to_record)
    top_features_json = json.dumps(
        [{"feature": r.feature, "gain": float(r.gain)} for r in top_features.itertuples(index=False)]
    )

    importance_dir = output_dir / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sweep_cfg.run_name.replace("/", "_").replace(" ", "_")[:160]
    importance_df.to_csv(importance_dir / f"importance_{safe_id}.csv", index=False)

    if args.save_plots and len(importance_df) > 0:
        top_plot = importance_df.head(20).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_plot)), top_plot["gain"].values, color="seagreen")
        plt.yticks(range(len(top_plot)), top_plot["feature"].values, fontsize=8)
        plt.xlabel("gain")
        plt.title(f"Top features by gain\n{safe_id}", fontsize=9)
        plt.tight_layout()
        plt.savefig(importance_dir / f"importance_{safe_id}.png", dpi=150)
        plt.close()

    # -------- Dataset diagnostics --------
    n_train_states = int(X_train_states.shape[0])
    n_val_states = int(X_val_states.shape[0])
    n_test_states = int(X_test_states.shape[0])
    num_unique_train_states = unique_state_count(X_train_states)
    num_unique_val_states = unique_state_count(X_val_states)
    num_unique_test_states = unique_state_count(X_test_states)
    label_summary = label_stats(y_train_t, y_val_t, y_test_t)

    # bfs_metadata was computed earlier (before Phase 3) so the distance map
    # could be used for the optimal-path comparison; it's reused here for the
    # summary row.

    raw_train_rows_generated = sweep_cfg.n_random_walks_to_generate * walk_length
    train_dedup_factor = (
        raw_train_rows_generated / max(int(X_train_states.shape[0]), 1)
        if args.dedup_strategy == "first-visit"
        else 1.0
    )

    summary_row = {
        "config_id": sweep_cfg.run_name,
        **asdict(sweep_cfg),
        "walk_length": walk_length,
        "walk_length_mode": args.walk_length_mode,
        "state_space_size": math.factorial(n),
        "dedup_strategy": args.dedup_strategy,
        "raw_train_rows_generated": raw_train_rows_generated,
        "train_dedup_factor": train_dedup_factor,
        "graph_family": "koltsov3",
        "koltsov3_k": args.koltsov3_k,
        "device": str(device),
        "n_features": len(feature_cols),
        "best_iteration": best_iteration,
        "n_boosted_rounds": int(model.num_boosted_rounds()),
        "final_train_rmse": float(train_rmse_hist[-1]) if train_rmse_hist else np.nan,
        "final_val_rmse": float(val_rmse_hist[-1]) if val_rmse_hist else np.nan,
        "best_val_rmse_during_training": float(np.min(val_rmse_hist)) if val_rmse_hist else np.nan,
        "best_iteration_by_val_rmse": int(np.argmin(val_rmse_hist)) if val_rmse_hist else np.nan,
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
        "train_val_rmse_gap": val_metrics["rmse"] - train_metrics["rmse"],
        "train_test_rmse_gap": test_metrics["rmse"] - train_metrics["rmse"],
        "val_test_rmse_gap": test_metrics["rmse"] - val_metrics["rmse"],
        "top_features_by_gain": top_features_json,
        "data_time_sec": data_time_sec,
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
        # Phase 2 (MDQN). When n_epochs_dqn == 0 these mirror the Phase 1 values.
        "mdqn_ran": sweep_cfg.n_epochs_dqn > 0,
        "mdqn_time_sec": mdqn_time_sec,
        "post_mdqn_train_rmse": post_mdqn_train_metrics["rmse"],
        "post_mdqn_val_rmse": post_mdqn_val_metrics["rmse"],
        "post_mdqn_test_rmse": post_mdqn_test_metrics["rmse"],
        "post_mdqn_train_r2": post_mdqn_train_metrics["r2"],
        "post_mdqn_val_r2": post_mdqn_val_metrics["r2"],
        "post_mdqn_test_r2": post_mdqn_test_metrics["r2"],
        "post_mdqn_train_spearman": post_mdqn_train_metrics["spearman"],
        "post_mdqn_val_spearman": post_mdqn_val_metrics["spearman"],
        "post_mdqn_test_spearman": post_mdqn_test_metrics["spearman"],
        "mdqn_final_train_rmse": (
            mdqn_history[-1]["dqn_train_rmse"] if mdqn_history else np.nan
        ),
        # Phase 3 (beam search).
        "beam_ran": bool(args.run_beam_search and sweep_cfg.n_scrambles > 0),
        "beam_time_sec": beam_time_sec,
        **beam_summary,
        **bfs_metadata,
    }

    if wandb_run is not None:
        wandb_run.log({f"final/{k}": v for k, v in summary_row.items() if isinstance(v, (int, float, str, bool))})
        wandb_run.finish()

    # Stamp config_id on every MDQN row so the CSV can be joined back to summary.
    for row in mdqn_history:
        row["config_id"] = sweep_cfg.run_name

    print(
        f"Finished {sweep_cfg.run_name}: "
        f"test_rmse={test_metrics['rmse']:.5f}, test_r2={test_metrics['r2']:.4f}, "
        f"test_spearman={test_metrics['spearman']:.4f}, best_iter={best_iteration}",
        flush=True,
    )
    return summary_row, iteration_rows, mdqn_history


# -----------------------------
# Output helpers
# -----------------------------


def save_plots(df_summary: pd.DataFrame, df_iters: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Compact overview: one point per configuration.
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

    # max_depth trend plots when max_depth is one of the varied dimensions.
    group_cols = [
        "n",
        "n_random_walks_to_generate",
        "walk_length_multiplier",
        "random_walks_type",
        "n_random_walks_steps_back_to_ban",
        "n_estimators",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "n_val_samples",
        "n_test_samples",
        "seed",
        "n_epochs_dqn",
        "dqn_n_random_walks",
        "dqn_clip",
        "beam_width",
        "n_steps_limit_mult",
        "beam_steps_back_to_ban",
        "n_scrambles",
        "beam_scramble_depth_mult",
    ]
    for _group_key, df_g in df_summary.groupby(group_cols, dropna=False):
        if df_g["max_depth"].nunique() < 2:
            continue
        df_g = df_g.sort_values("max_depth")
        n = df_g["n"].iloc[0]
        suffix = (
            f"n{n}_rw{df_g['n_random_walks_to_generate'].iloc[0]}_"
            f"wlm{df_g['walk_length_multiplier'].iloc[0]}_"
            f"ne{df_g['n_estimators'].iloc[0]}_seed{df_g['seed'].iloc[0]}"
        )

        plt.figure(figsize=(8, 5))
        plt.plot(df_g["max_depth"], df_g["train_rmse"], marker="o", label="train")
        plt.plot(df_g["max_depth"], df_g["val_rmse"], marker="o", label="val")
        plt.plot(df_g["max_depth"], df_g["test_rmse"], marker="o", label="test")
        plt.title(f"RMSE vs max_depth ({suffix})")
        plt.xlabel("max_depth")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"rmse_vs_max_depth_{suffix}.png", dpi=200)
        plt.close()

    # Per-iteration training curves for a limited number of configs.
    max_iter_plots = 25
    for idx, (config_id, df_cfg) in enumerate(df_iters.groupby("config_id")):
        if idx >= max_iter_plots:
            break
        df_cfg = df_cfg.sort_values("iteration")
        safe_id = str(config_id).replace("/", "_").replace(" ", "_")[:160]

        plt.figure(figsize=(8, 5))
        plt.plot(df_cfg["iteration"], df_cfg["train_rmse"], label="train")
        plt.plot(df_cfg["iteration"], df_cfg["val_rmse"], label="val")
        plt.title(f"RMSE by boosting iteration\n{safe_id}", fontsize=9)
        plt.xlabel("boosting iteration")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"rmse_by_iteration_{safe_id}.png", dpi=200)
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generalized Koltsov3 random-walk XGBoost sweeps.")

    # Data / random-walk axes.
    parser.add_argument("--n-values", type=parse_int_list, default=[8, 12, 16, 24], help="Comma-separated n values")
    parser.add_argument("--n-random-walks-values", type=parse_int_list, default=[2000], help="Training random-walk counts")
    parser.add_argument(
        "--dedup-strategy",
        choices=["none", "first-visit"],
        default="none",
        help=(
            "Label dedup applied to train/val/test walks. 'none' (default) keeps every "
            "(state, step) row; 'first-visit' keeps one row per unique state labeled by "
            "its earliest visit step (paper's diffusion-distance spec). Different "
            "strategies are different experiments (the target label is a different "
            "function), so val/test follow the same dedup as train."
        ),
    )
    parser.add_argument(
        "--walks-per-n",
        type=parse_walks_per_n,
        default=None,
        help=(
            "Optional per-n walk override, e.g. '5:500,16:25000,24:25000'. "
            "Any n listed here uses the mapped walk count; n values not listed "
            "fall back to --n-random-walks-values (Cartesian product)."
        ),
    )
    parser.add_argument("--walk-length-multipliers", type=parse_int_list, default=[8], help="walk_length = multiplier * n (only used when --walk-length-mode multiplier)")
    parser.add_argument(
        "--walk-length-mode",
        choices=["multiplier", "min_npairs_or_16n"],
        default="multiplier",
        help=(
            "How walk_length is computed. 'multiplier' (default): mult * n, "
            "driven by --walk-length-multipliers. 'min_npairs_or_16n': "
            "min(n*(n-1)//2, 16n) -- Danny's Algorithm-4 default with a 16n "
            "cap so n=64 doesn't explode to 2016-step walks."
        ),
    )
    parser.add_argument("--random-walk-types", type=parse_str_list, default=["non-backtracking-beam"], help="simple,non-backtracking-beam")
    parser.add_argument("--steps-back-to-ban-values", type=parse_int_list, default=[2], help="Previous move counts to ban")
    parser.add_argument("--n-val-samples-values", type=parse_int_list, default=[300], help="Validation random-walk counts")
    parser.add_argument("--n-test-samples-values", type=parse_int_list, default=[300], help="Test random-walk counts")
    parser.add_argument("--seed-values", type=parse_int_list, default=[0], help="Comma-separated seeds")

    # XGBoost hyperparameter axes.
    parser.add_argument("--n-estimators-values", type=parse_int_list, default=[600], help="num_boost_round ceilings")
    parser.add_argument("--max-depth-values", type=parse_int_list, default=[6], help="XGBoost max_depth values")
    parser.add_argument("--learning-rate-values", type=parse_float_list, default=[0.05], help="XGBoost learning rates")
    parser.add_argument("--subsample-values", type=parse_float_list, default=[0.8], help="Row subsample fractions")
    parser.add_argument("--colsample-bytree-values", type=parse_float_list, default=[0.8], help="Column subsample fractions")
    parser.add_argument("--min-child-weight-values", type=parse_float_list, default=[5.0], help="min_child_weight values")
    parser.add_argument("--reg-lambda-values", type=parse_float_list, default=[1.0], help="L2 regularization values")
    parser.add_argument("--reg-alpha-values", type=parse_float_list, default=[0.0], help="L1 regularization values")

    # Training controls.
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Early stopping patience; 0 disables")
    parser.add_argument("--max-train-samples", type=int, default=500_000, help="Cap on training rows; 0 disables")
    parser.add_argument("--nthread", type=int, default=-1, help="XGBoost threads; -1 uses all")
    parser.add_argument("--verbose-eval", type=int, default=0, help="XGBoost log period; 0 silences")
    parser.add_argument("--top-features-to-record", type=int, default=25, help="Top-gain features stored in the summary CSV")
    parser.add_argument("--val-metric-every", type=int, default=50, help="Recompute val r2/Spearman every N boosting rounds for the W&B training curve")

    # Phase 2 (MDQN) sweep axes.
    parser.add_argument(
        "--n-epochs-dqn-values",
        type=parse_int_list,
        default=[0],
        help="MDQN epochs per config. 0 disables Phase 2. Algorithm 4 outer-loop length M.",
    )
    parser.add_argument(
        "--dqn-n-random-walks-values",
        type=parse_int_list,
        default=[2000],
        help="Walks regenerated per MDQN epoch. Algorithm 4 inner N.",
    )
    parser.add_argument(
        "--dqn-clip-values",
        type=parse_bool_list,
        default=[True],
        help="Apply the [0, k] clip from Algorithm 4 line 7. Comma-separated true/false.",
    )
    parser.add_argument("--verbose-mdqn", type=str2bool, nargs="?", const=True, default=True, help="Print per-MDQN-epoch lines.")

    # Phase 3 (beam search) sweep axes + master switch.
    parser.add_argument(
        "--run-beam-search",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Master switch for Phase 3. Per-config beam search is only run when this is true AND n_scrambles > 0.",
    )
    parser.add_argument(
        "--beam-width-values",
        type=parse_int_list,
        default=[1024],
        help="Beam widths to try in Phase 3.",
    )
    parser.add_argument(
        "--n-steps-limit-mult-values",
        type=parse_int_list,
        default=[4],
        help="Beam search step-limit multiplier: n_steps_limit = mult * n**2.",
    )
    parser.add_argument(
        "--beam-steps-back-to-ban-values",
        type=parse_int_list,
        default=[8],
        help="Per-beam-search history depth for non-backtracking ban (cell 82's CFG['n_beam_search_steps_back_to_ban']).",
    )
    parser.add_argument(
        "--n-scrambles-values",
        type=parse_int_list,
        default=[1],
        help="Number of independent beam-search rollouts per config. Each uses a fresh scrambled start state.",
    )
    parser.add_argument(
        "--beam-scramble-depth-mult-values",
        type=parse_int_list,
        default=[100],
        help=(
            "Scramble depth multiplier: scramble_steps = n_generators**3 * mult. "
            "Notebook default is mult=100 (yielding ~2700 random moves for "
            "Koltsov3's 3 generators). Match teammates' scramble depth for "
            "apples-to-apples beam-search comparisons."
        ),
    )
    parser.add_argument("--verbose-beam", type=str2bool, nargs="?", const=True, default=True, help="Print per-beam-run lines.")

    # Output / problem.
    parser.add_argument("--output-dir", type=Path, default=Path("koltsov3_gb_pipeline_results"), help="Output directory")
    parser.add_argument(
        "--resume",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Resume from an existing --output-dir: load summary_results_partial.csv, "
            "skip configs whose config_id (== SweepConfig.run_name) already finished, "
            "and append new results to the same partial CSVs."
        ),
    )
    parser.add_argument("--koltsov3-k", type=int, default=0, help="k in the Koltsov3 S=(k,k+2) generator")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--save-plots", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--compute-bfs-metadata", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--max-bfs-states", type=int, default=50000, help="Safety cap for optional exact BFS metadata")

    # W&B.
    parser.add_argument("--use-wandb", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--wandb-entity", type=str, default="CayleyPy")
    parser.add_argument("--wandb-project", type=str, default="cayley-py")
    parser.add_argument("--wandb-group", type=str, default=None, help="Group name for the whole sweep")

    return parser


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    valid_walk_types = {"simple", "non-backtracking-beam"}
    invalid = sorted(set(args.random_walk_types) - valid_walk_types)
    if invalid:
        raise ValueError(f"Invalid random walk type(s): {invalid}. Valid options: {sorted(valid_walk_types)}")

    # steps_back_to_ban only affects non-backtracking-beam walks. With simple
    # walks every step samples uniformly, so the ban value is silently ignored
    # but still multiplies the Cartesian config count - flag that.
    non_zero_bans = [b for b in args.steps_back_to_ban_values if b != 0]
    if "non-backtracking-beam" not in args.random_walk_types and non_zero_bans:
        print(
            f"WARNING: --steps-back-to-ban-values={args.steps_back_to_ban_values} has no "
            "effect with --random-walk-types=simple. It is multiplied into the Cartesian "
            "config count but produces identical walks. Use [0] to drop this axis.",
            flush=True,
        )

    # MDQN <-> dedup invariant. The algorithm clips d(s) to [0, k] where k is
    # the first-visit step. Without first-visit dedup, k is ambiguous (multiple
    # k's per state), so the clip is meaningless. Fail fast instead of silently
    # mis-clipping.
    any_mdqn = any(v > 0 for v in args.n_epochs_dqn_values)
    if any_mdqn and args.dedup_strategy != "first-visit":
        raise ValueError(
            "MDQN is enabled (n_epochs_dqn > 0) but --dedup-strategy is not "
            "'first-visit'. Algorithm 4 line 7 clips d(s) to [0, k] where k is "
            "the first-visit step, which is only well-defined under first-visit "
            "dedup. Pass --dedup-strategy first-visit."
        )

    other_axes = 1
    for values in [
        args.walk_length_multipliers,
        args.random_walk_types,
        args.steps_back_to_ban_values,
        args.n_estimators_values,
        args.max_depth_values,
        args.learning_rate_values,
        args.subsample_values,
        args.colsample_bytree_values,
        args.min_child_weight_values,
        args.reg_lambda_values,
        args.reg_alpha_values,
        args.n_val_samples_values,
        args.n_test_samples_values,
        args.seed_values,
        args.n_epochs_dqn_values,
        args.dqn_n_random_walks_values,
        args.dqn_clip_values,
        args.beam_width_values,
        args.n_steps_limit_mult_values,
        args.beam_steps_back_to_ban_values,
        args.n_scrambles_values,
        args.beam_scramble_depth_mult_values,
    ]:
        other_axes *= len(values)

    n_configs = 0
    for n in args.n_values:
        n_configs += len(walks_for_n(args, n)) * other_axes
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

    sweep_group_name = args.wandb_group or f"koltsov3_gb_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70, flush=True)
    print(f"  Koltsov3 GB pipeline  -  {args.n_total_configs} configurations", flush=True)
    print("=" * 70, flush=True)
    print(f"  output_dir:        {output_dir}", flush=True)
    print(f"  sweep_group_name:  {sweep_group_name}", flush=True)
    print(f"  device:            {device}", flush=True)
    print(f"  xgboost:           {xgb.__version__}", flush=True)
    print("  --- data ---", flush=True)
    print(f"  n_values:          {args.n_values}", flush=True)
    walks_plan = ", ".join(f"n={n}:{walks_for_n(args, n)}" for n in args.n_values)
    print(f"  walks per n:       {walks_plan}", flush=True)
    print(f"  walk_length_mode:  {args.walk_length_mode}", flush=True)
    print(f"  walk_length_mult:  {args.walk_length_multipliers}", flush=True)
    print(f"  walk_type:         {args.random_walk_types}", flush=True)
    print(f"  dedup_strategy:    {args.dedup_strategy}", flush=True)
    print("  --- model (Phase 1) ---", flush=True)
    print(f"  n_estimators:      {args.n_estimators_values}", flush=True)
    print(f"  max_depth:         {args.max_depth_values}", flush=True)
    print(f"  learning_rate:     {args.learning_rate_values}", flush=True)
    print(f"  early_stopping:    {args.early_stopping_rounds}", flush=True)
    print(f"  max_train_samples: {args.max_train_samples}", flush=True)
    print("  --- MDQN (Phase 2) ---", flush=True)
    print(f"  n_epochs_dqn:      {args.n_epochs_dqn_values}", flush=True)
    print(f"  dqn_n_random_walks:{args.dqn_n_random_walks_values}", flush=True)
    print(f"  dqn_clip:          {args.dqn_clip_values}", flush=True)
    print("  --- beam search (Phase 3) ---", flush=True)
    print(f"  run_beam_search:   {args.run_beam_search}", flush=True)
    print(f"  beam_width:        {args.beam_width_values}", flush=True)
    print(f"  n_steps_limit_mult:{args.n_steps_limit_mult_values}", flush=True)
    print(f"  beam_steps_back:   {args.beam_steps_back_to_ban_values}", flush=True)
    print(f"  n_scrambles:       {args.n_scrambles_values}", flush=True)
    print(f"  scramble_depth_m:  {args.beam_scramble_depth_mult_values}", flush=True)
    print("=" * 70, flush=True)

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run with --use-wandb false.")
        api_key = load_wandb_key()
        if api_key:
            wandb.login(key=api_key)
            print("W&B: logged in using WANDB_API_KEY from environment/.env", flush=True)
        else:
            print(
                "W&B: no WANDB_API_KEY found in environment or .env; "
                "relying on a prior `wandb login`.",
                flush=True,
            )

    summary_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []
    mdqn_rows: List[Dict[str, Any]] = []

    # --resume: load any prior partial CSVs in this output_dir, treat their
    # config_ids as already done, and seed the row lists so the next
    # incremental write produces a complete file rather than a stub.
    done_ids: set = set()
    if args.resume:
        summary_partial = output_dir / "summary_results_partial.csv"
        if summary_partial.is_file():
            df_prev = pd.read_csv(summary_partial)
            # A config is "done" only if test_rmse was written (i.e. the row is
            # complete -- guards against a half-written final row).
            if "test_rmse" in df_prev.columns:
                df_prev = df_prev[df_prev["test_rmse"].notna()]
            done_ids = set(df_prev["config_id"].astype(str).tolist())
            summary_rows = df_prev.to_dict("records")
            iters_partial = output_dir / "iteration_results_partial.csv"
            if iters_partial.is_file():
                df_iters_prev = pd.read_csv(iters_partial)
                df_iters_prev = df_iters_prev[df_iters_prev["config_id"].astype(str).isin(done_ids)]
                iteration_rows = df_iters_prev.to_dict("records")
            mdqn_partial = output_dir / "mdqn_results_partial.csv"
            if mdqn_partial.is_file():
                df_mdqn_prev = pd.read_csv(mdqn_partial)
                df_mdqn_prev = df_mdqn_prev[df_mdqn_prev["config_id"].astype(str).isin(done_ids)]
                mdqn_rows = df_mdqn_prev.to_dict("records")
            print(
                f"--resume: loaded {len(done_ids)} finished configs from "
                f"{summary_partial.name}; they will be skipped.",
                flush=True,
            )
        else:
            print(
                f"--resume: no {summary_partial.name} in {output_dir}; "
                "starting fresh.",
                flush=True,
            )

    all_configs = list(iter_sweep_configs(args))
    n_to_run = sum(1 for cfg in all_configs if cfg.run_name not in done_ids)
    if done_ids:
        print(
            f"Plan: {len(all_configs)} total configs, {len(done_ids)} already done, "
            f"{n_to_run} to run.",
            flush=True,
        )
    for i, sweep_cfg in enumerate(all_configs, start=1):
        if sweep_cfg.run_name in done_ids:
            print(
                f"\n===== Configuration {i}/{len(all_configs)} -- SKIP (resume) {sweep_cfg.run_name} =====",
                flush=True,
            )
            continue
        print(f"\n===== Configuration {i}/{len(all_configs)} =====", flush=True)
        summary_row, iteration_rows_for_config, mdqn_rows_for_config = train_one_config(
            sweep_cfg=sweep_cfg,
            args=args,
            device=device,
            output_dir=output_dir,
            sweep_group_name=sweep_group_name,
        )
        summary_rows.append(summary_row)
        iteration_rows.extend(iteration_rows_for_config)
        mdqn_rows.extend(mdqn_rows_for_config)

        # Incremental writes after every config -- if the Slurm ckpt partition
        # preempts the job, the partial CSVs AND plots stay representative of
        # whatever finished. save_plots() overwrites the same filenames, so the
        # cost is bounded and the final state is always "everything done so
        # far". Wrap in try/except so a single bad config doesn't lose progress.
        df_summary_partial = pd.DataFrame(summary_rows)
        df_iters_partial = pd.DataFrame(iteration_rows)
        df_summary_partial.to_csv(output_dir / "summary_results_partial.csv", index=False)
        df_iters_partial.to_csv(output_dir / "iteration_results_partial.csv", index=False)
        if mdqn_rows:
            pd.DataFrame(mdqn_rows).to_csv(output_dir / "mdqn_results_partial.csv", index=False)
        if args.save_plots:
            try:
                save_plots(df_summary_partial, df_iters_partial, output_dir)
            except Exception as exc:
                # Plotting is a convenience; don't let a matplotlib hiccup kill
                # the sweep. Print and move on; the next config's call will retry.
                print(f"  WARNING: incremental save_plots failed: {exc}", flush=True)

    df_summary = pd.DataFrame(summary_rows)
    df_iters = pd.DataFrame(iteration_rows)
    df_mdqn = pd.DataFrame(mdqn_rows)

    summary_csv = output_dir / "summary_results.csv"
    iteration_csv = output_dir / "iteration_results.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_iters.to_csv(iteration_csv, index=False)
    if not df_mdqn.empty:
        df_mdqn.to_csv(output_dir / "mdqn_results.csv", index=False)

    with open(output_dir / "run_args.json", "w", encoding="utf-8") as fh:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fh, indent=2)

    if args.save_plots:
        # Final pass with the complete dataset. Same files as the incremental
        # passes; final overwrite ensures everything reflects the full sweep.
        save_plots(df_summary, df_iters, output_dir)

    print(f"\nSaved final summary CSV to {summary_csv}", flush=True)
    print(f"Saved per-iteration CSV to {iteration_csv}", flush=True)
    print(f"Saved plots to {output_dir / 'plots'}", flush=True)
    print(f"Saved feature importances to {output_dir / 'feature_importance'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
