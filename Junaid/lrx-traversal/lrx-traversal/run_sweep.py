#!/usr/bin/env python3
"""
Sweep script: Fair comparison of LightGBM vs MLP value functions for Koltsov3.

A value function predicts random-walk distance from any permutation to the
identity element on the Koltsov3 Cayley graph. This is the Diffusion Distance
(DD) approach from the CayleyPy-RL paper: train a model to guess how many
random-walk steps it takes to reach a state from the identity.

This script sweeps over permutation lengths (n) and hyperparameters for both
LightGBM (engineered features) and MLP (one-hot encoding), collecting
supervised regression metrics (R², RMSE, Spearman ρ).

Results are saved incrementally to sweep_results.csv — the script is resumable.
Progress is logged to wandb (project: koltsov3-sweep).

Usage:
    python run_sweep.py                  # full sweep (48 LGB + 6 MLP configs × 8 n values)
    python run_sweep.py --quick          # quick validation (1 config each, n=16,32)
    python run_sweep.py --n 16 32 48     # specific n values
    python run_sweep.py --model lgb      # LightGBM only
    python run_sweep.py --model mlp      # MLP only
    python run_sweep.py --save-models    # also save trained LightGBM models to disk

Environment:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # recommended for GPU
"""

# =============================================================================
# IMPORTS
# =============================================================================
import random, time, os, sys, argparse, itertools, json, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # headless — no display needed
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
import wandb

warnings.filterwarnings('ignore')

# =============================================================================
# CLI ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Sweep LightGBM vs MLP for Koltsov3')
parser.add_argument('--n', type=int, nargs='*', default=None,
                    help='Specific n values (default: sweep all)')
parser.add_argument('--quick', action='store_true',
                    help='Quick validation mode (reduced grid)')
parser.add_argument('--model', choices=['lgb', 'mlp', 'both'], default='both',
                    help='Which model to sweep')
parser.add_argument('--n-walks', type=int, default=25_000,
                    help='Random walks for training data')
parser.add_argument('--max-train', type=int, default=500_000,
                    help='Max training samples (subsample if more)')
parser.add_argument('--n-test-walks', type=int, default=500,
                    help='Random walks for test data')
parser.add_argument('--save-models', action='store_true',
                    help='Save LightGBM models to disk (off by default)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
args = parser.parse_args()

# =============================================================================
# SEEDING & DEVICE DETECTION
# =============================================================================
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use GPU if available (MLP training), keep LightGBM on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} ({props.total_memory/1024**3:.1f} GB)')
print(f'Device: {device}')

# =============================================================================
# FIXED CONFIGURATION
# =============================================================================
K_VAL = 0                      # Koltsov3 S-generator parameter: S = (k, k+2)
VALIDATION_FRAC = 0.15         # fraction of training data used for validation
EARLY_STOPPING_ROUNDS = 100    # LightGBM early stopping patience
MLP_BATCH_SIZE = 1024          # batch size for MLP training
RESULTS_CSV = 'sweep_results.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# SWEEP GRID DEFINITION
# =============================================================================
# Full sweep: 48 LightGBM configs × 8 n values = 384 runs (12h)
#              6 MLP configs × 8 n values = 48 runs (2h)
# Total: 432 runs ≈ 14 hours

if args.quick:
    # Quick validation: minimal grid, fast feedback
    N_VALUES = [16, 32]
    LGB_GRID = {
        'num_leaves': [63],
        'max_depth': [8],
        'learning_rate': [0.1],
    }
    MLP_GRID = {
        'hidden_dim': [256],
        'epochs': [15],
        'learning_rate': [0.001],
    }
else:
    # Full sweep: comprehensive hyperparameter exploration
    # n capped at 64 — beyond this beam search is infeasible (64! ≈ 10^89 states)
    N_VALUES = [8, 16, 24, 32, 40, 48, 56, 64]
    LGB_GRID = {
        'num_leaves': [31, 63, 127, 255],      # tree complexity
        'max_depth': [6, 8, 10, 12],            # depth constraint
        'learning_rate': [0.05, 0.1, 0.2],      # shrinkage
    }
    MLP_GRID = {
        'hidden_dim': [128, 256, 512],           # hidden layer width
        'epochs': [15, 25],                      # training epochs (50 dropped: too slow)
        'learning_rate': [0.001],                # best performer from n=16 test
    }

if args.n is not None:
    N_VALUES = args.n

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
total_lgb_runs = (len(list(itertools.product(*LGB_GRID.values()))) * len(N_VALUES)
                  if args.model in ('lgb', 'both') else 0)
total_mlp_runs = (len(list(itertools.product(*MLP_GRID.values()))) * len(N_VALUES)
                  if args.model in ('mlp', 'both') else 0)
total_runs = total_lgb_runs + total_mlp_runs

print(f'{"="*60}')
print(f'SWEEP CONFIGURATION')
print(f'{"="*60}')
print(f'n values: {N_VALUES}')
print(f'Model: {args.model}')
print(f'Quick mode: {args.quick}')
print(f'LGB configs: {total_lgb_runs // max(len(N_VALUES),1)} / {total_lgb_runs} total')
print(f'MLP configs: {total_mlp_runs // max(len(N_VALUES),1)} / {total_mlp_runs} total')
print(f'Saving models: {args.save_models}')

# =============================================================================
# WANDB MONITORING
# =============================================================================
# Logs metrics from each run so you can track progress at wandb.ai
# Project: koltsov3-sweep
wandb.init(
    project='koltsov3-sweep',
    name=f'sweep_{time.strftime("%Y%m%d_%H%M%S")}',
    config={
        'n_values': N_VALUES,
        'lgb_grid': LGB_GRID,
        'mlp_grid': MLP_GRID,
        'total_runs': total_runs,
        'quick': args.quick,
        'max_train': args.max_train,
        'n_walks': args.n_walks,
        'seed': SEED,
    },
    tags=['sweep', 'quick' if args.quick else 'full'],
)

# =============================================================================
# KOLTSOV3 GENERATORS
# =============================================================================
# Three generators for the symmetric group S_n:
#   I = (0 1)(2 3)(4 5)...    swaps adjacent even-odd pairs
#   K = (1 2)(3 4)(5 6)...    swaps adjacent odd-even pairs
#   S = (k, k+2)              swaps position k with k+2
# I and K preserve position parity; only S breaks it.
# All moves shift elements by at most 2 positions.

def get_koltsov3_moves(n, k=0):
    """Return Koltsov3 generators as list of numpy arrays (each length n)."""
    I = np.arange(n); K = np.arange(n); S = np.arange(n)
    for i in range(0, n - 1, 2):  I[i], I[i + 1] = I[i + 1], I[i]
    for i in range(1, n - 1, 2):  K[i], K[i + 1] = K[i + 1], K[i]
    S[k], S[k + 2] = S[k + 2], S[k]
    return [I, K, S]

# =============================================================================
# RANDOM WALK GENERATION (shared by both models)
# =============================================================================
# Simple uniform random walks from the identity permutation.
# Every step of every walk becomes a training sample.
# This is the Diffusion Distance (DD) approach from the paper:
# the model learns to predict random-walk distance, not true graph distance.
# No first-visit deduplication — states may appear at multiple steps.

def generate_random_walks(n, n_walks, walk_length, k=0):
    """
    Generate simple random walks on Koltsov3 graph.

    Args:
        n: permutation length
        n_walks: number of independent walks
        walk_length: steps per walk
        k: S-generator parameter

    Returns:
        states: (n_walks * walk_length, n) int32 array of permutations
        labels: (n_walks * walk_length,) float32 array of step numbers
    """
    moves_list = get_koltsov3_moves(n, k)
    moves_t = torch.tensor(np.array(moves_list), dtype=torch.long)
    state_dest = torch.arange(n, dtype=torch.int64)
    n_gen = moves_t.shape[0]

    # All walks start at identity
    states = state_dest.unsqueeze(0).repeat(n_walks, 1).to(torch.uint8)
    all_s, all_l = [], []

    for step in range(1, walk_length + 1):
        # Uniform random generator choice — fully vectorized, no Python loop
        move_ids = torch.randint(0, n_gen, (n_walks,))
        states = torch.gather(states, 1, moves_t[move_ids])
        all_s.append(states.clone())
        all_l.append(torch.full((n_walks,), step, dtype=torch.float32))

    # Stack all walk steps into a single training set
    all_states = torch.cat(all_s, dim=0).numpy().astype(np.int32)
    all_labels = torch.cat(all_l, dim=0).numpy().astype(np.float32)
    return all_states, all_labels

# =============================================================================
# FEATURE EXTRACTION (LightGBM only — MLP uses raw one-hot encoding)
# =============================================================================
# These features encode Koltsov3-specific structure:
# - Displacement statistics (how far each element is from home)
# - Parity mismatch (I,K preserve parity, only S breaks it)
# - Adjacent pair sorting (generators swap adjacent positions)
# - Inversions/descents (global permutation structure)
# - I/K/S generator-specific features (which pairs are sorted/correct)
# - Theoretical lower bounds on distance
# - Inverse permutation positions (where each value sits)
# Total: ~3n + 20 features

def extract_features(states_array, n, k=0):
    """
    Extract generator-aware features for Koltsov3 from a batch of permutations.

    Args:
        states_array: (N, n) int32 array — each row is a permutation p[i]=value at position i
        n: permutation length
        k: S-generator parameter

    Returns:
        pandas DataFrame with ~3n+20 feature columns
    """
    N = states_array.shape[0]
    f = {}
    ident = np.arange(n)

    # ---- 1. Positional displacement ----
    # |p_i - i|: how far element at position i is from its home position
    disp = np.abs(states_array - ident)
    for i in range(n): f[f'd_{i}'] = disp[:, i]
    f['d_sum'] = disp.sum(1)           # total Manhattan distance
    f['d_max'] = disp.max(1)           # worst-case displacement
    f['d_mean'] = disp.mean(1)         # average displacement
    f['d_std'] = disp.std(1)           # spread of displacement
    f['d_med'] = np.median(disp, 1)    # median displacement
    f['d_eq0'] = (disp == 0).sum(1)    # count of elements already at home
    f['d_eq1'] = (disp == 1).sum(1)    # count of elements 1 step from home
    f['d_ge2'] = (disp >= 2).sum(1)    # count needing at least 1 move
    f['d_ge4'] = (disp >= 4).sum(1)    # count needing at least 2 moves

    # ---- 2. Parity mismatch ----
    # I and K swaps preserve position parity (even↔even, odd↔odd).
    # Only S can move an element to the other parity class.
    # So parity mismatches directly constrain the minimum number of S-moves.
    pm = ((states_array & 1) != (ident & 1)).astype(np.float32)
    for i in range(n): f[f'pm_{i}'] = pm[:, i]
    f['pm_sum'] = pm.sum(1)            # how many elements on wrong parity
    f['pm_pct'] = pm.mean(1)           # fraction on wrong parity
    f['pm_x_disp'] = (pm * disp).sum(1) # parity mismatch weighted by displacement

    # ---- 3. Adjacent pair structure ----
    # All generators swap adjacent positions, so adjacent ordering is key.
    sorted_adj = (states_array[:, :-1] < states_array[:, 1:]).astype(np.float32)
    for i in range(n - 1): f[f's_{i}{i+1}'] = sorted_adj[:, i]
    f['s_sum'] = sorted_adj.sum(1)     # how many adjacent pairs are sorted

    adj_diff = np.abs(np.diff(states_array, axis=1))
    for i in range(n - 1): f[f'ad_{i}'] = adj_diff[:, i]
    f['ad_sum'] = adj_diff.sum(1)      # total adjacent difference
    f['ad_max'] = adj_diff.max(1)      # largest adjacent difference

    # Count pairs where both elements are at their correct positions
    pair_correct = ((states_array[:, :-1] == ident[:-1]) &
                    (states_array[:, 1:] == ident[1:])).astype(np.float32)
    f['pair_correct'] = pair_correct.sum(1)

    # ---- 4. Inversions ----
    # p_i > p_j for i < j. Measures global disorder of the permutation.
    invs = np.zeros(N, dtype=np.float32)
    for i in range(n):
        invs += (states_array[:, i:i+1] > states_array[:, i+1:]).sum(axis=1)
    f['inv'] = invs
    f['inv_frac'] = invs / (n * (n - 1) / 2)     # normalized to [0,1]
    f['inv_parity'] = (invs.astype(np.int64) & 1).astype(np.float32)  # 0=even, 1=odd

    # ---- 5. Descents ----
    # p_i > p_{i+1}. Each descent is an adjacent pair that needs fixing.
    descs = np.zeros(N, dtype=np.float32)
    for i in range(n - 1):
        descs += (states_array[:, i] > states_array[:, i + 1]).astype(np.float32)
    f['desc'] = descs

    # ---- 6. I-generator specific ----
    # I = (0,1)(2,3)... swaps even-odd adjacent pairs.
    ieu = np.zeros(N, dtype=np.float32)  # number of I-pairs that are unsorted
    iec = np.zeros(N, dtype=np.float32)  # number of I-pairs that are identity
    for idx in range(0, n - 1, 2):
        ieu += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        iec += ((states_array[:, idx] == idx) & (states_array[:, idx + 1] == idx + 1)).astype(np.float32)
    f['I_unsorted'] = ieu
    f['I_correct'] = iec

    # ---- 7. K-generator specific ----
    # K = (1,2)(3,4)... swaps odd-even adjacent pairs.
    keu = np.zeros(N, dtype=np.float32)
    kec = np.zeros(N, dtype=np.float32)
    for idx in range(1, n - 1, 2):
        keu += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        kec += ((states_array[:, idx] == idx) & (states_array[:, idx + 1] == idx + 1)).astype(np.float32)
    f['K_unsorted'] = keu
    f['K_correct'] = kec

    # ---- 8. S-generator specific ----
    # S = (k, k+2). Only applied to the specific pair.
    if k + 2 < n:
        f['S_sorted'] = (states_array[:, k] < states_array[:, k + 2]).astype(np.float32)
        f['S_diff'] = np.abs(states_array[:, k].astype(np.float32)
                             - states_array[:, k + 2].astype(np.float32))
        f['S_disp'] = (np.abs(states_array[:, k] - k)
                       + np.abs(states_array[:, k + 2] - (k + 2)))
        f['S_need_swap'] = ((states_array[:, k] == k + 2)
                            & (states_array[:, k + 2] == k)).astype(np.float32)

    # ---- 9. Theoretical lower bounds on distance ----
    # Each generator moves elements by at most 2 positions.
    # Each S-move can fix at most 2 parity mismatches.
    f['lb_disp'] = disp.max(1) / 2.0       # based on worst displacement
    f['lb_parity'] = pm.sum(1) / 2.0        # based on parity errors
    f['lb_comb'] = np.maximum(f['lb_disp'], f['lb_parity'])  # combined bound

    # ---- 10. Inverse permutation ----
    # For each value v, what position is it in? Captures complement to raw positions.
    inv_perm = np.argsort(states_array, axis=1)
    for v in range(n):
        f[f'pos_of_{v}'] = inv_perm[:, v].astype(np.float32)
    f['where_0'] = np.abs(inv_perm[:, 0] - 0).astype(np.float32)      # where is element 0?
    f['where_n1'] = np.abs(inv_perm[:, n - 1] - (n - 1)).astype(np.float32)  # where is element n-1?

    # ---- 11. Consecutive correct runs ----
    # Longest run of elements already at their correct positions.
    run_lens = np.zeros(N, dtype=np.float32)
    for i in range(N):
        cur = best = 0
        for x in (states_array[i] == ident):
            if x: cur += 1; best = max(best, cur)
            else: cur = 0
        run_lens[i] = best
    f['run_correct'] = run_lens

    # ---- 12. Displacement skew ----
    # How many elements shifted left vs right? Should balance (~0 net).
    sd = states_array.astype(np.float32) - ident.astype(np.float32)
    f['disp_pos'] = np.maximum(sd, 0).sum(1)   # total rightward shift
    f['disp_neg'] = np.maximum(-sd, 0).sum(1)  # total leftward shift
    f['disp_net'] = sd.sum(1)                  # net displacement (should be ~0)

    # ---- 13. Raw positions ----
    # Give the tree the raw material to discover its own splits.
    for i in range(n): f[f'p_{i}'] = states_array[:, i].astype(np.float32)

    return pd.DataFrame(f)

# =============================================================================
# METRICS HELPER
# =============================================================================
def evaluate(y_true, y_pred):
    """Compute R², RMSE, and Spearman rank correlation."""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'spearman': stats.spearmanr(y_true, y_pred).statistic,
    }

# =============================================================================
# RESUME SUPPORT — track already-completed runs
# =============================================================================
def load_completed():
    """Read sweep_results.csv and return set of (n, model, params_hash) tuples
    that have already been completed. Used for resume-after-interrupt."""
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        return set(zip(df['n'], df['model'], df['params_hash']))
    return set()

completed = load_completed()
print(f'Resuming: {len(completed)} completed runs found')

# =============================================================================
# TEST SET CACHE — generate once per n, reuse across all configs for fairness
# =============================================================================
TEST_CACHE = {}

def get_test_data(n, k=0):
    """
    Generate (or retrieve cached) test data for a given n.
    All configs for the same n share the same test set — fair comparison.
    """
    if n not in TEST_CACHE:
        walk_length = 8 * n
        # Scale down test walks for large n to keep inference fast
        n_test = min(args.n_test_walks, max(500, 2000 - (n - 16) * 10))
        states, labels = generate_random_walks(n, n_test, walk_length, k)
        TEST_CACHE[n] = {
            'states_raw': states,               # for MLP (one-hot)
            'labels': labels,                   # ground truth
            'features': extract_features(states, n, k),  # for LightGBM
        }
    return TEST_CACHE[n]

# =============================================================================
# MLP MODEL DEFINITION
# =============================================================================
# Architecture: one-hot encoding → Linear(n², hidden) → ReLU → Linear(hidden, 1)
# Matches the original koltsov3 notebook's Net class exactly.
# One-hot is done inside forward() so the model takes raw uint8 permutations.

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(input_size * num_classes, hidden_dim),   # n² → hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),                           # hidden → scalar distance
        )

    def forward(self, x):
        # x: (batch, n) uint8 → one-hot → (batch, n, n) float → flatten → (batch, n²)
        x = torch.nn.functional.one_hot(
            x.long(), num_classes=self.num_classes
        ).float().flatten(start_dim=-2)
        return self.layers(x)

# =============================================================================
# LIGHTGBM TRAINER
# =============================================================================
# Trains a gradient-boosted tree ensemble on engineered features.
# LightGBM cannot do online streaming like MLPs — it trains on a fixed dataset.
# Uses early stopping on a 15% validation split.
# The test set is the same pre-generated cached set shared across all configs.

def train_lightgbm(n, params, test_data):
    """
    Train a LightGBM regressor for permutation length n.

    Args:
        n: permutation length
        params: dict with keys num_leaves, max_depth, learning_rate
        test_data: cached dict with 'features' and 'labels' for evaluation

    Returns:
        dict of results (metrics, timing, etc.) to append to sweep_results.csv
    """
    k = K_VAL
    walk_length = 8 * n

    # ---- Generate training data ----
    states, labels = generate_random_walks(n, args.n_walks, walk_length, k)
    total = len(labels)
    if total > args.max_train:
        idx = np.random.choice(total, args.max_train, replace=False)
        states = states[idx]
        labels = labels[idx]

    # ---- Extract feature DataFrame ----
    df = extract_features(states, n, k)
    df['label'] = labels
    feature_cols = [c for c in df.columns if c != 'label']

    # ---- Train/validation split (15% val) ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        df[feature_cols], df['label'],
        test_size=VALIDATION_FRAC, random_state=SEED
    )

    # ---- Train model with early stopping ----
    t0 = time.time()
    lgb_train = lgb.Dataset(X_tr, y_tr, feature_name=list(feature_cols))
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train,
                          feature_name=list(feature_cols))
    evals_result = {}   # dict to capture training curves via callback

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'n_estimators': 1000,               # max trees (early stopping cuts this)
        'min_child_samples': 20,            # regularization: min samples per leaf
        'subsample': 0.8,                   # row bagging
        'colsample_bytree': 0.8,            # feature bagging
        'reg_alpha': 0.01,                  # L1 regularization
        'reg_lambda': 0.01,                 # L2 regularization
        'verbose': -1,                      # silent
        'seed': SEED,
        'n_jobs': -1,                       # use all CPU cores
    }

    model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.record_evaluation(evals_result),       # captures per-iteration RMSE
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),  # stops when val RMSE stalls
        ],
    )
    fit_time = time.time() - t0

    # ---- Evaluate on train/val ----
    y_tr_pred = model.predict(X_tr)
    y_va_pred = model.predict(X_val)
    y_tr_arr = np.asarray(y_tr).ravel()
    y_va_arr = np.asarray(y_val).ravel()
    m_tr = evaluate(y_tr_arr, y_tr_pred)
    m_va = evaluate(y_va_arr, y_va_pred)

    # ---- Evaluate on (cached) test set ----
    t_pred = time.time()
    y_te_pred = model.predict(test_data['features'])
    pred_time = time.time() - t_pred
    m_te = evaluate(test_data['labels'], y_te_pred)

    # ---- Optionally save model to disk ----
    if args.save_models:
        model_path = os.path.join(MODELS_DIR,
                                  f'lgb_n{n}_{params_to_hash(params)}.txt')
        model.save_model(model_path)
    else:
        model_path = ''

    # ---- Return results dict ----
    return {
        'n': n,
        'model': 'lgb',
        'params_hash': params_to_hash(params),
        'params': json.dumps(params),
        'train_r2': m_tr['r2'],
        'val_r2': m_va['r2'],
        'test_r2': m_te['r2'],
        'train_rmse': m_tr['rmse'],
        'val_rmse': m_va['rmse'],
        'test_rmse': m_te['rmse'],
        'train_spearman': m_tr['spearman'],
        'val_spearman': m_va['spearman'],
        'test_spearman': m_te['spearman'],
        'fit_time': fit_time,
        'predict_time': pred_time,
        'n_features': len(feature_cols),
        'n_train': len(X_tr),
        'n_val': len(X_val),
        'n_test': len(test_data['labels']),
        'best_iteration': model.best_iteration,
        'model_path': model_path,
    }

# =============================================================================
# MLP TRAINER
# =============================================================================
# Trains a one-hot MLP with fresh random walks generated EVERY epoch.
# This matches the original notebook's approach: each epoch sees new data,
# giving the MLP higher effective data diversity over the course of training.
# Training happens on GPU if available. LightGBM stays on CPU.

def train_mlp(n, params, test_data):
    """
    Train an MLP value function for permutation length n.

    Args:
        n: permutation length
        params: dict with keys hidden_dim, epochs, learning_rate
        test_data: cached dict with 'states_raw' and 'labels' for evaluation

    Returns:
        dict of results (metrics, timing, etc.) to append to sweep_results.csv
    """
    k = K_VAL
    walk_length = 8 * n
    n_gen = 3   # I, K, S

    # ---- Clear GPU memory from previous runs (prevents fragmentation OOM) ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Move generators and identity to device ----
    moves_list = get_koltsov3_moves(n, k)
    moves_t = torch.tensor(np.array(moves_list), dtype=torch.long, device=device)
    state_dest = torch.arange(n, dtype=torch.int64, device=device)

    # ---- Scale walks per epoch with n to avoid GPU OOM ----
    # At n=64, 20K walks × 512 steps = 10M samples ≈ 640 MB epoch data.
    # Halving to 10K keeps peak VRAM under 1 GB for all n ≤ 64.
    n_walks_per_epoch = 20_000 if n <= 40 else 10_000
    n_epochs = params['epochs']

    # ---- Build model ----
    model = MLP(n, params['hidden_dim'], n).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # ---- Load test set onto GPU (uint8, small: ~16 MB for n=64) ----
    X_te = torch.tensor(test_data['states_raw'], dtype=torch.uint8, device=device)
    y_te = torch.tensor(test_data['labels'], dtype=torch.float32, device=device)

    mlp_train_losses = []
    mlp_val_losses = []

    t0 = time.time()
    for epoch in range(n_epochs):
        # ---- Generate fresh random walks for this epoch (on GPU) ----
        # This is the key difference from LightGBM: MLP sees new data each epoch.
        # The total data diversity over n_epochs is n_epochs × n_walks_per_epoch
        # × walk_length distinct random walk steps.
        states_t = state_dest.unsqueeze(0).repeat(n_walks_per_epoch, 1).to(torch.uint8)
        epoch_states, epoch_labels = [], []

        for step in range(1, walk_length + 1):
            move_ids = torch.randint(0, n_gen, (n_walks_per_epoch,), device=device)
            states_t = torch.gather(states_t, 1, moves_t[move_ids])
            epoch_states.append(states_t.clone())
            epoch_labels.append(torch.full((n_walks_per_epoch,), step,
                                           dtype=torch.float32, device=device))

        # Concatenate all walk steps into one training tensor
        X_tr = torch.cat(epoch_states, dim=0)    # (n_walks * walk_length, n)
        y_tr = torch.cat(epoch_labels, dim=0)    # (n_walks * walk_length,)

        # Shuffle samples
        perm = torch.randperm(len(X_tr), device=device)
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        # ---- Training loop (minibatch SGD) ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_tr), MLP_BATCH_SIZE):
            bx = X_tr[i:i + MLP_BATCH_SIZE]
            by = y_tr[i:i + MLP_BATCH_SIZE]
            loss = criterion(model(bx).squeeze(), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        mlp_train_losses.append(train_loss / n_batches)

        # ---- CRITICAL: Free epoch data to prevent GPU memory accumulation ----
        # Without this, the epoch_states list (hundreds of MB) and X_tr tensor
        # accumulate across epochs and cause OOM on 4 GB GPUs.
        n_train_this_epoch = len(X_tr)   # capture before deletion
        del X_tr, y_tr, epoch_states, epoch_labels
        gc.collect()                     # force Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()     # release CUDA memory blocks

        # ---- Validation on test set subset (10K samples for speed) ----
        model.eval()
        with torch.no_grad():
            vi = np.random.choice(len(X_te), min(10000, len(X_te)), replace=False)
            val_loss = criterion(model(X_te[vi]).squeeze(), y_te[vi]).item()
        mlp_val_losses.append(val_loss)

        print(f'    Epoch {epoch+1}/{n_epochs}: '
              f'train={mlp_train_losses[-1]:.1f}, val={val_loss:.1f}', flush=True)

    fit_time = time.time() - t0

    # ---- Evaluate on test set (BATCHED to avoid one-hot OOM) ----
    # One-hot on the full test set would need (test_samples × n² × 4) bytes.
    # For n=64, that's 256K × 4096 × 4 = 4.2 GB — exceeds GPU memory.
    # Processing in batch_size chunks keeps per-batch one-hot at ~16 MB.
    model.eval()
    t_pred = time.time()
    y_te_pred_list = []
    with torch.no_grad():
        for i in range(0, len(X_te), MLP_BATCH_SIZE):
            y_te_pred_list.append(
                model(X_te[i:i + MLP_BATCH_SIZE]).squeeze().cpu().numpy())
    y_te_pred = np.concatenate(y_te_pred_list)
    pred_time = time.time() - t_pred
    m_te = evaluate(test_data['labels'], y_te_pred)

    # ---- Evaluate on a fresh train sample for train metrics (batched) ----
    states, labels = generate_random_walks(n, 500, walk_length, k)
    n_tr_eval = min(10000, len(states))
    states_t = torch.tensor(states[:n_tr_eval], dtype=torch.uint8, device=device)
    labels_t = torch.tensor(labels[:n_tr_eval], dtype=torch.float32, device=device)
    model.eval()
    y_tr_pred_list = []
    with torch.no_grad():
        for i in range(0, len(states_t), MLP_BATCH_SIZE):
            y_tr_pred_list.append(
                model(states_t[i:i + MLP_BATCH_SIZE]).squeeze().cpu().numpy())
    y_tr_pred = np.concatenate(y_tr_pred_list)
    m_tr = evaluate(labels[:n_tr_eval], y_tr_pred)

    # ---- Return results dict ----
    return {
        'n': n,
        'model': 'mlp',
        'params_hash': params_to_hash(params),
        'params': json.dumps(params),
        'train_r2': m_tr['r2'],
        'val_r2': m_te['r2'],              # MLP val IS the test set (no separate val split)
        'test_r2': m_te['r2'],
        'train_rmse': m_tr['rmse'],
        'val_rmse': m_te['rmse'],
        'test_rmse': m_te['rmse'],
        'train_spearman': m_tr['spearman'],
        'val_spearman': m_te['spearman'],
        'test_spearman': m_te['spearman'],
        'fit_time': fit_time,
        'predict_time': pred_time,
        'n_features': n * n,               # one-hot encoding: n² features
        'n_train': n_train_this_epoch,
        'n_val': len(y_te),
        'n_test': len(y_te),
        'best_iteration': n_epochs,
        'model_path': '',
    }

# =============================================================================
# UTILITIES
# =============================================================================
def params_to_hash(params):
    """Convert param dict to a short string identifier for CSV/resume tracking."""
    return '_'.join(f'{k}={v}' for k, v in sorted(params.items()))

def save_result(result):
    """Append a single result row to sweep_results.csv (creates file if needed)."""
    df = pd.DataFrame([result])
    if os.path.exists(RESULTS_CSV):
        df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_CSV, index=False)

def is_completed(n, model_type, params):
    """Check if this (n, model, params) combination was already run."""
    return (n, model_type, params_to_hash(params)) in completed

# =============================================================================
# SWEEP ORCHESTRATOR
# =============================================================================
# Iterates over all n values and hyperparameter combinations.
# For each (n, model, config): trains, evaluates, saves result to CSV, logs to wandb.
# Supports resume: already-completed runs are skipped.

print(f'\n{"="*60}')
print(f'STARTING SWEEP')
print(f'{"="*60}\n')

run_count = len(completed)     # how many runs are already done (for resume)
run_start_time = time.time()   # for ETA computation

def log_to_wandb(result, model_type):
    """Log metrics from a completed run to wandb for real-time monitoring."""
    wandb.log({
        f'{model_type}/test_r2': result['test_r2'],
        f'{model_type}/test_rmse': result['test_rmse'],
        f'{model_type}/test_spearman': result['test_spearman'],
        f'{model_type}/val_r2': result['val_r2'],
        f'{model_type}/fit_time': result['fit_time'],
        f'{model_type}/n_features': result['n_features'],
        f'{model_type}/best_iteration': result['best_iteration'],
        'n': result['n'],
        'progress': run_count / total_runs,
        'eta_hours': (time.time() - run_start_time) / max(run_count, 1)
                     * (total_runs - run_count) / 3600,
    })

for n in N_VALUES:
    if n < 3:
        print(f'Skipping n={n} (too small for Koltsov3)')
        continue

    print(f'\n{"#"*60}')
    print(f'# n = {n}')
    print(f'{"#"*60}')

    # ---- Generate (or retrieve cached) test data for this n ----
    print(f'  Generating test data...')
    t0 = time.time()
    test_data = get_test_data(n, K_VAL)
    print(f'  Test data: {len(test_data["labels"]):,} samples '
          f'({time.time()-t0:.1f}s)')

    # ============================
    # LightGBM Sweep
    # ============================
    if args.model in ('lgb', 'both'):
        lgb_configs = list(itertools.product(*LGB_GRID.values()))
        for ci, combo in enumerate(lgb_configs):
            params = dict(zip(LGB_GRID.keys(), combo))

            # Skip if already completed (resume support)
            if is_completed(n, 'lgb', params):
                print(f'  [lgb {ci+1}/{len(lgb_configs)}] {params} '
                      f'— SKIPPED (already done)')
                continue

            print(f'  [lgb {ci+1}/{len(lgb_configs)}] {params}', flush=True)
            try:
                result = train_lightgbm(n, params, test_data)
                save_result(result)
                run_count += 1
                log_to_wandb(result, 'lgb')
                eta_h = ((time.time() - run_start_time) / run_count
                         * (total_runs - run_count) / 3600)
                print(f"    Test R²={result['test_r2']:.4f}  "
                      f"RMSE={result['test_rmse']:.2f}  "
                      f"ρ={result['test_spearman']:.4f}  "
                      f"time={result['fit_time']:.1f}s  "
                      f"[{run_count}/{total_runs}] ETA {eta_h:.1f}h")
            except Exception as e:
                print(f'    FAILED: {e}')

    # ============================
    # MLP Sweep
    # ============================
    if args.model in ('mlp', 'both'):
        mlp_configs = list(itertools.product(*MLP_GRID.values()))
        for ci, combo in enumerate(mlp_configs):
            params = dict(zip(MLP_GRID.keys(), combo))

            # Skip if already completed (resume support)
            if is_completed(n, 'mlp', params):
                print(f'  [mlp {ci+1}/{len(mlp_configs)}] {params} '
                      f'— SKIPPED (already done)')
                continue

            print(f'  [mlp {ci+1}/{len(mlp_configs)}] {params}', flush=True)
            try:
                result = train_mlp(n, params, test_data)
                save_result(result)
                run_count += 1
                log_to_wandb(result, 'mlp')
                eta_h = ((time.time() - run_start_time) / run_count
                         * (total_runs - run_count) / 3600)
                print(f"    Test R²={result['test_r2']:.4f}  "
                      f"RMSE={result['test_rmse']:.2f}  "
                      f"ρ={result['test_spearman']:.4f}  "
                      f"time={result['fit_time']:.1f}s  "
                      f"[{run_count}/{total_runs}] ETA {eta_h:.1f}h")
            except Exception as e:
                print(f'    FAILED: {e}')

# =============================================================================
# SWEEP COMPLETE — final summary
# =============================================================================
print(f'\n{"="*60}')
print(f'SWEEP COMPLETE')
print(f'{"="*60}')
print(f'Results saved to: {RESULTS_CSV}')

wandb.finish()

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    print(f'\nTotal runs: {len(df)}')
    print(f'LGB runs:   {len(df[df.model == "lgb"])}')
    print(f'MLP runs:   {len(df[df.model == "mlp"])}')
    print(f'n values:   {sorted(df.n.unique())}')

    # Show best configuration per (n, model) by test R²
    best = df.loc[df.groupby(['n', 'model'])['test_r2'].idxmax()]
    print(f'\n--- Best per (n, model) by R² ---')
    for (n_val, model), row in best.set_index(['n', 'model']).iterrows():
        print(f'  n={int(n_val):3d}  {model:3s}: '
              f'R²={row["test_r2"]:.4f}  '
              f'RMSE={row["test_rmse"]:.1f}  '
              f'ρ={row["test_spearman"]:.4f}  '
              f'params={row["params"]}')
