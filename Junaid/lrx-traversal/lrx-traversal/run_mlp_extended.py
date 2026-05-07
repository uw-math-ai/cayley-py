#!/usr/bin/env python3
"""
Extended MLP test: push the best MLP config further.

The full sweep showed MLP (hidden_dim=512, epochs=25, Adam, lr=0.001)
slightly beating LightGBM at all n values. This script tests whether
more epochs, AdamW, or different learning rates close the gap further.

Usage:
    python run_mlp_extended.py          # test epochs 50,100 + AdamW
    python run_mlp_extended.py --quick  # test n=16,64 only
"""

import random, time, os, sys, argparse, itertools, json, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy import stats
import wandb

warnings.filterwarnings('ignore')

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} ({props.total_memory/1024**3:.1f} GB)')
print(f'Device: {device}')

K_VAL = 0; MLP_BATCH_SIZE = 1024; RESULTS_CSV = 'mlp_extended_results.csv'

# Test grid: epochs × optimizer × lr
N_VALUES = [16, 32, 48, 64] if args.quick else [16, 32, 48, 64]
EPOCHS_VALS = [25, 50, 100]
OPTIMIZER_CONFIGS = [
    {'name': 'adam',   'lr': 0.001,  'weight_decay': 0.0},
    {'name': 'adam',   'lr': 0.0005, 'weight_decay': 0.0},
    {'name': 'adamw',  'lr': 0.001,  'weight_decay': 0.01},
    {'name': 'adamw',  'lr': 0.0005, 'weight_decay': 0.01},
]

total_runs = len(N_VALUES) * len(EPOCHS_VALS) * len(OPTIMIZER_CONFIGS)
print(f'Configs: {len(N_VALUES)} n × {len(EPOCHS_VALS)} epochs × {len(OPTIMIZER_CONFIGS)} optim = {total_runs} runs')

# =============================================================================
# Koltsov3 generators + random walks (same as main sweep)
# =============================================================================
def get_koltsov3_moves(n, k=0):
    I = np.arange(n); K = np.arange(n); S = np.arange(n)
    for i in range(0, n-1, 2): I[i], I[i+1] = I[i+1], I[i]
    for i in range(1, n-1, 2): K[i], K[i+1] = K[i+1], K[i]
    S[k], S[k+2] = S[k+2], S[k]
    return [I, K, S]

def generate_random_walks(n, n_walks, walk_length, k=0):
    moves_list = get_koltsov3_moves(n, k)
    moves_t = torch.tensor(np.array(moves_list), dtype=torch.long)
    state_dest = torch.arange(n, dtype=torch.int64)
    n_gen = moves_t.shape[0]
    states = state_dest.unsqueeze(0).repeat(n_walks, 1).to(torch.uint8)
    all_s, all_l = [], []
    for step in range(1, walk_length + 1):
        move_ids = torch.randint(0, n_gen, (n_walks,))
        states = torch.gather(states, 1, moves_t[move_ids])
        all_s.append(states.clone())
        all_l.append(torch.full((n_walks,), step, dtype=torch.float32))
    all_states = torch.cat(all_s, dim=0).numpy().astype(np.int32)
    all_labels = torch.cat(all_l, dim=0).numpy().astype(np.float32)
    return all_states, all_labels

def evaluate(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'spearman': stats.spearmanr(y_true, y_pred).statistic,
    }

# =============================================================================
# MLP model
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(input_size * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes).float().flatten(start_dim=-2)
        return self.layers(x)

# =============================================================================
# Resume support
# =============================================================================
def load_completed():
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        return set(zip(df['n'], df['optimizer'], df['epochs'], df['lr']))
    return set()

completed = load_completed()
print(f'Resuming: {len(completed)} completed runs')

# =============================================================================
# Test set cache
# =============================================================================
TEST_CACHE = {}
def get_test_data(n):
    if n not in TEST_CACHE:
        walk_length = 8 * n
        states, labels = generate_random_walks(n, 500, walk_length, K_VAL)
        TEST_CACHE[n] = {'states_raw': states, 'labels': labels}
    return TEST_CACHE[n]

# =============================================================================
# Train MLP with config
# =============================================================================
def train_mlp_extended(n, epochs, opt_config):
    k = K_VAL; walk_length = 8 * n; n_gen = 3

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    moves_list = get_koltsov3_moves(n, k)
    moves_t = torch.tensor(np.array(moves_list), dtype=torch.long, device=device)
    state_dest = torch.arange(n, dtype=torch.int64, device=device)

    n_walks_per_epoch = 20_000 if n <= 40 else 10_000
    hidden_dim = 512  # best from sweep

    model = MLP(n, hidden_dim, n).to(device)
    criterion = nn.MSELoss()

    # Optimizer selection
    if opt_config['name'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt_config['lr'],
                                weight_decay=opt_config['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt_config['lr'])

    test_data = get_test_data(n)
    X_te = torch.tensor(test_data['states_raw'], dtype=torch.uint8, device=device)
    y_te = torch.tensor(test_data['labels'], dtype=torch.float32, device=device)

    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(epochs):
        # Fresh random walks
        states_t = state_dest.unsqueeze(0).repeat(n_walks_per_epoch, 1).to(torch.uint8)
        epoch_states, epoch_labels = [], []
        for step in range(1, walk_length + 1):
            move_ids = torch.randint(0, n_gen, (n_walks_per_epoch,), device=device)
            states_t = torch.gather(states_t, 1, moves_t[move_ids])
            epoch_states.append(states_t.clone())
            epoch_labels.append(torch.full((n_walks_per_epoch,), step,
                                           dtype=torch.float32, device=device))

        X_tr = torch.cat(epoch_states, dim=0)
        y_tr = torch.cat(epoch_labels, dim=0)
        perm = torch.randperm(len(X_tr), device=device)
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        model.train()
        train_loss = 0.0; n_batches = 0
        for i in range(0, len(X_tr), MLP_BATCH_SIZE):
            bx = X_tr[i:i+MLP_BATCH_SIZE]; by = y_tr[i:i+MLP_BATCH_SIZE]
            loss = criterion(model(bx).squeeze(), by)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item(); n_batches += 1
        train_losses.append(train_loss / n_batches)

        n_train_this_epoch = len(X_tr)
        del X_tr, y_tr, epoch_states, epoch_labels
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            vi = np.random.choice(len(X_te), min(10000, len(X_te)), replace=False)
            val_losses.append(criterion(model(X_te[vi]).squeeze(), y_te[vi]).item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'    Ep {epoch+1}/{epochs}: '
                  f'train={train_losses[-1]:.1f}, val={val_losses[-1]:.1f}', flush=True)

    fit_time = time.time() - t0

    # Batched test evaluation
    model.eval(); t_pred = time.time()
    y_te_pred_list = []
    with torch.no_grad():
        for i in range(0, len(X_te), MLP_BATCH_SIZE):
            y_te_pred_list.append(model(X_te[i:i+MLP_BATCH_SIZE]).squeeze().cpu().numpy())
    y_te_pred = np.concatenate(y_te_pred_list)
    pred_time = time.time() - t_pred
    m_te = evaluate(test_data['labels'], y_te_pred)

    # Train metrics
    states, labels = generate_random_walks(n, 500, walk_length, k)
    n_tr_eval = min(10000, len(states))
    states_t = torch.tensor(states[:n_tr_eval], dtype=torch.uint8, device=device)
    model.eval()
    with torch.no_grad():
        y_tr_pred_list = []
        for i in range(0, len(states_t), MLP_BATCH_SIZE):
            y_tr_pred_list.append(model(states_t[i:i+MLP_BATCH_SIZE]).squeeze().cpu().numpy())
    y_tr_pred = np.concatenate(y_tr_pred_list)
    m_tr = evaluate(labels[:n_tr_eval], y_tr_pred)

    return {
        'n': n, 'epochs': epochs,
        'optimizer': opt_config['name'], 'lr': opt_config['lr'],
        'weight_decay': opt_config['weight_decay'],
        'hidden_dim': hidden_dim,
        'test_r2': m_te['r2'], 'test_rmse': m_te['rmse'],
        'test_spearman': m_te['spearman'],
        'train_r2': m_tr['r2'], 'train_rmse': m_tr['rmse'],
        'fit_time': fit_time, 'predict_time': pred_time,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
    }

# =============================================================================
# Wandb
# =============================================================================
wandb.init(
    project='koltsov3-sweep',
    name=f'mlp_extended_{time.strftime("%Y%m%d_%H%M%S")}',
    config={'n_values': N_VALUES, 'epochs': EPOCHS_VALS, 'optimizers': OPTIMIZER_CONFIGS},
    tags=['mlp_extended'],
)

# =============================================================================
# Run sweep
# =============================================================================
run_count = len(completed); run_start = time.time()

for n in N_VALUES:
    print(f'\n{"#"*60}\n# n = {n}\n{"#"*60}')
    test_data = get_test_data(n)
    print(f'  Test data: {len(test_data["labels"]):,} samples')

    for epochs in EPOCHS_VALS:
        for opt_config in OPTIMIZER_CONFIGS:
            key = (n, opt_config['name'], epochs, opt_config['lr'])
            if key in completed:
                print(f'  [n={n}] optimizer={opt_config["name"]} lr={opt_config["lr"]} '
                      f'epochs={epochs} — SKIPPED')
                continue

            print(f'  [n={n}] optimizer={opt_config["name"]} lr={opt_config["lr"]} '
                  f'epochs={epochs}', flush=True)
            try:
                result = train_mlp_extended(n, epochs, opt_config)
                df = pd.DataFrame([result])
                if os.path.exists(RESULTS_CSV):
                    df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
                else:
                    df.to_csv(RESULTS_CSV, index=False)
                run_count += 1
                eta_h = ((time.time() - run_start) / run_count * (total_runs - run_count) / 3600)
                print(f"    R²={result['test_r2']:.4f}  RMSE={result['test_rmse']:.2f}  "
                      f"ρ={result['test_spearman']:.4f}  time={result['fit_time']:.1f}s  "
                      f"[{run_count}/{total_runs}] ETA {eta_h:.1f}h")
            except Exception as e:
                print(f'    FAILED: {e}')

print(f'\nDONE — {run_count} runs')
wandb.finish()

# Summary
if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    print(f'\n--- Best per (n, epochs, optimizer) by R² ---')
    best = df.loc[df.groupby(['n'])['test_r2'].idxmax()]
    for _, row in best.iterrows():
        print(f'  n={int(row["n"]):2d}: R²={row["test_r2"]:.4f}  ρ={row["test_spearman"]:.4f}  '
              f'opt={row["optimizer"]}  lr={row["lr"]}  epochs={int(row["epochs"])}  '
              f'time={row["fit_time"]:.0f}s')
