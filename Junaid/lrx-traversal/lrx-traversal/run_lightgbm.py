#!/usr/bin/env python3
"""LightGBM Value Function for Koltsov3 Cayley Graph.

Usage:
    python run_lightgbm.py            # default n=16
    python run_lightgbm.py 24         # n=24
    python run_lightgbm.py 32         # n=32
"""

import random, time, os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend if running headless, but don't override if already set (e.g. by Jupyter)
if 'inline' not in matplotlib.get_backend():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats

# ========================
# Parse arguments
# ========================
parser = argparse.ArgumentParser(description='LightGBM vs MLP for Koltsov3 value function')
parser.add_argument('n', type=int, nargs='?', default=16, help='Permutation length (default: 16)')
parser.add_argument('--walks', type=int, default=25_000, help='Number of random walks')
parser.add_argument('--max-train', type=int, default=500_000, help='Max training samples')
parser.add_argument('--mlp-epochs', type=int, default=15, help='MLP training epochs')
parser.add_argument('--lgb-trees', type=int, default=1000, help='LightGBM num boost rounds')
args = parser.parse_args()

# ========================
# Config
# ========================
SEED = 42
N_PERM = args.n
K_VAL = 0
WALK_LENGTH = 8 * N_PERM
N_WALKS = args.walks
MAX_TRAIN = args.max_train
N_TEST_WALKS = 2_000
VALIDATION_FRAC = 0.15
EARLY_STOPPING_ROUNDS = 100

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.1,
    'n_estimators': args.lgb_trees, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.01, 'reg_lambda': 0.01,
    'verbose': -1, 'seed': SEED, 'n_jobs': -1,
}

MLP_HIDDEN_DIM = 128
MLP_EPOCHS = args.mlp_epochs
MLP_BATCH_SIZE = 1024
MLP_LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cpu')
print(f'device: {device}  |  lightgbm: {lgb.__version__}')

# ========================
# Koltsov3 Generators
# ========================
def get_koltsov3_moves(n, k=0):
    I = np.arange(n); K = np.arange(n); S = np.arange(n)
    for i in range(0, n - 1, 2):  I[i], I[i + 1] = I[i + 1], I[i]
    for i in range(1, n - 1, 2):  K[i], K[i + 1] = K[i + 1], K[i]
    S[k], S[k + 2] = S[k + 2], S[k]
    return [I, K, S]

moves_list = get_koltsov3_moves(N_PERM, K_VAL)
moves_t = torch.tensor(np.array(moves_list), dtype=torch.long)
state_dest = torch.arange(N_PERM, dtype=torch.int64)
n_gen = moves_t.shape[0]
print(f'Generators: {[list(g) for g in moves_list]}')
print(f'n={N_PERM}, walk_length={WALK_LENGTH}, n_walks={N_WALKS:,}')

# ========================
# Random Walk Generation
# ========================
print('\n--- Generating random walks ---')
t0 = time.time()
states = state_dest.unsqueeze(0).repeat(N_WALKS, 1).to(torch.uint8)
all_s, all_l = [], []
for step in range(1, WALK_LENGTH + 1):
    move_ids = torch.randint(0, n_gen, (N_WALKS,))
    states = torch.gather(states, 1, moves_t[move_ids])
    all_s.append(states.clone())
    all_l.append(torch.full((N_WALKS,), step, dtype=torch.float32))
all_states_raw = torch.cat(all_s, dim=0)
all_labels_raw = torch.cat(all_l, dim=0)
total = len(all_labels_raw)
print(f'  {total:,} samples in {time.time()-t0:.1f}s')

if total > MAX_TRAIN:
    idx = np.random.choice(total, MAX_TRAIN, replace=False)
    states_arr = all_states_raw[idx].numpy().astype(np.int32)
    labels_arr = all_labels_raw[idx].numpy().astype(np.float32)
    print(f'  Subsampled to {MAX_TRAIN:,}')
else:
    states_arr = all_states_raw.numpy().astype(np.int32)
    labels_arr = all_labels_raw.numpy().astype(np.float32)
print(f'  Labels range: [{labels_arr.min():.0f}, {labels_arr.max():.0f}]')

# ========================
# Feature Extraction
# ========================
print('\n--- Extracting features ---')
t0 = time.time()

def extract_features(states_array, n, k=0):
    N = states_array.shape[0]
    f = {}
    ident = np.arange(n)
    
    # 1. Displacement
    disp = np.abs(states_array - ident)
    for i in range(n): f[f'd_{i}'] = disp[:, i]
    f['d_sum'] = disp.sum(1); f['d_max'] = disp.max(1); f['d_mean'] = disp.mean(1)
    f['d_std'] = disp.std(1); f['d_med'] = np.median(disp, 1)
    f['d_eq0'] = (disp == 0).sum(1); f['d_eq1'] = (disp == 1).sum(1)
    f['d_ge2'] = (disp >= 2).sum(1); f['d_ge4'] = (disp >= 4).sum(1)
    
    # 2. Parity mismatch
    pm = ((states_array & 1) != (ident & 1)).astype(np.float32)
    for i in range(n): f[f'pm_{i}'] = pm[:, i]
    f['pm_sum'] = pm.sum(1); f['pm_pct'] = pm.mean(1)
    f['pm_x_disp'] = (pm * disp).sum(1)
    
    # 3. Adjacent pairs
    sorted_adj = (states_array[:, :-1] < states_array[:, 1:]).astype(np.float32)
    for i in range(n - 1): f[f's_{i}{i+1}'] = sorted_adj[:, i]
    f['s_sum'] = sorted_adj.sum(1)
    adj_diff = np.abs(np.diff(states_array, axis=1))
    for i in range(n - 1): f[f'ad_{i}'] = adj_diff[:, i]
    f['ad_sum'] = adj_diff.sum(1); f['ad_max'] = adj_diff.max(1)
    pair_correct = ((states_array[:, :-1] == ident[:-1]) &
                    (states_array[:, 1:] == ident[1:])).astype(np.float32)
    f['pair_correct'] = pair_correct.sum(1)
    
    # 4. Inversions
    invs = np.zeros(N, dtype=np.float32)
    for i in range(n):
        invs += (states_array[:, i:i+1] > states_array[:, i+1:]).sum(axis=1)
    f['inv'] = invs
    f['inv_frac'] = invs / (n * (n - 1) / 2)
    f['inv_parity'] = (invs.astype(np.int64) & 1).astype(np.float32)
    
    # 5. Descents
    descs = np.zeros(N, dtype=np.float32)
    for i in range(n - 1):
        descs += (states_array[:, i] > states_array[:, i + 1]).astype(np.float32)
    f['desc'] = descs
    
    # 6. I-generator features
    ieu = np.zeros(N, dtype=np.float32); iec = np.zeros(N, dtype=np.float32)
    for idx in range(0, n - 1, 2):
        ieu += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        iec += ((states_array[:, idx] == idx) & (states_array[:, idx + 1] == idx + 1)).astype(np.float32)
    f['I_unsorted'] = ieu; f['I_correct'] = iec
    
    # 7. K-generator features
    keu = np.zeros(N, dtype=np.float32); kec = np.zeros(N, dtype=np.float32)
    for idx in range(1, n - 1, 2):
        keu += (states_array[:, idx] > states_array[:, idx + 1]).astype(np.float32)
        kec += ((states_array[:, idx] == idx) & (states_array[:, idx + 1] == idx + 1)).astype(np.float32)
    f['K_unsorted'] = keu; f['K_correct'] = kec
    
    # 8. S-generator features
    if k + 2 < n:
        f['S_sorted'] = (states_array[:, k] < states_array[:, k + 2]).astype(np.float32)
        f['S_diff'] = np.abs(states_array[:, k].astype(np.float32) - states_array[:, k+2].astype(np.float32))
        f['S_disp'] = np.abs(states_array[:, k] - k) + np.abs(states_array[:, k+2] - (k+2))
        f['S_need_swap'] = ((states_array[:, k] == k + 2) & (states_array[:, k + 2] == k)).astype(np.float32)
    
    # 9. Theoretical bounds
    f['lb_disp'] = disp.max(1) / 2.0
    f['lb_parity'] = pm.sum(1) / 2.0
    f['lb_comb'] = np.maximum(f['lb_disp'], f['lb_parity'])
    
    # 10. Inverse permutation
    inv_perm = np.argsort(states_array, axis=1)
    for v in range(n): f[f'pos_of_{v}'] = inv_perm[:, v].astype(np.float32)
    f['where_0'] = np.abs(inv_perm[:, 0] - 0).astype(np.float32)
    f['where_n1'] = np.abs(inv_perm[:, n - 1] - (n - 1)).astype(np.float32)
    
    # 11. Correct runs
    at_home = (states_array == ident)
    run_lens = np.zeros(N, dtype=np.float32)
    for i in range(N):
        cur = best = 0
        for x in at_home[i]:
            if x: cur += 1; best = max(best, cur)
            else: cur = 0
        run_lens[i] = best
    f['run_correct'] = run_lens
    
    # 12. Skew
    sd = states_array.astype(np.float32) - ident.astype(np.float32)
    f['disp_pos'] = np.maximum(sd, 0).sum(1)
    f['disp_neg'] = np.maximum(-sd, 0).sum(1)
    f['disp_net'] = sd.sum(1)
    
    # 13. Raw positions
    for i in range(n): f[f'p_{i}'] = states_array[:, i].astype(np.float32)
    
    return pd.DataFrame(f)

df_features = extract_features(states_arr, N_PERM, k=K_VAL)
df_features['label'] = labels_arr
feature_cols = [c for c in df_features.columns if c != 'label']
print(f'  {len(feature_cols)} features extracted in {time.time()-t0:.1f}s')

# ========================
# Train/Val Split
# ========================
X_train, X_val, y_train, y_val = train_test_split(
    df_features[feature_cols], df_features['label'],
    test_size=VALIDATION_FRAC, random_state=SEED
)
print(f'Train: {len(X_train):,}  |  Val: {len(X_val):,}')

# ========================
# Train LightGBM
# ========================
print('\n--- Training LightGBM ---')
t0 = time.time()
lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(feature_cols))
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=list(feature_cols))
evals_result = {}
model = lgb.train(
    params=LGB_PARAMS,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=200),
    ],
)
t_train = time.time() - t0
best_score = model.best_score
if isinstance(best_score, dict):
    best_score = best_score['val']['rmse']
print(f'Training: {t_train:.1f}s  |  Best iter: {model.best_iteration}  |  Best RMSE: {best_score:.2f}')

# ========================
# Evaluate on train/val
# ========================
def evaluate(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'spearman': stats.spearmanr(y_true, y_pred).statistic,
    }

y_tr_pred = model.predict(X_train)
y_va_pred = model.predict(X_val)
m_tr = evaluate(y_train, y_tr_pred)
m_va = evaluate(y_val, y_va_pred)
baseline = np.sqrt(np.mean((y_val - y_val.mean())**2))

print(f'\n{"="*60}')
print(f'LightGBM (n={N_PERM})')
print(f'{"="*60}')
print(f'              R²        RMSE      Spearman')
print(f'Train:    {m_tr["r2"]:8.4f}  {m_tr["rmse"]:8.2f}  {m_tr["spearman"]:10.4f}')
print(f'Val:      {m_va["r2"]:8.4f}  {m_va["rmse"]:8.2f}  {m_va["spearman"]:10.4f}')
print(f'Baseline RMSE: {baseline:.2f}')

# ========================
# Test Set
# ========================
print('\n--- Generating test set ---')
t0 = time.time()
ts = state_dest.unsqueeze(0).repeat(N_TEST_WALKS, 1).to(torch.uint8)
ts_list, tl_list = [], []
for step in range(1, WALK_LENGTH + 1):
    move_ids = torch.randint(0, n_gen, (N_TEST_WALKS,))
    ts = torch.gather(ts, 1, moves_t[move_ids])
    ts_list.append(ts.clone())
    tl_list.append(torch.full((N_TEST_WALKS,), step, dtype=torch.float32))
test_states = torch.cat(ts_list, dim=0).numpy().astype(np.int32)
test_labels = torch.cat(tl_list, dim=0).numpy().astype(np.float32)
df_test = extract_features(test_states, N_PERM, k=K_VAL)
X_test = df_test
y_test = test_labels
print(f'  {len(y_test):,} test samples in {time.time()-t0:.1f}s')

# ========================
# LightGBM Test
# ========================
t_pred = time.time()
y_test_pred_lgb = model.predict(X_test)
t_pred_lgb = time.time() - t_pred
m_test_lgb = evaluate(y_test, y_test_pred_lgb)

print(f'\n=== LightGBM Test ===')
print(f'R²={m_test_lgb["r2"]:.4f}  RMSE={m_test_lgb["rmse"]:.2f}  Spearman={m_test_lgb["spearman"]:.4f}')
print(f'Inference: {t_pred_lgb*1000:.1f}ms for {len(y_test):,} samples ({len(y_test)/t_pred_lgb:.0f} samples/s)')

# ========================
# MLP Baseline
# ========================
print('\n--- Training MLP baseline ---')

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

mlp_n = min(len(states_arr), 100_000)
mlp_idx = np.random.choice(len(states_arr), mlp_n, replace=False)
X_tr_mlp = torch.tensor(states_arr[mlp_idx], dtype=torch.uint8)
y_tr_mlp = torch.tensor(labels_arr[mlp_idx], dtype=torch.float32)
X_te_mlp = torch.tensor(test_states, dtype=torch.uint8)
y_te_mlp = torch.tensor(test_labels, dtype=torch.float32)
print(f'  MLP train: {len(X_tr_mlp):,}  test: {len(X_te_mlp):,}')

mlp = MLP(N_PERM, MLP_HIDDEN_DIM, N_PERM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=MLP_LR)
mlp_train_losses, mlp_val_losses = [], []

t0 = time.time()
for epoch in range(MLP_EPOCHS):
    mlp.train()
    perm = torch.randperm(len(X_tr_mlp))
    Xs, ys = X_tr_mlp[perm], y_tr_mlp[perm]
    train_loss = 0.0; n_batches = 0
    for i in range(0, len(Xs), MLP_BATCH_SIZE):
        bx = Xs[i:i+MLP_BATCH_SIZE]; by = ys[i:i+MLP_BATCH_SIZE]
        loss = criterion(mlp(bx).squeeze(), by)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item(); n_batches += 1
    mlp_train_losses.append(train_loss / n_batches)
    mlp.eval()
    with torch.no_grad():
        vi = np.random.choice(len(X_te_mlp), min(20000, len(X_te_mlp)), replace=False)
        mlp_val_losses.append(criterion(mlp(X_te_mlp[vi]).squeeze(), y_te_mlp[vi]).item())
    if epoch % 3 == 0:
        print(f'  MLP epoch {epoch:2d}: train={mlp_train_losses[-1]:.1f}, val={mlp_val_losses[-1]:.1f}')
t_mlp = time.time() - t0
print(f'  MLP training: {t_mlp:.1f}s')

# ========================
# MLP Test
# ========================
mlp.eval()
t_pred = time.time()
with torch.no_grad():
    y_test_pred_mlp = mlp(X_te_mlp).squeeze().numpy()
t_pred_mlp = time.time() - t_pred
m_test_mlp = evaluate(y_test, y_test_pred_mlp)

print(f'\n=== MLP Test ===')
print(f'R²={m_test_mlp["r2"]:.4f}  RMSE={m_test_mlp["rmse"]:.2f}  Spearman={m_test_mlp["spearman"]:.4f}')

# ========================
# Head-to-Head
# ========================
print(f'\n{"="*60}')
print(f'HEAD-TO-HEAD (n={N_PERM})')
print(f'{"="*60}')
print(f'{"Model":12s}  {"R²":>8s}  {"RMSE":>8s}  {"Spearman":>10s}  {"Train":>8s}')
print(f'{"LightGBM":12s}  {m_test_lgb["r2"]:8.4f}  {m_test_lgb["rmse"]:8.2f}  {m_test_lgb["spearman"]:10.4f}  {t_train:7.1f}s')
print(f'{"MLP":12s}  {m_test_mlp["r2"]:8.4f}  {m_test_mlp["rmse"]:8.2f}  {m_test_mlp["spearman"]:10.4f}  {t_mlp:7.1f}s')
print(f'\nLightGBM features: {len(feature_cols)}  |  MLP features (one-hot): {N_PERM * N_PERM}')

# ========================
# Feature Importance
# ========================
imp = pd.DataFrame({
    'feature': model.feature_name(),
    'gain': model.feature_importance(importance_type='gain'),
}).sort_values('gain', ascending=False)

print(f'\n--- Top 15 Features by Gain ---')
for _, row in imp.head(15).iterrows():
    print(f'  {row["feature"]:25s}  {row["gain"]:12.0f}')

fig, ax = plt.subplots(figsize=(8, 6))
top20 = imp.head(20)
ax.barh(range(20), top20['gain'].values, color='steelblue')
ax.set_yticks(range(20))
ax.set_yticklabels(top20['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_title(f'Top 20 Features by Gain (n={N_PERM})')
ax.set_xlabel('Gain')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
print('Saved feature_importance.png')

# ========================
# Plots
# ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, yp, name, color, met in [
    (axes[0], y_test_pred_lgb, 'LightGBM', 'green', m_test_lgb),
    (axes[1], y_test_pred_mlp, 'MLP', 'blue', m_test_mlp),
]:
    ax.scatter(y_test, yp, alpha=0.2, s=8, c=color, edgecolors='none')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('True Distance (step)'); ax.set_ylabel('Prediction')
    ax.set_title(f'{name}: R²={met["r2"]:.3f}, RMSE={met["rmse"]:.1f}, ρ={met["spearman"]:.3f}')
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('predictions_vs_true.png', dpi=100, bbox_inches='tight')
print('Saved predictions_vs_true.png')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(evals_result['train']['rmse'], label='Train', color='green')
axes[0].plot(evals_result['val']['rmse'], label='Val', color='orange')
axes[0].axvline(x=model.best_iteration, color='red', linestyle='--', label=f'Best')
axes[0].set_title('LightGBM Training'); axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('RMSE')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(mlp_train_losses, label='Train', color='blue')
axes[1].plot(mlp_val_losses, label='Val', color='orange')
axes[1].set_title('MLP Training'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
print('Saved training_curves.png')

print('\n✓ ALL DONE')
