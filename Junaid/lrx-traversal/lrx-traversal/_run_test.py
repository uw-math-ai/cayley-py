# --- Cell 1 ---
import random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cpu')
print(f'device: {device}  |  lightgbm: {lgb.__version__}')

# --- Cell 2 ---
# ========================
# Configuration
# ========================
N_PERM = 16                # permutation length
K_VAL = 0                  # S = (k, k+2)
WALK_LENGTH = 8 * N_PERM   # heuristic for Koltsov3
N_WALKS = 25_000           # number of independent walks
MAX_TRAIN = 500_000        # subsample to this many max
N_TEST_WALKS = 2_000       # test set size in walks

# LightGBM hyperparameters
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'verbose': -1,
    'seed': SEED,
    'n_jobs': -1,
}
VALIDATION_FRAC = 0.15
EARLY_STOPPING_ROUNDS = 100

# MLP baseline
MLP_HIDDEN_DIM = 128
MLP_EPOCHS = 15
MLP_BATCH_SIZE = 1024
MLP_LR = 0.001

# --- Cell 3 ---
# ========================
# Koltsov3 Generators
# ========================
def get_koltsov3_moves(n, k=0):
    """Koltsov3: I=(0,1)(2,3)..., K=(1,2)(3,4)..., S=(k,k+2)"""
    I = np.arange(n); K = np.arange(n); S = np.arange(n)
    for i in range(0, n-1, 2): I[i], I[i+1] = I[i+1], I[i]
    for i in range(1, n-1, 2): K[i], K[i+1] = K[i+1], K[i]
    S[k], S[k+2] = S[k+2], S[k]
    return [I, K, S]

moves_list = get_koltsov3_moves(N_PERM, K_VAL)
moves_t = torch.tensor(np.array(moves_list), dtype=torch.long)
state_dest = torch.arange(N_PERM, dtype=torch.int64)
n_gen = moves_t.shape[0]

for i, g in enumerate(moves_list):
    print(f'  Gen {i}: {g}')
print(f'n={N_PERM}, walk_length={WALK_LENGTH}, n_walks={N_WALKS:,}')

# --- Cell 4 ---
# ========================
# Random Walk Generation (Vectorized)
# ========================
print('Generating random walks...')
t0 = time.time()

states = state_dest.unsqueeze(0).repeat(N_WALKS, 1).to(torch.uint8)
all_states_list, all_labels_list = [], []

for step in range(1, WALK_LENGTH + 1):
    move_ids = torch.randint(0, n_gen, (N_WALKS,))
    states = torch.gather(states, 1, moves_t[move_ids])
    all_states_list.append(states.clone())
    all_labels_list.append(torch.full((N_WALKS,), step, dtype=torch.float32))

all_states_raw = torch.cat(all_states_list, dim=0)
all_labels_raw = torch.cat(all_labels_list, dim=0)
total_samples = len(all_labels_raw)
print(f'  {total_samples:,} samples in {time.time()-t0:.1f}s')

# Subsample
if total_samples > MAX_TRAIN:
    idx = np.random.choice(total_samples, MAX_TRAIN, replace=False)
    states_arr = all_states_raw[idx].numpy().astype(np.int32)
    labels_arr = all_labels_raw[idx].numpy().astype(np.float32)
    print(f'  Subsampled to {MAX_TRAIN:,}')
else:
    states_arr = all_states_raw.numpy().astype(np.int32)
    labels_arr = all_labels_raw.numpy().astype(np.float32)

print(f'  Training data: {len(labels_arr):,} samples, labels in [{labels_arr.min():.0f}, {labels_arr.max():.0f}]')

# --- Cell 5 ---
# ========================
# Feature Extraction
# ========================
def extract_features(states_array, n, k=0):
    """
    Generator-aware features for Koltsov3.
    I=(0,1)(2,3)..., K=(1,2)(3,4)..., S=(k,k+2)
    I and K preserve position parity; only S breaks it.
    All moves shift elements by at most 2 positions.
    """
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

    # 2. Parity mismatch (I,K preserve parity; only S breaks it)
    pm = ((states_array & 1) != (ident & 1)).astype(np.float32)
    for i in range(n): f[f'pm_{i}'] = pm[:, i]
    f['pm_sum'] = pm.sum(1); f['pm_pct'] = pm.mean(1)
    f['pm_x_disp'] = (pm * disp).sum(1)

    # 3. Adjacent pair structure
    sorted_adj = (states_array[:, :-1] < states_array[:, 1:]).astype(np.float32)
    for i in range(n-1): f[f's_{i}{i+1}'] = sorted_adj[:, i]
    f['s_sum'] = sorted_adj.sum(1)
    adj_diff = np.abs(np.diff(states_array, axis=1))
    for i in range(n-1): f[f'ad_{i}'] = adj_diff[:, i]
    f['ad_sum'] = adj_diff.sum(1); f['ad_max'] = adj_diff.max(1)
    pair_correct = ((states_array[:, :-1] == ident[:-1]) & 
                     (states_array[:, 1:] == ident[1:])).astype(np.float32)
    f['pair_correct'] = pair_correct.sum(1)

    # 4. Inversions
    invs = np.zeros(N, dtype=np.float32)
    for i in range(n):
        invs += (states_array[:, i:i+1] > states_array[:, i+1:]).sum(axis=1)
    f['inv'] = invs
    f['inv_frac'] = invs / (n * (n-1) / 2)
    f['inv_parity'] = (invs.astype(np.int64) & 1).astype(np.float32)

    # 5. Descents
    descs = np.zeros(N, dtype=np.float32)
    for i in range(n-1):
        descs += (states_array[:, i] > states_array[:, i+1]).astype(np.float32)
    f['desc'] = descs

    # 6. Generator-specific: I-pair structure
    ieu = np.zeros(N, dtype=np.float32)
    iec = np.zeros(N, dtype=np.float32)
    for idx in range(0, n-1, 2):
        ieu += (states_array[:, idx] > states_array[:, idx+1]).astype(np.float32)
        iec += ((states_array[:, idx] == idx) & (states_array[:, idx+1] == idx+1)).astype(np.float32)
    f['I_unsorted'] = ieu; f['I_correct'] = iec

    # 7. Generator-specific: K-pair structure
    keu = np.zeros(N, dtype=np.float32)
    kec = np.zeros(N, dtype=np.float32)
    for idx in range(1, n-1, 2):
        keu += (states_array[:, idx] > states_array[:, idx+1]).astype(np.float32)
        kec += ((states_array[:, idx] == idx) & (states_array[:, idx+1] == idx+1)).astype(np.float32)
    f['K_unsorted'] = keu; f['K_correct'] = kec

    # 8. Generator-specific: S-pair
    if k + 2 < n:
        p_k = states_array[:, k].astype(np.float32)
        p_k2 = states_array[:, k+2].astype(np.float32)
        f['S_sorted'] = (p_k < p_k2).astype(np.float32)
        f['S_diff'] = np.abs(p_k - p_k2)
        f['S_disp'] = np.abs(states_array[:, k] - k) + np.abs(states_array[:, k+2] - (k+2))
        f['S_need_swap'] = ((states_array[:, k] == k+2) & (states_array[:, k+2] == k)).astype(np.float32)

    # 9. Theoretical lower bounds
    f['lb_disp'] = disp.max(1) / 2.0
    f['lb_parity'] = pm.sum(1) / 2.0
    f['lb_comb'] = np.maximum(f['lb_disp'], f['lb_parity'])

    # 10. Inverse permutation (where is each value?)
    inv_perm = np.argsort(states_array, axis=1)
    for v in range(n):
        f[f'pos_of_{v}'] = inv_perm[:, v].astype(np.float32)
    f['where_0'] = np.abs(inv_perm[:, 0] - 0).astype(np.float32)
    f['where_n1'] = np.abs(inv_perm[:, n-1] - (n-1)).astype(np.float32)

    # 11. Correct position runs
    at_home = (states_array == ident)
    run_lens = np.zeros(N, dtype=np.float32)
    for i in range(N):
        cur = best = 0
        for x in at_home[i]:
            if x: cur += 1; best = max(best, cur)
            else: cur = 0
        run_lens[i] = best
    f['run_correct'] = run_lens

    # 12. Displacement skew
    sd = states_array.astype(np.float32) - ident.astype(np.float32)
    f['disp_pos'] = np.maximum(sd, 0).sum(1)
    f['disp_neg'] = np.maximum(-sd, 0).sum(1)
    f['disp_net'] = sd.sum(1)

    # 13. Raw positions
    for i in range(n):
        f[f'p_{i}'] = states_array[:, i].astype(np.float32)

    return pd.DataFrame(f)


print('Extracting features...')
t0 = time.time()
df_features = extract_features(states_arr, N_PERM, k=K_VAL)
df_features['label'] = labels_arr
feature_cols = [c for c in df_features.columns if c != 'label']
print(f'  {df_features.shape[1]} features ({len(feature_cols)} + label) in {time.time()-t0:.1f}s')
df_features.head()

# --- Cell 6 ---
# ========================
# Train / Validation Split
# ========================
X = df_features[feature_cols]
y = df_features['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_FRAC, random_state=SEED
)
print(f'Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Features: {X.shape[1]}')

# --- Cell 7 ---
# ========================
# Train LightGBM
# ========================
print('Training LightGBM...')
t0 = time.time()

# Pass DataFrames directly so feature names are preserved
lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(feature_cols))
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=list(feature_cols))

model = lgb.train(
    params=LGB_PARAMS,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=200),
    ],
)

t_train = time.time() - t0
best_score = model.best_score
if isinstance(best_score, dict):
    best_score = best_score['val']['rmse']
print(f'\nTraining time: {t_train:.1f}s  |  Best iter: {model.best_iteration}  |  Best RMSE: {best_score:.2f}')

# --- Cell 8 ---
# ========================
# LightGBM Evaluation (Train + Val)
# ========================
def evaluate_predictions(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'spearman': stats.spearmanr(y_true, y_pred).statistic,
    }

y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)

m_tr = evaluate_predictions(y_train, y_train_pred)
m_va = evaluate_predictions(y_val,   y_val_pred)
baseline_rmse = np.sqrt(np.mean((y_val - y_val.mean())**2))

print(f'              R²        RMSE      Spearman')
print(f'Train:    {m_tr["r2"]:8.4f}  {m_tr["rmse"]:8.2f}  {m_tr["spearman"]:10.4f}')
print(f'Val:      {m_va["r2"]:8.4f}  {m_va["rmse"]:8.2f}  {m_va["spearman"]:10.4f}')
print(f'Baseline RMSE (mean): {baseline_rmse:.2f}')
print(f'RMSE gap: {m_va["rmse"] - m_tr["rmse"]:.2f}')

# --- Cell 9 ---
# ========================
# Generate Test Set (fresh walks, unseen states)
# ========================
print('Generating test set...')
t0 = time.time()

test_states_t = state_dest.unsqueeze(0).repeat(N_TEST_WALKS, 1).to(torch.uint8)
test_s_list, test_l_list = [], []
for step in range(1, WALK_LENGTH + 1):
    move_ids = torch.randint(0, n_gen, (N_TEST_WALKS,))
    test_states_t = torch.gather(test_states_t, 1, moves_t[move_ids])
    test_s_list.append(test_states_t.clone())
    test_l_list.append(torch.full((N_TEST_WALKS,), step, dtype=torch.float32))

test_states = torch.cat(test_s_list, dim=0).numpy().astype(np.int32)
test_labels = torch.cat(test_l_list, dim=0).numpy().astype(np.float32)

df_test = extract_features(test_states, N_PERM, k=K_VAL)
X_test = df_test
y_test = test_labels

print(f'  Test samples: {len(y_test):,} ({N_TEST_WALKS} walks x {WALK_LENGTH} steps)')
print(f'  Generation + features: {time.time()-t0:.1f}s')

# --- Cell 10 ---
# ========================
# LightGBM Test Evaluation
# ========================
t_pred = time.time()
y_test_pred_lgb = model.predict(X_test)
pred_time = time.time() - t_pred

m_test = evaluate_predictions(y_test, y_test_pred_lgb)

print(f'=== LightGBM Test Metrics ===')
print(f'R²:        {m_test["r2"]:.4f}')
print(f'RMSE:      {m_test["rmse"]:.2f}')
print(f'Spearman:  {m_test["spearman"]:.4f}')
print(f'Inference: {pred_time*1000:.1f} ms for {len(y_test):,} samples ')
       f'({len(y_test)/pred_time:.0f} samples/s)')

# --- Cell 11 ---
# ========================
# Feature Importance
# ========================
imp = pd.DataFrame({
    'feature': model.feature_name(),
    'gain': model.feature_importance(importance_type='gain'),
    'split': model.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top20 = imp.head(20)
axes[0].barh(range(20), top20['gain'].values, color='steelblue')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top20['feature'].values, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_title('Top 20 Features by Gain')
axes[0].set_xlabel('Gain')

top20s = imp.sort_values('split', ascending=False).head(20)
axes[1].barh(range(20), top20s['split'].values, color='darkorange')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top20s['feature'].values, fontsize=9)
axes[1].invert_yaxis()
axes[1].set_title('Top 20 Features by Split Count')
axes[1].set_xlabel('Split Count')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

# --- Cell 12 ---
# ========================
# MLP Baseline
# ========================
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
        x = torch.nn.functional.one_hot(
            x.long(), num_classes=self.num_classes
        ).float().flatten(start_dim=-2)
        return self.layers(x)


# Subsample train data for MLP
mlp_n = min(len(states_arr), 100_000)
mlp_idx = np.random.choice(len(states_arr), mlp_n, replace=False)
X_tr_mlp = torch.tensor(states_arr[mlp_idx], dtype=torch.uint8)
y_tr_mlp = torch.tensor(labels_arr[mlp_idx], dtype=torch.float32)
X_te_mlp = torch.tensor(test_states, dtype=torch.uint8)
y_te_mlp = torch.tensor(test_labels, dtype=torch.float32)

print(f'MLP train: {len(X_tr_mlp):,}  |  test: {len(X_te_mlp):,}')

mlp = MLP(N_PERM, MLP_HIDDEN_DIM, N_PERM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=MLP_LR)

mlp_train_losses = []
mlp_val_losses = []

t0 = time.time()
for epoch in range(MLP_EPOCHS):
    mlp.train()
    perm = torch.randperm(len(X_tr_mlp))
    Xs, ys = X_tr_mlp[perm], y_tr_mlp[perm]
    
    train_loss = 0.0
    n_batches = 0
    for i in range(0, len(Xs), MLP_BATCH_SIZE):
        bx = Xs[i:i+MLP_BATCH_SIZE]
        by = ys[i:i+MLP_BATCH_SIZE]
        out = mlp(bx).squeeze()
        loss = criterion(out, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    train_loss /= n_batches
    mlp_train_losses.append(train_loss)
    
    mlp.eval()
    with torch.no_grad():
        vi = np.random.choice(len(X_te_mlp), min(20000, len(X_te_mlp)), replace=False)
        val_loss = criterion(mlp(X_te_mlp[vi]).squeeze(), y_te_mlp[vi]).item()
        mlp_val_losses.append(val_loss)
    
    if epoch % 3 == 0:
        print(f'  MLP epoch {epoch:2d}: train_loss={train_loss:.2f}, val_loss={val_loss:.2f}')

t_mlp = time.time() - t0
print(f'MLP training: {t_mlp:.1f}s')

# --- Cell 13 ---
# ========================
# MLP Test Evaluation
# ========================
mlp.eval()
t_pred_mlp = time.time()
with torch.no_grad():
    y_test_pred_mlp = mlp(X_te_mlp).squeeze().numpy()
t_pred_mlp = time.time() - t_pred_mlp

m_mlp = evaluate_predictions(y_test, y_test_pred_mlp)

print(f'=== MLP Test Metrics ===')
print(f'R²:        {m_mlp["r2"]:.4f}')
print(f'RMSE:      {m_mlp["rmse"]:.2f}')
print(f'Spearman:  {m_mlp["spearman"]:.4f}')
print(f'Inference: {t_pred_mlp*1000:.1f} ms for {len(y_test):,} samples')

# --- Cell 14 ---
# ========================
# Head-to-Head Comparison
# ========================
print('\n' + '='*65)
print(f'COMPARISON: LightGBM vs MLP  (n={N_PERM})')
print('='*65)

comparison = pd.DataFrame([
    {'Model': 'LightGBM', 'R²': round(m_test['r2'], 4), 
     'RMSE': round(m_test['rmse'], 2), 'Spearman': round(m_test['spearman'], 4),
     'Train time': f'{t_train:.1f}s'},
    {'Model': 'MLP', 'R²': round(m_mlp['r2'], 4),
     'RMSE': round(m_mlp['rmse'], 2), 'Spearman': round(m_mlp['spearman'], 4),
     'Train time': f'{t_mlp:.1f}s'},
])
print(comparison.to_string(index=False))
print(f'\nLightGBM features: {X.shape[1]}  |  MLP features (one-hot): {N_PERM * N_PERM}')

# --- Cell 15 ---
# ========================
# Scatter: Predictions vs True
# ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, yp, name, color, met in [
    (axes[0], y_test_pred_lgb, 'LightGBM', 'green', m_test),
    (axes[1], y_test_pred_mlp, 'MLP', 'blue', m_mlp),
]:
    ax.scatter(y_test, yp, alpha=0.2, s=8, c=color, edgecolors='none')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('True Distance (step)')
    ax.set_ylabel('Prediction')
    ax.set_title(f'{name}: R²={met["r2"]:.3f}, RMSE={met["rmse"]:.1f}, ρ={met["spearman"]:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_vs_true.png', dpi=100, bbox_inches='tight')
plt.show()

# --- Cell 16 ---
# ========================
# Residual Analysis
# ========================
res_lgb = y_test - y_test_pred_lgb
res_mlp = y_test - y_test_pred_mlp

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_test, res_lgb, alpha=0.2, s=8, c='green')
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('LightGBM Residuals')
axes[0, 0].set_xlabel('True Distance'); axes[0, 0].set_ylabel('Residual')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test, res_mlp, alpha=0.2, s=8, c='blue')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_title('MLP Residuals')
axes[0, 1].set_xlabel('True Distance'); axes[0, 1].set_ylabel('Residual')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(res_lgb, bins=60, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='r', linestyle='--')
axes[1, 0].set_title(f'LightGBM (μ={res_lgb.mean():.2f}, σ={res_lgb.std():.2f})')
axes[1, 0].set_xlabel('Residual')

axes[1, 1].hist(res_mlp, bins=60, color='blue', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_title(f'MLP (μ={res_mlp.mean():.2f}, σ={res_mlp.std():.2f})')
axes[1, 1].set_xlabel('Residual')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# --- Cell 17 ---
# ========================
# Training Curves
# ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

evals = model.evals_result_
axes[0].plot(evals['train']['rmse'], label='Train', color='green')
axes[0].plot(evals['val']['rmse'], label='Validation', color='orange')
axes[0].axvline(x=model.best_iteration, color='red', linestyle='--', 
               label=f'Best ({model.best_iteration})')
axes[0].set_title('LightGBM Training Curve')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('RMSE')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(mlp_train_losses, label='Train MSE', color='blue')
axes[1].plot(mlp_val_losses, label='Test MSE', color='orange')
axes[1].set_title('MLP Training Curve')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
plt.show()
