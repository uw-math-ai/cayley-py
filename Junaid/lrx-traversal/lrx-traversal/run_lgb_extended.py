#!/usr/bin/env python3
"""
Extended LightGBM test: push the best tree configs further.

The main sweep showed LGB plateauing at R²≈0.67-0.69 with 1000 trees.
This script tests ideas to break the plateau:

1. More boosting rounds (2000, 3000) — training never early-stopped at 1000
2. More leaves (511) — current max was 255
3. DART boosting — dropout for trees, prevents overfitting
4. Lower learning rate with more trees — finer steps
5. Huber loss — less sensitive to label noise from random walks

Usage:
    python run_lgb_extended.py
    python run_lgb_extended.py --quick
"""

import random, time, os, sys, argparse, itertools, json, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
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
random.seed(SEED); np.random.seed(SEED)

K_VAL = 0; VALIDATION_FRAC = 0.15; EARLY_STOPPING = 150
RESULTS_CSV = 'lgb_extended_results.csv'
MODELS_DIR = 'lgb_models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Test grid
N_VALUES = [16, 32, 48, 64] if args.quick else [16, 32, 48, 64]

# Each config tests one idea vs baseline
CONFIGS = [
    # (name, n_estimators, num_leaves, lr, boosting, objective, extra_params)
    # --- Baselines ---
    ('baseline_1k',   1000, 127, 0.05, 'gbdt', 'rmse', {}),  # sweep best
    ('baseline_255L', 1000, 255, 0.05, 'gbdt', 'rmse', {}),  # larger leaves
    
    # --- More trees ---
    ('more_trees_2k',   2000, 127, 0.05, 'gbdt', 'rmse', {}),
    ('more_trees_3k',   3000, 127, 0.05, 'gbdt', 'rmse', {}),
    ('more_trees_2k_255L', 2000, 255, 0.05, 'gbdt', 'rmse', {}),
    
    # --- More leaves ---
    ('leaves_511',     2000, 511, 0.05, 'gbdt', 'rmse', {}),
    
    # --- Lower LR + more trees ---
    ('low_lr_0.02',    3000, 127, 0.02, 'gbdt', 'rmse', {}),
    ('low_lr_0.01',    3000, 127, 0.01, 'gbdt', 'rmse', {}),
    
    # --- DART boosting (dropout for trees) ---
    ('dart_2k',         2000, 127, 0.05, 'dart', 'rmse', 
     {'drop_rate': 0.1, 'max_drop': 50, 'skip_drop': 0.5}),
    ('dart_3k',         3000, 127, 0.03, 'dart', 'rmse', 
     {'drop_rate': 0.1, 'max_drop': 50, 'skip_drop': 0.5}),
    
    # --- Huber loss (robust to outliers) ---
    ('huber_2k',        2000, 127, 0.05, 'gbdt', 'huber', {'alpha': 0.9}),
    ('huber_3k',        3000, 127, 0.05, 'gbdt', 'huber', {'alpha': 0.9}),
]

total_runs = len(N_VALUES) * len(CONFIGS)
print(f'Configs: {len(CONFIGS)} × {len(N_VALUES)} n = {total_runs} runs')

# =============================================================================
# Koltsov3 + features (same as main sweep)
# =============================================================================
import torch

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
    return (torch.cat(all_s, 0).numpy().astype(np.int32),
            torch.cat(all_l, 0).numpy().astype(np.float32))

def extract_features(states_array, n, k=0):
    N = states_array.shape[0]; f = {}; ident = np.arange(n)
    disp = np.abs(states_array - ident)
    for i in range(n): f[f'd_{i}'] = disp[:, i]
    f['d_sum'] = disp.sum(1); f['d_max'] = disp.max(1); f['d_mean'] = disp.mean(1)
    f['d_std'] = disp.std(1); f['d_med'] = np.median(disp, 1)
    f['d_eq0'] = (disp==0).sum(1); f['d_eq1'] = (disp==1).sum(1)
    f['d_ge2'] = (disp>=2).sum(1); f['d_ge4'] = (disp>=4).sum(1)
    pm = ((states_array&1)!=(ident&1)).astype(np.float32)
    for i in range(n): f[f'pm_{i}'] = pm[:,i]
    f['pm_sum'] = pm.sum(1); f['pm_pct'] = pm.mean(1); f['pm_x_disp'] = (pm*disp).sum(1)
    sorted_adj = (states_array[:,:-1]<states_array[:,1:]).astype(np.float32)
    for i in range(n-1): f[f's_{i}{i+1}'] = sorted_adj[:,i]
    f['s_sum'] = sorted_adj.sum(1)
    adj_diff = np.abs(np.diff(states_array, axis=1))
    for i in range(n-1): f[f'ad_{i}'] = adj_diff[:,i]
    f['ad_sum'] = adj_diff.sum(1); f['ad_max'] = adj_diff.max(1)
    pair_correct = ((states_array[:,:-1]==ident[:-1])&(states_array[:,1:]==ident[1:])).astype(np.float32)
    f['pair_correct'] = pair_correct.sum(1)
    invs = np.zeros(N, dtype=np.float32)
    for i in range(n): invs += (states_array[:,i:i+1]>states_array[:,i+1:]).sum(axis=1)
    f['inv'] = invs; f['inv_frac'] = invs/(n*(n-1)/2)
    f['inv_parity'] = (invs.astype(np.int64)&1).astype(np.float32)
    descs = np.zeros(N, dtype=np.float32)
    for i in range(n-1): descs += (states_array[:,i]>states_array[:,i+1])
    f['desc'] = descs
    ieu=np.zeros(N,dtype=np.float32); iec=np.zeros(N,dtype=np.float32)
    for idx in range(0,n-1,2): ieu+=(states_array[:,idx]>states_array[:,idx+1]); iec+=((states_array[:,idx]==idx)&(states_array[:,idx+1]==idx+1))
    f['I_unsorted']=ieu; f['I_correct']=iec
    keu=np.zeros(N,dtype=np.float32); kec=np.zeros(N,dtype=np.float32)
    for idx in range(1,n-1,2): keu+=(states_array[:,idx]>states_array[:,idx+1]); kec+=((states_array[:,idx]==idx)&(states_array[:,idx+1]==idx+1))
    f['K_unsorted']=keu; f['K_correct']=kec
    if k+2<n:
        f['S_sorted']=(states_array[:,k]<states_array[:,k+2]).astype(np.float32)
        f['S_diff']=np.abs(states_array[:,k].astype(np.float32)-states_array[:,k+2].astype(np.float32))
        f['S_disp']=np.abs(states_array[:,k]-k)+np.abs(states_array[:,k+2]-(k+2))
        f['S_need_swap']=((states_array[:,k]==k+2)&(states_array[:,k+2]==k)).astype(np.float32)
    f['lb_disp']=disp.max(1)/2.0; f['lb_parity']=pm.sum(1)/2.0
    f['lb_comb']=np.maximum(f['lb_disp'],f['lb_parity'])
    inv_perm=np.argsort(states_array,axis=1)
    for v in range(n): f[f'pos_of_{v}']=inv_perm[:,v].astype(np.float32)
    f['where_0']=np.abs(inv_perm[:,0]-0).astype(np.float32)
    f['where_n1']=np.abs(inv_perm[:,n-1]-(n-1)).astype(np.float32)
    run_lens=np.zeros(N,dtype=np.float32)
    for i in range(N):
        cur=best=0
        for x in (states_array[i]==ident):
            if x: cur+=1; best=max(best,cur)
            else: cur=0
        run_lens[i]=best
    f['run_correct']=run_lens
    sd=states_array.astype(np.float32)-ident.astype(np.float32)
    f['disp_pos']=np.maximum(sd,0).sum(1); f['disp_neg']=np.maximum(-sd,0).sum(1); f['disp_net']=sd.sum(1)
    for i in range(n): f[f'p_{i}']=states_array[:,i].astype(np.float32)
    return pd.DataFrame(f)

def evaluate(y_true, y_pred):
    return {'r2': r2_score(y_true, y_pred), 'rmse': root_mean_squared_error(y_true, y_pred),
            'spearman': stats.spearmanr(y_true, y_pred).statistic}

# =============================================================================
# Resume
# =============================================================================
def load_completed():
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        return set(zip(df['n'], df['config_name']))
    return set()
completed = load_completed()
print(f'Resuming: {len(completed)} completed')

# =============================================================================
# Test cache
# =============================================================================
TEST_CACHE = {}
def get_test_data(n):
    if n not in TEST_CACHE:
        states, labels = generate_random_walks(n, 500, 8*n, K_VAL)
        TEST_CACHE[n] = {'features': extract_features(states, n, K_VAL), 'labels': labels}
    return TEST_CACHE[n]

# =============================================================================
# Wandb
# =============================================================================
wandb.init(project='koltsov3-sweep', name=f'lgb_ext_{time.strftime("%Y%m%d_%H%M%S")}',
           config={'n_values': N_VALUES, 'configs': [c[0] for c in CONFIGS]},
           tags=['lgb_extended'])

# =============================================================================
# Sweep
# =============================================================================
run_count = len(completed); run_start = time.time()

for n in N_VALUES:
    print(f'\n{"#"*60}\n# n = {n}\n{"#"*60}')
    test_data = get_test_data(n)
    print(f'  Test data: {len(test_data["labels"]):,} samples')

    for (name, n_est, leaves, lr, boosting, objective, extra) in CONFIGS:
        if (n, name) in completed:
            print(f'  [{name}] — SKIPPED')
            continue

        print(f'  [{name}] n_est={n_est} leaves={leaves} lr={lr} '
              f'boosting={boosting} obj={objective}', flush=True)
        try:
            walk_length = 8 * n
            states, labels = generate_random_walks(n, 25_000, walk_length, K_VAL)
            if len(labels) > 500_000:
                idx = np.random.choice(len(labels), 500_000, replace=False)
                states, labels = states[idx], labels[idx]

            df = extract_features(states, n, K_VAL)
            df['label'] = labels
            feature_cols = [c for c in df.columns if c != 'label']

            X_tr, X_val, y_tr, y_val = train_test_split(
                df[feature_cols], df['label'], test_size=0.15, random_state=SEED)

            t0 = time.time()
            dtrain = lgb.Dataset(X_tr, y_tr, feature_name=list(feature_cols))
            dval = lgb.Dataset(X_val, y_val, reference=dtrain, feature_name=list(feature_cols))
            evals_result = {}

            params = {
                'objective': 'regression' if objective == 'rmse' else objective,
                'metric': 'rmse',
                'boosting_type': boosting,
                'num_leaves': leaves, 'max_depth': 12,
                'learning_rate': lr, 'n_estimators': n_est,
                'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.01, 'reg_lambda': 0.01,
                'verbose': -1, 'seed': SEED, 'n_jobs': -1,
                **extra,
            }
            # DART requires different early stopping setup
            callbacks = [lgb.record_evaluation(evals_result)]
            if n_est > 1000:
                callbacks.append(lgb.early_stopping(EARLY_STOPPING))

            model = lgb.train(params, dtrain, valid_sets=[dtrain, dval],
                              valid_names=['train', 'val'], callbacks=callbacks)
            fit_time = time.time() - t0

            y_te_pred = model.predict(test_data['features'])
            m_te = evaluate(test_data['labels'], y_te_pred)

            result = {
                'n': n, 'config_name': name,
                'n_estimators': n_est, 'num_leaves': leaves,
                'learning_rate': lr, 'boosting': boosting,
                'objective': objective,
                'test_r2': m_te['r2'], 'test_rmse': m_te['rmse'],
                'test_spearman': m_te['spearman'],
                'fit_time': fit_time,
                'best_iteration': model.best_iteration,
            }
            df_r = pd.DataFrame([result])
            if os.path.exists(RESULTS_CSV):
                df_r.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
            else:
                df_r.to_csv(RESULTS_CSV, index=False)

            run_count += 1
            eta_h = ((time.time()-run_start)/run_count*(total_runs-run_count)/3600)
            print(f"    R²={result['test_r2']:.4f}  RMSE={result['test_rmse']:.2f}  "
                  f"ρ={result['test_spearman']:.4f}  best_iter={result['best_iteration']}  "
                  f"time={fit_time:.1f}s  [{run_count}/{total_runs}] ETA {eta_h:.1f}h")

        except Exception as e:
            print(f'    FAILED: {e}')

print(f'\nDONE — {run_count} runs')
wandb.finish()

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    print(f'\n--- Best per n ---')
    best = df.loc[df.groupby('n')['test_r2'].idxmax()]
    for _, row in best.iterrows():
        print(f'  n={int(row["n"]):2d}: R²={row["test_r2"]:.4f}  ρ={row["test_spearman"]:.4f}  '
              f'config={row["config_name"]}')
