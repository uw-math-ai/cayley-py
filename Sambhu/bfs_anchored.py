# =============================================================================
# LRX CayleyPy — W&B 30-trial runner with BFS-anchored mDQN + residual rollout + beam-aware lambda sweep
# Pipeline: warmup -> BFS exact anchor -> pass1 mDQN with BFS regularization -> pass2 residual calibration -> pass3 beam-aware penalty on residual -> lambda sweep.
#
# v2 target change:
#   Instead of expecting an 8-step greedy rollout to reduce V by 8, use
#   CFG['pass2_beam_aware_expected_progress'] = 4.0. This avoids saturating
#   most beam-aware penalty targets at the cap.
# =============================================================================

import os
import time
import json
import gc
import subprocess
import sys

# Install wandb if missing. Comment this out if your GPU box already has wandb.
try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])
    import wandb

# Option 1: paste your key here.
WANDB_API_KEY = "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"  # <-- paste your W&B API key here, or set env var WANDB_API_KEY

_wandb_key = os.environ.get("WANDB_API_KEY", WANDB_API_KEY)
if _wandb_key:
    wandb.login(key=_wandb_key)
else:
    print("WARNING: WANDB_API_KEY is blank and env var WANDB_API_KEY is not set.")
    print("         If you are already logged in on this machine, wandb.init may still work.")


import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from scipy.stats import spearmanr

# Headless-safe. No graph-generation cells are included in this script.
import matplotlib
matplotlib.use("Agg")

# =============================================================================
# 1. CONFIG — same parameters as notebook
# =============================================================================
CFG = {}

CFG['n_permutations_length']  = 28
n_permutations_length = CFG['n_permutations_length']

CFG['random_walks_type'] = 'non-backtracking-beam'
CFG['n_random_walk_length']  = int(n_permutations_length * (n_permutations_length - 1) / 2)
CFG['n_random_walks_to_generate']  = 10_000
CFG['n_random_walks_steps_back_to_ban']  = 8

CFG['model_type'] = 'MLP'
CFG['list_layers_sizes'] = [128]
CFG['n_epochs'] = 30
CFG['batch_size'] = 1024
CFG['lr'] = 0.001

CFG['n_epochs_dqn'] = 200
CFG['flag_dqn_round'] = False
CFG['n_random_walks_to_generate_dqn'] = 10_000

# -----------------------------------------------------------------------------
# Exact shallow-BFS anchoring / regularization
# -----------------------------------------------------------------------------
# Motivation: random-walk labels are cheap but noisy upper-bound-like labels.
# BFS labels near identity are exact shortest-path distances. This uses the
# existing BFS diagnostic set as a supervised anchor for V(s) during pass 1.
CFG['bfs_anchor_enabled'] = True
# A short pre-pass on exact BFS states after the BFS set is generated and before mDQN.
CFG['bfs_anchor_pretrain_epochs'] = 5
CFG['bfs_anchor_pretrain_batches_per_epoch'] = 64
# During every mDQN epoch, add this many extra exact-BFS supervised update steps.
CFG['bfs_anchor_batches_per_mdqn_epoch'] = 8
CFG['bfs_anchor_batch_size'] = 4096
CFG['bfs_anchor_loss_weight'] = 0.10
# Optional: limit BFS anchor samples if you later generate a much larger BFS set.
# None means use all BFS states returned by the BFS routine.
CFG['bfs_anchor_max_states'] = None

CFG['pass2_enabled']                = True
CFG['pass2_n_epochs']               = 50
CFG['pass2_n_walks']                = 10_000
CFG['pass2_walk_length']            = int(n_permutations_length * (n_permutations_length - 1) / 2)
CFG['pass2_rollout_length']         = 8
CFG['pass2_use_rollout_correction'] = True
CFG['pass2_rollout_mix_alpha']      = 0.9
# Residual pass-2: correction is bounded so pass 2 cannot overwrite pass 1.
CFG['pass2_residual_max_correction'] = 2.0
# Residual pass-2 margin: ignore small rollout-vs-Bellman gaps before correction.
CFG['pass2_residual_margin'] = 0.25
# Beam-aware value correction. This trains pass 2 on actual pass-1 beam-frontier
# states instead of random-walk states. It learns a bounded penalty for beam states
# whose short rollout under V1 does not make enough progress.
CFG['pass2_mode'] = 'bfs_anchor_residual_plus_beam_aware_sweep'
CFG['pass2_beam_aware_epochs'] = 50
CFG['pass2_beam_aware_rollout_length'] = 8
CFG['pass2_beam_aware_expected_progress'] = 4.0
CFG['pass2_beam_aware_beta'] = 1.0
CFG['pass2_beam_aware_margin'] = 0.25
CFG['pass2_beam_aware_max_penalty'] = 1.0
# Final score: V_final(s) = V_residual_calibrated(s) + lambda * beam_penalty(s)
CFG['pass3_beam_aware_score_lambda'] = 0.25
# Inference-only sweep: train the Pass 3 penalty once, then run beam search with each lambda.
CFG['pass3_beam_aware_score_lambdas'] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
CFG['pass2_beam_aware_max_states_per_snapshot'] = 20_000
CFG['pass2_beam_aware_min_step'] = 21

CFG['beam_search_torch'] = True
CFG['beam_search_Fironov'] = False
CFG['beam_width']  = 2**16
CFG['n_steps_limit']  = 4 * n_permutations_length**2
CFG['alpha_previous_cost_accumulation']  = 0
CFG['beam_search_models_or_heuristics'] = 'model_torch'
CFG['ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted'] = False
CFG['n_beam_search_steps_back_to_ban'] = 32

CFG['solve_random_or_longest_state'] = 'solve_LRX_longest'

with open('CFG.json', 'w') as json_file:
    json.dump(CFG, json_file)

print(CFG)
for k in CFG:
    print(k, ':', CFG[k])

WANDB_PROJECT = "lrx-cayleypy-rl-bfs-anchored-full"
WANDB_GROUP = "n28_bfs_anchor_residual_beamaware_lambda_sweep_30trials_v1"


# =============================================================================
# 2. UTILS
# =============================================================================
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    sync_cuda()
    return time.time()


def elapsed(t0):
    sync_cuda()
    return time.time() - t0


def safe_float(x):
    if x is None:
        return float('nan')
    try:
        if np.isnan(x):
            return float('nan')
    except Exception:
        pass
    return float(x)


def spearman_value(x, y):
    """Return Spearman correlation across SciPy versions.

    New SciPy exposes .statistic; older SciPy exposes .correlation;
    very old versions may return a tuple-like object.
    """
    res = spearmanr(x, y)
    if hasattr(res, "statistic"):
        return float(res.statistic)
    if hasattr(res, "correlation"):
        return float(res.correlation)
    return float(res[0])


def get_LRX_moves(n):
    L = np.array(list(np.arange(1, n)) + [0])
    R = np.array([n - 1] + list(np.arange(n - 1)))
    X = np.array([1, 0] + list(np.arange(2, n)))
    return L, R, X


def get_neighbors(states, moves):
    return torch.gather(
        states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
        2,
        moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1))
    )


def get_unique_elements_first_idx(tensor, stable=True):
    sorted_tensor, indices = torch.sort(tensor, stable=stable)
    unique_mask = torch.cat((torch.tensor([True], device=tensor.device), sorted_tensor[1:] != sorted_tensor[:-1]))
    return indices[unique_mask], sorted_tensor[unique_mask]


def get_unique_states(states, vec_hasher):
    hashed = torch.sum(vec_hasher * states, dim=1)
    hashed_sorted, idx = torch.sort(hashed)
    mask = torch.concat((torch.tensor([True], device=states.device), (hashed_sorted[1:] - hashed_sorted[:-1]) != 0))
    return states[idx][mask]


class Net(nn.Module):
    def __init__(self, input_size, hidden_dims, num_classes_for_one_hot):
        super(Net, self).__init__()
        self.num_classes_for_one_hot = num_classes_for_one_hot
        self.input_layer_size_for_one_hot = input_size * num_classes_for_one_hot

        layers = []
        in_features = self.input_layer_size_for_one_hot
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes_for_one_hot).float().flatten(start_dim=-2)
        return self.layers(x)


def random_walks(generators, n_random_walk_length, n_random_walks_to_generate, state_rw_start='01234...',
                 n_random_walks_steps_back_to_ban=0, random_walks_type='simple',
                 device='Auto', dtype='Auto', vec_hasher='Auto', verbose=0):
    if random_walks_type == 'non-backtracking-beam':
        return random_walks_nbt(generators, n_random_walk_length, n_random_walks_to_generate,
                                state_rw_start=state_rw_start,
                                n_random_walks_steps_back_to_ban=n_random_walks_steps_back_to_ban,
                                random_walks_type=random_walks_type,
                                device=device, dtype=dtype, vec_hasher=vec_hasher, verbose=verbose)
    return random_walks_simple(generators, n_random_walk_length, n_random_walks_to_generate,
                               state_rw_start=state_rw_start, device=device, dtype=dtype, verbose=verbose)


def random_walks_simple(generators, n_random_walk_length, n_random_walks_to_generate,
                        state_rw_start='01234...', device='Auto', dtype='Auto', verbose=0):
    if device == 'Auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(generators, list):
        list_generators = generators
    elif isinstance(generators, tuple):
        list_generators = list(generators)
    elif isinstance(generators, torch.Tensor):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    elif isinstance(generators, np.ndarray):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    else:
        raise ValueError('Unsupported format for generators ' + str(type(generators)))

    state_size = len(list_generators[0])
    all_moves = torch.tensor(list_generators, device=device, dtype=torch.int64)
    n_generators = len(list_generators)

    if dtype == 'Auto':
        dtype = torch.uint8 if state_size <= 256 else torch.uint16

    if isinstance(state_rw_start, str) and state_rw_start == '01234...':
        state_rw_start = torch.arange(state_size, device=device, dtype=dtype).reshape(-1, state_size)
    elif isinstance(state_rw_start, torch.Tensor):
        state_rw_start = state_rw_start.to(device).to(dtype).reshape(-1, state_size)
    else:
        state_rw_start = torch.tensor(state_rw_start, device=device, dtype=dtype).reshape(-1, state_size)

    array_of_states = state_rw_start.view(1, state_size).expand(n_random_walks_to_generate, state_size).clone()
    X = torch.zeros((n_random_walks_to_generate) * n_random_walk_length, state_size, device=device, dtype=dtype)
    y = torch.zeros((n_random_walks_to_generate) * n_random_walk_length, device=device, dtype=torch.uint32)
    X[:n_random_walks_to_generate, :] = array_of_states
    y[:n_random_walks_to_generate] = 0

    row_indices = np.arange(array_of_states.shape[0])[:, np.newaxis]
    for i_step in range(1, n_random_walk_length):
        y[i_step * n_random_walks_to_generate:(i_step + 1) * n_random_walks_to_generate] = i_step
        IX_moves = np.random.randint(0, n_generators, size=n_random_walks_to_generate, dtype=int)
        new_array_of_states = array_of_states[row_indices, all_moves[IX_moves, :]]
        array_of_states = new_array_of_states
        X[i_step * n_random_walks_to_generate:(i_step + 1) * n_random_walks_to_generate, :] = new_array_of_states
    return X, y


def random_walks_nbt(generators, n_random_walk_length, n_random_walks_to_generate, state_rw_start='01234...',
                     n_random_walks_steps_back_to_ban=0, random_walks_type='non-backtracking-beam',
                     device='Auto', dtype='Auto', vec_hasher='Auto', verbose=0):
    t0 = time.time()
    if device == 'Auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(generators, list):
        list_generators = generators
    elif isinstance(generators, tuple):
        list_generators = list(generators)
    elif isinstance(generators, torch.Tensor):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    elif isinstance(generators, np.ndarray):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    else:
        raise ValueError('Unsupported format for generators ' + str(type(generators)))

    state_size = len(list_generators[0])
    n_generators = len(list_generators)

    if dtype == 'Auto':
        dtype = torch.uint8 if state_size <= 256 else torch.uint16

    if isinstance(state_rw_start, str) and ((state_rw_start == '01234...') or (state_rw_start == 'Auto')):
        state_rw_start = torch.arange(state_size, device=device, dtype=dtype).reshape(-1, state_size)
    elif isinstance(state_rw_start, torch.Tensor):
        state_rw_start = state_rw_start.to(device).to(dtype).reshape(-1, state_size)
    else:
        state_rw_start = torch.tensor(state_rw_start, device=device, dtype=dtype).reshape(-1, state_size)

    tensor_generators = torch.tensor(list_generators, device=device, dtype=torch.int64)

    max_int = int(2**62)
    dtype_for_hash = torch.int64
    if isinstance(vec_hasher, str) and vec_hasher == 'Auto':
        vec_hasher = torch.randint(-max_int, max_int + 1, size=(state_size,), device=device, dtype=dtype_for_hash)

    array_current_states = state_rw_start.view(1, state_size).expand(n_random_walks_to_generate, state_size).clone()

    X = torch.zeros((n_random_walks_to_generate) * n_random_walk_length, state_size, device=device, dtype=dtype)
    y = torch.zeros((n_random_walks_to_generate) * n_random_walk_length, device=device, dtype=torch.uint32)
    X[:n_random_walks_to_generate, :] = array_current_states
    y[:n_random_walks_to_generate] = 0

    if n_random_walks_steps_back_to_ban > 0:
        hash_initial_state = torch.sum(state_rw_start.view(-1, state_size) * vec_hasher, dim=1)
        vec_hashes_current = hash_initial_state.expand(n_random_walks_to_generate * n_generators,
                                                       n_random_walks_steps_back_to_ban).clone()
        i_cyclic_index_for_hash_storage = 0

    i_step_corrected = 0
    for i_step in range(1, n_random_walk_length):
        t_full_step = time.time()
        array_new_states = get_neighbors(array_current_states, tensor_generators).flatten(end_dim=1)
        vec_hashes_new = torch.sum(array_new_states * vec_hasher, dim=1)

        if n_random_walks_steps_back_to_ban > 0:
            mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
            mask_new_sum = mask_new.sum().item()
            if mask_new_sum >= n_random_walks_to_generate:
                array_new_states = array_new_states[mask_new, :]
                i_step_corrected += 1
            else:
                if mask_new_sum > 0:
                    i_tmp0 = int(np.ceil(n_random_walks_to_generate / mask_new_sum))
                    array_new_states = array_new_states[mask_new, :].repeat(i_tmp0, 1)[:n_random_walks_to_generate, :]
                    i_step_corrected += 1
                else:
                    array_new_states = array_current_states

        perm = torch.randperm(array_new_states.size(0), device=device)
        array_current_states = array_new_states[perm][:n_random_walks_to_generate]

        y[i_step * n_random_walks_to_generate:(i_step + 1) * n_random_walks_to_generate] = i_step_corrected
        X[i_step * n_random_walks_to_generate:(i_step + 1) * n_random_walks_to_generate, :] = array_current_states

        if n_random_walks_steps_back_to_ban > 0:
            i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % n_random_walks_steps_back_to_ban
            vec_hashes_current[:, i_cyclic_index_for_hash_storage] = vec_hashes_new

        if verbose >= 10:
            print(i_step, 'i_step', 'array_current_states.shape:', array_current_states.shape,
                  'Time %.3f' % (time.time() - t0), 't_full_step %.3f' % (time.time() - t_full_step))

    return X, y


def bfs_growth_permutations_torch_simple(generators, center_states=None, radius_max=10000,
                                         stop_threshold_total_states=np.inf, device=None, dtype=None,
                                         vec_hasher=None, flag_return_all_hashes=True,
                                         flag_return_all_states=True, flag_return_list_distances=True,
                                         verbose=0):
    t0 = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(generators, list):
        list_generators = generators
    elif isinstance(generators, torch.Tensor):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    elif isinstance(generators, np.ndarray):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    else:
        raise ValueError('Unsupported format for generators')

    state_size = len(list_generators[0])
    tensor_all_generators = torch.tensor(list_generators, device=device, dtype=torch.int64)

    if dtype is None:
        if isinstance(center_states, torch.Tensor):
            dtype = center_states.dtype
        else:
            dtype = torch.uint8 if state_size <= 256 else torch.uint16

    if center_states is None:
        center_states = torch.arange(state_size, device=device, dtype=dtype).reshape(1, state_size)
    if not isinstance(center_states, torch.Tensor):
        center_states = torch.tensor(center_states)
    center_states = center_states.reshape(-1, state_size).to(device)

    if vec_hasher is None:
        max_int = int(2**62)
        vec_hasher = torch.randint(-max_int, max_int, size=(state_size,), device=device, dtype=torch.int64)
    if not isinstance(vec_hasher, torch.Tensor):
        vec_hasher = torch.tensor(vec_hasher, device=device, dtype=torch.int64)

    dict_growth = {0: center_states.shape[0]}
    dict_additional_data = {'vec_hasher': vec_hasher}
    list_distances = [0] * center_states.shape[0]
    array_states_all = center_states.clone()

    array_states_achieved_previous_step = center_states.clone()
    vec_hashes_all = torch.sum(array_states_achieved_previous_step * vec_hasher, dim=1)
    vec_hashes_previous_step = vec_hashes_all.clone()
    vec_hashes_pre_previous_step = vec_hashes_all.clone()
    vec_hashes_current = vec_hashes_all.clone()

    for i_distance in range(1, radius_max + 1):
        t_all = time.time()
        new_states_candidates = get_neighbors(array_states_achieved_previous_step, tensor_all_generators).flatten(end_dim=1)
        vec_hashes_new_all_gens = torch.sum(new_states_candidates * vec_hasher, dim=1)
        unique_idx, unique_hashes = get_unique_elements_first_idx(vec_hashes_new_all_gens, stable=False)
        mask_new = ~torch.isin(unique_hashes, vec_hashes_current, assume_unique=True)

        n_new_states = mask_new.sum().item()
        if n_new_states == 0:
            break

        array_new_states = new_states_candidates[unique_idx, :][mask_new, :]
        array_states_achieved_previous_step = array_new_states

        vec_hashes_new_all_gens = unique_hashes[mask_new]
        vec_hashes_pre_previous_step = vec_hashes_previous_step
        vec_hashes_previous_step = vec_hashes_new_all_gens
        vec_hashes_current = torch.cat([vec_hashes_pre_previous_step, vec_hashes_previous_step], dim=0)

        dict_growth[i_distance] = n_new_states
        if flag_return_list_distances:
            list_distances += [i_distance] * int(array_new_states.shape[0])
        if flag_return_all_hashes:
            vec_hashes_all = torch.cat([vec_hashes_all, vec_hashes_new_all_gens], dim=0)
        if flag_return_all_states:
            array_states_all = torch.cat([array_states_all, array_new_states], dim=0)

        if verbose >= 10:
            print(i_distance, 'i_distance', n_new_states, 'new states',
                  't_all %.3f' % (time.time() - t_all), 'Cummulat Time %.3f' % (time.time() - t0))

        if np.sum(list(dict_growth.values())) >= stop_threshold_total_states:
            break

    if flag_return_all_hashes:
        dict_additional_data['vec_hashes_all'] = vec_hashes_all
    if flag_return_all_states:
        dict_additional_data['array_states_all'] = array_states_all
    if flag_return_list_distances:
        dict_additional_data['list_distances'] = list_distances
    return dict_growth, dict_additional_data


def greedy_rollout_targets(states, guide_model, generators_t, rollout_length, bs=4096):
    N = states.shape[0]
    state_sz = states.shape[1]
    G = generators_t.shape[0]
    dev = states.device

    cur = states.clone()
    L_taken = torch.full((N,), float(rollout_length), device=dev)

    guide_model.eval()
    with torch.no_grad():
        for _ in range(rollout_length):
            nb = get_neighbors(cur, generators_t)
            nb_flat = nb.reshape(-1, state_sz)
            V_flat = torch.zeros(nb_flat.shape[0], device=dev)
            for i0 in range(0, nb_flat.shape[0], bs):
                i1 = min(i0 + bs, nb_flat.shape[0])
                out = guide_model(nb_flat[i0:i1])
                if isinstance(out, tuple):
                    out = out[0]
                V_flat[i0:i1] = out.view(-1)
            V_nb = V_flat.reshape(N, G)
            best = torch.argmin(V_nb, dim=1)
            cur = nb[torch.arange(N, device=dev), best, :]

        V_final = torch.zeros(N, device=dev)
        for i0 in range(0, N, bs):
            i1 = min(i0 + bs, N)
            out = guide_model(cur[i0:i1])
            if isinstance(out, tuple):
                out = out[0]
            V_final[i0:i1] = out.view(-1)

    return L_taken + V_final


def run_error_analysis(stage_name, model, states_snap, true_d_snap, tensor_generators, vec_hasher, device, trial_idx):
    print("=" * 70)
    print(f"{stage_name.upper()} ERROR ANALYSIS")
    print("=" * 70)
    t0 = now()

    true_d_snap = np.array(true_d_snap)
    N_snap = states_snap.shape[0]
    bs_snap = 4096

    model.eval()
    V_snap = torch.zeros(N_snap, device=device)
    with torch.no_grad():
        for i in range(0, N_snap, bs_snap):
            j = min(i + bs_snap, N_snap)
            out = model(states_snap[i:j])
            if isinstance(out, tuple):
                out = out[0]
            V_snap[i:j] = out.view(-1)
    V_snap_np = V_snap.detach().cpu().numpy()

    neigb_snap = get_neighbors(states_snap, tensor_generators)
    state_d_snap = neigb_snap.shape[-1]
    V_nb_snap = torch.zeros(N_snap, tensor_generators.shape[0], device=device)
    with torch.no_grad():
        for i in range(0, N_snap, bs_snap):
            j = min(i + bs_snap, N_snap)
            out = model(neigb_snap[i:j].reshape(-1, state_d_snap))
            if isinstance(out, tuple):
                out = out[0]
            V_nb_snap[i:j] = out.view(-1).reshape(-1, tensor_generators.shape[0])
    V_nb_snap_np = V_nb_snap.detach().cpu().numpy()

    hashes_nb_snap = (neigb_snap.reshape(-1, state_d_snap) * vec_hasher).sum(dim=1)
    hashes_all_snap = (states_snap * vec_hasher).sum(dim=1)
    order_snap = torch.argsort(hashes_all_snap)
    hashes_sorted_snap = hashes_all_snap[order_snap]
    true_d_sorted_snap = torch.from_numpy(true_d_snap).to(device)[order_snap]
    idx_snap = torch.searchsorted(hashes_sorted_snap, hashes_nb_snap).clamp(max=N_snap - 1)
    true_d_nb_snap = true_d_sorted_snap[idx_snap].reshape(N_snap, tensor_generators.shape[0]).detach().cpu().numpy()
    valid_snap = (hashes_sorted_snap[idx_snap] == hashes_nb_snap).reshape(N_snap, tensor_generators.shape[0]).detach().cpu().numpy()
    all_valid_snap = valid_snap.all(axis=1)
    match_snap = (true_d_nb_snap.argmin(axis=1) == V_nb_snap_np.argmin(axis=1)) & all_valid_snap

    err = V_snap_np - true_d_snap
    spearman_all = spearman_value(true_d_snap, V_snap_np)
    if np.std(true_d_snap) > 0 and np.std(V_snap_np) > 0:
        pearson_all = float(np.corrcoef(true_d_snap, V_snap_np)[0, 1])
    else:
        pearson_all = float('nan')
    argmin_match = match_snap.mean()
    mean_err = err.mean()
    std_err = err.std()
    mae = np.mean(np.abs(err))

    print(f"{stage_name}: Spearman={spearman_all:.4f} Pearson={pearson_all:.4f} ArgminMatch={argmin_match:.4f}")
    print(f"{stage_name}: MeanErr={mean_err:.4f} StdErr={std_err:.4f} MAE={mae:.4f}")

    # Same style as notebook: bucket error by true distance, but logged as a W&B table instead of plotted.
    df_snap = pd.DataFrame({"true": true_d_snap, "pred": V_snap_np, "err": err})
    buckets_snap = pd.cut(df_snap["true"], bins=10)
    agg_snap = df_snap.groupby(buckets_snap, observed=True)["err"].agg(["mean", "std", "count"]).reset_index()
    agg_snap["bucket"] = agg_snap["true"].astype(str)
    bucket_table = wandb.Table(dataframe=agg_snap[["bucket", "mean", "std", "count"]])

    # Bucket correlations used in the notebook around low/mid distances.
    bucket_corrs = {}
    for low, high in [(0, 10), (10, 30)]:
        mask = (true_d_snap >= low) & (true_d_snap < high)
        n_bucket = int(mask.sum())
        sp = float('nan')
        pe = float('nan')

        if n_bucket >= 10:
            true_bucket = true_d_snap[mask]
            pred_bucket = V_snap_np[mask]
            if np.std(true_bucket) > 0 and np.std(pred_bucket) > 0:
                sp = spearman_value(true_bucket, pred_bucket)
                pe = float(np.corrcoef(true_bucket, pred_bucket)[0, 1])

        bucket_corrs[f"{stage_name}/spearman_d_{low}_{high}"] = safe_float(sp)
        bucket_corrs[f"{stage_name}/pearson_d_{low}_{high}"] = safe_float(pe)
        bucket_corrs[f"{stage_name}/n_d_{low}_{high}"] = n_bucket

    analysis_time = elapsed(t0)
    wandb.log({
        "trial_idx": trial_idx,
        f"{stage_name}/error_analysis_time_sec": analysis_time,
        f"{stage_name}/spearman_all": safe_float(spearman_all),
        f"{stage_name}/pearson_all": safe_float(pearson_all),
        f"{stage_name}/argmin_match": safe_float(argmin_match),
        f"{stage_name}/mean_error": safe_float(mean_err),
        f"{stage_name}/std_error": safe_float(std_err),
        f"{stage_name}/mae": safe_float(mae),
        f"{stage_name}/error_by_distance_table": bucket_table,
        **bucket_corrs,
    })

    return {
        "time_sec": analysis_time,
        "spearman_all": safe_float(spearman_all),
        "pearson_all": safe_float(pearson_all),
        "argmin_match": safe_float(argmin_match),
        "mean_error": safe_float(mean_err),
        "std_error": safe_float(std_err),
        "mae": safe_float(mae),
    }


def _predict_model_values_np(model, states, device, batch_size=4096):
    model.eval()
    vals = []
    n_states = int(states.shape[0])
    with torch.no_grad():
        for i in range(0, n_states, batch_size):
            j = min(i + batch_size, n_states)
            x = states[i:j].to(device, non_blocking=True)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            vals.append(out.view(-1).detach().cpu())
    return torch.cat(vals).numpy() if vals else np.array([], dtype=np.float32)


def _maybe_sample_states(states, max_states=None, seed=0):
    n_states = int(states.shape[0])
    if max_states is None or n_states <= max_states:
        return states, None
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    idx = torch.randperm(n_states, generator=g)[:int(max_states)]
    return states[idx], idx.numpy()


def log_value_delta_stats(prefix, base_model, new_model, states, trial_idx, device,
                          true_d=None, batch_size=4096, max_states=None, seed=0):
    """
    Log how much a new model changes values relative to the pass-1 base model.
    This diagnoses whether pass 2 is only nudging pass 1 or changing enough
    values/rankings to affect beam pruning.
    """
    t0 = now()
    states_cpu = states.detach().cpu() if isinstance(states, torch.Tensor) and states.is_cuda else states
    states_eval, idx_np = _maybe_sample_states(states_cpu, max_states=max_states, seed=seed)
    if true_d is not None:
        true_arr = np.asarray(true_d)
        if idx_np is not None:
            true_arr = true_arr[idx_np]
    else:
        true_arr = None

    v_base = _predict_model_values_np(base_model, states_eval, device, batch_size=batch_size)
    v_new = _predict_model_values_np(new_model, states_eval, device, batch_size=batch_size)
    delta = v_new - v_base

    metrics = {
        f"{prefix}/n_states": int(states_eval.shape[0]),
        f"{prefix}/time_sec": elapsed(t0),
        f"{prefix}/base_value_mean": safe_float(np.mean(v_base)),
        f"{prefix}/new_value_mean": safe_float(np.mean(v_new)),
        f"{prefix}/delta_mean": safe_float(np.mean(delta)),
        f"{prefix}/delta_std": safe_float(np.std(delta)),
        f"{prefix}/delta_median": safe_float(np.median(delta)),
        f"{prefix}/delta_p10": safe_float(np.percentile(delta, 10)),
        f"{prefix}/delta_p25": safe_float(np.percentile(delta, 25)),
        f"{prefix}/delta_p75": safe_float(np.percentile(delta, 75)),
        f"{prefix}/delta_p90": safe_float(np.percentile(delta, 90)),
        f"{prefix}/delta_p95": safe_float(np.percentile(delta, 95)),
        f"{prefix}/delta_p99": safe_float(np.percentile(delta, 99)),
        f"{prefix}/delta_min": safe_float(np.min(delta)),
        f"{prefix}/delta_max": safe_float(np.max(delta)),
        f"{prefix}/frac_delta_gt_0_25": safe_float(np.mean(delta > 0.25)),
        f"{prefix}/frac_delta_gt_0_50": safe_float(np.mean(delta > 0.50)),
        f"{prefix}/frac_delta_gt_1_00": safe_float(np.mean(delta > 1.00)),
        f"{prefix}/frac_delta_gt_1_50": safe_float(np.mean(delta > 1.50)),
        f"{prefix}/frac_delta_gt_1_90": safe_float(np.mean(delta > 1.90)),
        f"{prefix}/frac_delta_lt_minus_0_25": safe_float(np.mean(delta < -0.25)),
    }

    if true_arr is not None:
        base_err = v_base - true_arr
        new_err = v_new - true_arr
        metrics.update({
            f"{prefix}/base_mean_error": safe_float(np.mean(base_err)),
            f"{prefix}/new_mean_error": safe_float(np.mean(new_err)),
            f"{prefix}/base_mae": safe_float(np.mean(np.abs(base_err))),
            f"{prefix}/new_mae": safe_float(np.mean(np.abs(new_err))),
            f"{prefix}/base_spearman": safe_float(spearman_value(true_arr, v_base)),
            f"{prefix}/new_spearman": safe_float(spearman_value(true_arr, v_new)),
        })
        df_delta = pd.DataFrame({
            "true": true_arr,
            "base": v_base,
            "new": v_new,
            "delta": delta,
            "base_err": base_err,
            "new_err": new_err,
        })
        buckets = pd.cut(df_delta["true"], bins=10)
        agg = df_delta.groupby(buckets, observed=True).agg(
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            delta_p90=("delta", lambda x: np.percentile(x, 90)),
            base_err_mean=("base_err", "mean"),
            new_err_mean=("new_err", "mean"),
            count=("delta", "count"),
        ).reset_index()
        agg["bucket"] = agg["true"].astype(str)
        metrics[f"{prefix}/delta_by_true_distance_table"] = wandb.Table(
            dataframe=agg[["bucket", "delta_mean", "delta_std", "delta_p90", "base_err_mean", "new_err_mean", "count"]]
        )

    wandb.log({"trial_idx": trial_idx, **metrics})
    print(f"Logged value deltas for {prefix}: mean={np.mean(delta):.4f}, p95={np.percentile(delta, 95):.4f}, max={np.max(delta):.4f}")
    return metrics


def log_beam_snapshot_value_deltas(base_model, new_model, snapshots, trial_idx, device,
                                   batch_size=4096, max_states_per_snapshot=65536):
    if not snapshots:
        wandb.log({"trial_idx": trial_idx, "value_delta_beam_snapshots/n_snapshots": 0})
        return []
    wandb.log({"trial_idx": trial_idx, "value_delta_beam_snapshots/n_snapshots": len(snapshots)})
    all_metrics = []
    for snap in snapshots:
        step = int(snap["step"])
        kind = str(snap["kind"])
        states = snap["states"]
        safe_kind = kind.replace("/", "_").replace(" ", "_")
        prefix = f"value_delta_beam_snapshots/step_{step:04d}/{safe_kind}"
        metrics = log_value_delta_stats(
            prefix,
            base_model,
            new_model,
            states,
            trial_idx,
            device,
            true_d=None,
            batch_size=batch_size,
            max_states=max_states_per_snapshot,
            seed=trial_idx + step,
        )
        wandb.log({
            "trial_idx": trial_idx,
            f"{prefix}/snapshot_n_total": int(snap.get("n_total", states.shape[0])),
        })
        all_metrics.append(metrics)
    return all_metrics


def run_beam_search(stage_name, model, state_start, state_destination, list_generators, tensor_generators,
                    vec_hasher, dtype, device, CFG, trial_idx,
                    snapshot_store=None, snapshot_every=100, snapshot_max_states=65536):
    print("\n--- Starting Beam Search:", stage_name, "---")
    t0_global = now()

    n_generators = len(list_generators)
    state_size = state_start.shape[0]
    beam_width = CFG['beam_width']
    n_steps_limit = CFG['n_steps_limit']
    batch_size = CFG['batch_size']
    n_steps_back_to_ban = CFG['n_beam_search_steps_back_to_ban']
    beam_search_models_or_heuristics = CFG['beam_search_models_or_heuristics']
    ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted = CFG['ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted']

    def should_snapshot(step):
        if snapshot_store is None:
            return False
        return (step in (1, 11, 21)) or (snapshot_every > 0 and step % snapshot_every == 0)

    def store_snapshot(step, kind, states):
        if snapshot_store is None:
            return
        if states is None or states.numel() == 0:
            return
        n_take = min(int(snapshot_max_states), int(states.shape[0]))
        snapshot_store.append({
            "stage": stage_name,
            "step": int(step),
            "kind": str(kind),
            "n_total": int(states.shape[0]),
            "states": states[:n_take].detach().cpu().clone(),
        })

    X_loc = np.array([1, 0] + list(np.arange(2, CFG['n_permutations_length'])))
    i_position_X_in_list_generators = -1
    for k in range(len(list_generators)):
        if np.all(list_generators[k] == X_loc):
            i_position_X_in_list_generators = k
            break

    flag_found_destination = False
    array_beam_states = state_start.view(1, state_size).clone().to(dtype).to(device)

    if n_steps_back_to_ban > 0:
        hash_initial_state = torch.sum(state_start.view(-1, state_size) * vec_hasher, dim=1)
        vec_hashes_current = hash_initial_state.expand(beam_width * n_generators, n_steps_back_to_ban).clone()
        i_cyclic_index_for_hash_storage = 0

    last_step = 0
    for i_step in range(1, n_steps_limit + 1):
        last_step = i_step
        if not ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted:
            array_new_states = get_neighbors(array_beam_states, tensor_generators).flatten(end_dim=1)
        else:
            array_new_states = torch.empty((0, array_beam_states.shape[1]), device=device, dtype=dtype)
            row_indices = np.arange(array_beam_states.shape[0])[:, np.newaxis]
            for ii1, move in enumerate(list_generators):
                if ii1 != i_position_X_in_list_generators:
                    array_states_tmp = array_beam_states[row_indices, move]
                else:
                    mask_X_condtion = array_beam_states[:, 0] > array_beam_states[:, 1]
                    row_indices_tmp = np.arange(mask_X_condtion.sum().item())[:, np.newaxis]
                    array_states_tmp = array_beam_states[mask_X_condtion][row_indices_tmp, move]
                array_new_states = torch.cat([array_new_states, array_states_tmp], dim=0)

        array_new_states = get_unique_states(array_new_states, vec_hasher)

        vec_tmp = torch.all(array_new_states == state_destination, axis=1)
        flag_found_destination = torch.any(vec_tmp).item()
        if flag_found_destination:
            print('Found destination state. ', 'i_step:', i_step, ' n_ways:', (vec_tmp).sum().item())
            store_snapshot(i_step, "found_destination_candidates", array_new_states)
            break

        if n_steps_back_to_ban > 0:
            vec_hashes_new = torch.sum(array_new_states * vec_hasher, dim=1)
            mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
            mask_new_sum = mask_new.sum().item()
            if mask_new_sum > 0:
                array_new_states = array_new_states[mask_new, :]
            else:
                flag_found_destination = False
                print('Cannot find new states. i_step:', i_step)
                break
            i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % n_steps_back_to_ban
            i_tmp = len(vec_hashes_new)
            vec_hashes_current[:i_tmp, i_cyclic_index_for_hash_storage] = vec_hashes_new

        if array_new_states.shape[0] > beam_width:
            if beam_search_models_or_heuristics == 'model_torch':
                model.eval()
                with torch.no_grad():
                    n_states_all = array_new_states.shape[0]
                    q_value = torch.zeros(n_states_all, device=device)
                    for i_start_batch in range(0, n_states_all, batch_size):
                        i_end_batch = min(i_start_batch + batch_size, n_states_all)
                        q_value[i_start_batch:i_end_batch] = model(array_new_states[i_start_batch:i_end_batch, :]).view(-1)
                idx_sorted = torch.argsort(q_value)
                idx = idx_sorted[:beam_width]

                if should_snapshot(i_step):
                    store_snapshot(i_step, "selected_top_by_current_model", array_new_states[idx])
                    if idx_sorted.shape[0] > beam_width:
                        idx_pruned = idx_sorted[beam_width:beam_width + snapshot_max_states]
                        store_snapshot(i_step, "just_pruned_by_current_model", array_new_states[idx_pruned])

                array_beam_states = array_new_states[idx, :]
            elif beam_search_models_or_heuristics == 'Hamming':
                q_value = torch.sum((array_new_states - state_destination) != 0, axis=1)
                idx_sorted = torch.argsort(q_value)
                idx = idx_sorted[:beam_width]

                if should_snapshot(i_step):
                    store_snapshot(i_step, "selected_top_by_current_model", array_new_states[idx])
                    if idx_sorted.shape[0] > beam_width:
                        idx_pruned = idx_sorted[beam_width:beam_width + snapshot_max_states]
                        store_snapshot(i_step, "just_pruned_by_current_model", array_new_states[idx_pruned])

                array_beam_states = array_new_states[idx, :]
            else:
                raise Exception("Unrecognized models_or_heuristics: " + str(beam_search_models_or_heuristics))
        else:
            array_beam_states = array_new_states
            if should_snapshot(i_step):
                store_snapshot(i_step, "selected_all_unsaturated", array_beam_states)

        if (i_step - 1) % 10 == 0:
            print('Step:', i_step, '| Beam states:', array_beam_states.shape[0], '| Time:', f'{elapsed(t0_global):.2f}s')

    beam_time = elapsed(t0_global)
    print('beam_width:', beam_width)
    print('n=', len(list_generators[0]))
    print('n(n-1)/2=', int(CFG['n_permutations_length'] * (CFG['n_permutations_length'] - 1) / 2))
    print('Found Path Length:', last_step, 'flag_found_destination:', flag_found_destination)

    wandb.log({
        "trial_idx": trial_idx,
        f"{stage_name}/path_length": int(last_step),
        f"{stage_name}/found_destination": bool(flag_found_destination),
        f"{stage_name}/beam_time_sec": beam_time,
    })
    return int(last_step), bool(flag_found_destination), beam_time


def train_warmup(model, optimizer, criterion, list_generators, state_destination, vec_hasher, dtype,
                 device, CFG, trial_idx):
    print("\n--- Warmup Training ---")
    t_total = now()
    losses = []
    for epoch in range(CFG['n_epochs']):
        t_epoch = now()
        t0 = now()
        X_train, y_train = random_walks(
            list_generators,
            n_random_walk_length=CFG['n_random_walk_length'],
            n_random_walks_to_generate=CFG['n_random_walks_to_generate'],
            n_random_walks_steps_back_to_ban=CFG['n_random_walks_steps_back_to_ban'],
            random_walks_type=CFG['random_walks_type'],
            state_rw_start=state_destination,
            device=device,
            dtype=dtype,
            vec_hasher=vec_hasher,
        )
        t_rw = elapsed(t0)

        y_train = y_train.float()
        indices = torch.randperm(X_train.shape[0], device=device)
        X_train = X_train[indices]
        y_train = y_train[indices]

        t0 = now()
        model.train()
        n_states_all = X_train.shape[0]
        cc = 0
        train_loss = 0.0
        for i_start_batch in range(0, n_states_all, CFG['batch_size']):
            i_end_batch = min(i_start_batch + CFG['batch_size'], n_states_all)
            outputs = model(X_train[i_start_batch:i_end_batch])
            loss = criterion(outputs.squeeze(), y_train[i_start_batch:i_end_batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            cc += 1
        train_loss /= cc
        losses.append(train_loss)
        t_train = elapsed(t0)
        t_epoch_sec = elapsed(t_epoch)

        print('warmup epoch:', epoch, 'train_loss:', np.round(train_loss, 4),
              'Time:', np.round(elapsed(t_total), 2), 'RW %.2f' % t_rw, 'Train %.2f' % t_train)
        wandb.log({
            "trial_idx": trial_idx,
            "warmup/epoch": epoch,
            "warmup/train_loss": train_loss,
            "warmup/rw_time_sec": t_rw,
            "warmup/train_time_sec": t_train,
            "warmup/epoch_time_sec": t_epoch_sec,
        })

    total_time = elapsed(t_total)
    print('Warmup finished. Timing:', np.round(total_time, 1))
    wandb.log({
        "trial_idx": trial_idx,
        "warmup/total_time_sec": total_time,
        "warmup/final_loss": losses[-1] if losses else float('nan'),
    })
    return losses, total_time



def train_bfs_anchor_prepass(model, optimizer, criterion, bfs_states, bfs_distances, device, CFG, trial_idx):
    """Stochastic exact-distance supervised pretraining on the shallow BFS ball.

    This is intentionally cheap: it does not iterate over the entire BFS set
    unless you choose enough batches. It simply anchors the network to exact
    shortest-path labels near the identity before self-bootstrapped mDQN begins.
    """
    if not CFG.get('bfs_anchor_enabled', False):
        return [], 0.0
    n_epochs = int(CFG.get('bfs_anchor_pretrain_epochs', 0))
    if n_epochs <= 0 or bfs_states is None or bfs_distances is None:
        return [], 0.0

    print("\n--- BFS Exact Anchor Pretraining ---")
    t_total = now()
    losses = []
    n = bfs_states.shape[0]
    batch_size = int(CFG.get('bfs_anchor_batch_size', 4096))
    batches_per_epoch = int(CFG.get('bfs_anchor_pretrain_batches_per_epoch', 64))

    for epoch in range(n_epochs):
        t_epoch = now()
        model.train()
        running = 0.0
        for _ in range(batches_per_epoch):
            idx = torch.randint(0, n, (batch_size,), device=device)
            xb = bfs_states[idx]
            yb = bfs_distances[idx]
            pred = model(xb).view(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / max(1, batches_per_epoch)
        losses.append(avg_loss)
        epoch_time = elapsed(t_epoch)
        print('bfs anchor pretrain epoch:', epoch, 'loss:', np.round(avg_loss, 4),
              'Time:', np.round(elapsed(t_total), 2), 'Epoch %.2f' % epoch_time)
        wandb.log({
            "trial_idx": trial_idx,
            "bfs_anchor_pretrain/epoch": epoch,
            "bfs_anchor_pretrain/loss": avg_loss,
            "bfs_anchor_pretrain/epoch_time_sec": epoch_time,
            "bfs_anchor_pretrain/batches_per_epoch": batches_per_epoch,
            "bfs_anchor_pretrain/batch_size": batch_size,
        })

    total_time = elapsed(t_total)
    wandb.log({
        "trial_idx": trial_idx,
        "bfs_anchor_pretrain/total_time_sec": total_time,
        "bfs_anchor_pretrain/final_loss": losses[-1] if losses else float('nan'),
    })
    print('BFS exact anchor pretraining finished. Timing:', np.round(total_time, 1))
    return losses, total_time


def train_mdqn_pass1(model, optimizer, criterion, list_generators, tensor_generators, state_destination,
                     vec_hasher, dtype, device, CFG, trial_idx, bfs_states=None, bfs_distances=None):
    print("\n--- Pass 1: mDQN Training ---")
    t_total = now()
    losses = []
    for epoch in range(CFG['n_epochs_dqn']):
        t_epoch = now()
        t0 = now()
        X_train, y_train = random_walks(
            list_generators,
            n_random_walk_length=CFG['n_random_walk_length'],
            n_random_walks_to_generate=CFG['n_random_walks_to_generate_dqn'],
            n_random_walks_steps_back_to_ban=CFG['n_random_walks_steps_back_to_ban'],
            random_walks_type=CFG['random_walks_type'],
            state_rw_start=state_destination,
            device=device,
            dtype=dtype,
            vec_hasher=vec_hasher,
        )
        t_rw = elapsed(t0)

        t0 = now()
        neigb = get_neighbors(X_train, tensor_generators)
        y_bellman = torch.zeros(X_train.shape[0], device=device, dtype=torch.float)
        model.eval()
        with torch.no_grad():
            for i_start_batch in range(0, X_train.shape[0], CFG['batch_size']):
                i_end_batch = min(i_start_batch + CFG['batch_size'], X_train.shape[0])
                y_pred = model(neigb[i_start_batch:i_end_batch])
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                y_bellman[i_start_batch:i_end_batch] = (1 + torch.min(y_pred, dim=1)[0]).reshape(-1)
        y_train = torch.min(y_bellman, y_train.float())
        y_train = torch.clamp_min(y_train, 1)
        y_train[:CFG['n_random_walks_to_generate_dqn']] = 0
        if CFG['flag_dqn_round']:
            y_train = torch.round(y_train)
        t_bellman = elapsed(t0)

        indices = torch.randperm(X_train.shape[0], device=device)
        X_train = X_train[indices]
        y_train = y_train[indices]

        t0 = now()
        model.train()
        n_states_all = X_train.shape[0]
        cc = 0
        train_loss = 0.0
        for i_start_batch in range(0, n_states_all, CFG['batch_size']):
            i_end_batch = min(i_start_batch + CFG['batch_size'], n_states_all)
            outputs = model(X_train[i_start_batch:i_end_batch])
            loss = criterion(outputs.squeeze(), y_train[i_start_batch:i_end_batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            cc += 1
        train_loss /= cc
        t_train_rw = elapsed(t0)

        # Optional exact-BFS anchor updates. These are separate optimizer steps,
        # not a replacement for mDQN. They act as a shallow exact-distance
        # regularizer so the value function stays grounded near the identity.
        bfs_anchor_loss = float('nan')
        t_bfs_anchor = 0.0
        if (CFG.get('bfs_anchor_enabled', False)
                and bfs_states is not None
                and bfs_distances is not None
                and int(CFG.get('bfs_anchor_batches_per_mdqn_epoch', 0)) > 0
                and float(CFG.get('bfs_anchor_loss_weight', 0.0)) > 0.0):
            t0_bfs = now()
            model.train()
            n_bfs = bfs_states.shape[0]
            bfs_batch_size = int(CFG.get('bfs_anchor_batch_size', CFG['batch_size']))
            n_bfs_batches = int(CFG.get('bfs_anchor_batches_per_mdqn_epoch', 8))
            bfs_weight = float(CFG.get('bfs_anchor_loss_weight', 0.10))
            bfs_running = 0.0
            for _ in range(n_bfs_batches):
                idx_bfs = torch.randint(0, n_bfs, (bfs_batch_size,), device=device)
                xb = bfs_states[idx_bfs]
                yb = bfs_distances[idx_bfs]
                pred_bfs = model(xb).view(-1)
                loss_bfs_raw = criterion(pred_bfs, yb)
                loss_bfs = bfs_weight * loss_bfs_raw
                optimizer.zero_grad()
                loss_bfs.backward()
                optimizer.step()
                bfs_running += loss_bfs_raw.item()
            bfs_anchor_loss = bfs_running / max(1, n_bfs_batches)
            t_bfs_anchor = elapsed(t0_bfs)

        losses.append(train_loss)
        t_train = t_train_rw + t_bfs_anchor
        t_epoch_sec = elapsed(t_epoch)

        if epoch % 10 == 0:
            print('pass1 epoch:', epoch, 'train_loss:', np.round(train_loss, 4),
                  'bfs_anchor_loss:', np.round(bfs_anchor_loss, 4) if not np.isnan(bfs_anchor_loss) else 'nan',
                  'Time:', np.round(elapsed(t_total), 2), 'RW %.2f' % t_rw,
                  'Bellman %.2f' % t_bellman, 'TrainRW %.2f' % t_train_rw,
                  'BFSAnchor %.2f' % t_bfs_anchor)

        wandb.log({
            "trial_idx": trial_idx,
            "pass1_mdqn/epoch": epoch,
            "pass1_mdqn/train_loss": train_loss,
            "pass1_mdqn/bfs_anchor_loss": bfs_anchor_loss,
            "pass1_mdqn/bfs_anchor_weight": float(CFG.get('bfs_anchor_loss_weight', 0.0)),
            "pass1_mdqn/bfs_anchor_batches": int(CFG.get('bfs_anchor_batches_per_mdqn_epoch', 0)),
            "pass1_mdqn/rw_time_sec": t_rw,
            "pass1_mdqn/bellman_time_sec": t_bellman,
            "pass1_mdqn/train_rw_time_sec": t_train_rw,
            "pass1_mdqn/bfs_anchor_time_sec": t_bfs_anchor,
            "pass1_mdqn/train_time_sec": t_train,
            "pass1_mdqn/epoch_time_sec": t_epoch_sec,
        })

    total_time = elapsed(t_total)
    print('Pass 1 mDQN finished. Timing:', np.round(total_time, 1))
    wandb.log({
        "trial_idx": trial_idx,
        "pass1_mdqn/total_time_sec": total_time,
        "pass1_mdqn/final_loss": losses[-1] if losses else float('nan'),
    })
    return losses, total_time


class ResidualNet(nn.Module):
    """
    Bounded residual correction model for pass 2.

    It predicts only a nonnegative correction C(s) in [0, max_correction].
    The final pass-2 value used by beam search is:

        V_pass2(s) = V_pass1_frozen(s) + C(s)

    This is intentionally different from the old pass 2, which continued training
    the original value model and could destroy useful pass-1 rankings.
    """
    def __init__(self, input_size, hidden_dims, num_classes_for_one_hot, max_correction=2.0):
        super(ResidualNet, self).__init__()
        self.num_classes_for_one_hot = num_classes_for_one_hot
        self.max_correction = float(max_correction)
        in_features = input_size * num_classes_for_one_hot

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

        # Start close to zero correction so the residual pass initially behaves
        # like pass 1 instead of immediately shifting every value upward.
        last = self.layers[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -4.0)

    def forward(self, x):
        x = torch.nn.functional.one_hot(
            x.long(), num_classes=self.num_classes_for_one_hot
        ).float().flatten(start_dim=-2)
        raw = self.layers(x)
        return self.max_correction * torch.sigmoid(raw)


class CorrectedValueModel(nn.Module):
    """Wrapper used by beam search/error analysis: frozen pass1 + residual correction."""
    def __init__(self, base_model, residual_model):
        super(CorrectedValueModel, self).__init__()
        self.base_model = base_model
        self.residual_model = residual_model

    def forward(self, x):
        self.base_model.eval()
        with torch.no_grad():
            base = self.base_model(x)
            if isinstance(base, tuple):
                base = base[0]
        correction = self.residual_model(x)
        return base + correction



class SafeMinResidualValueModel(nn.Module):
    """Search-safe residual wrapper.

    Given pass1 value V1 and residual-corrected value V2, return

        V_safe(s) = min(V2(s), V1(s) + margin)

    With margin=0.0 this is exactly min(V1, V2). This preserves the full beam
    capacity while preventing residual rollout from over-penalizing states that
    pass1 strongly liked.
    """
    def __init__(self, base_model, residual_corrected_model, upward_margin=0.0):
        super(SafeMinResidualValueModel, self).__init__()
        self.base_model = base_model
        self.residual_corrected_model = residual_corrected_model
        self.upward_margin = float(upward_margin)

    def forward(self, x):
        self.base_model.eval()
        self.residual_corrected_model.eval()
        with torch.no_grad():
            v1 = self.base_model(x)
            if isinstance(v1, tuple):
                v1 = v1[0]
            v2 = self.residual_corrected_model(x)
            if isinstance(v2, tuple):
                v2 = v2[0]
        return torch.minimum(v2, v1 + self.upward_margin)


class WeightedCorrectedValueModel(nn.Module):
    """Frozen guide value + lambda * learned positive beam-aware penalty.

    Used for Pass 3:
        V_final(s) = V_calibrated(s) + lambda_beam * B(s)

    V_calibrated is usually the Pass 2 residual-corrected value model.
    B(s) is a positive-only beam-aware trap penalty in [0, max_penalty].
    """
    def __init__(self, guide_model, correction_model, alpha=0.25):
        super(WeightedCorrectedValueModel, self).__init__()
        self.guide_model = guide_model
        self.correction_model = correction_model
        self.alpha = float(alpha)

    def forward(self, x):
        self.guide_model.eval()
        with torch.no_grad():
            base = self.guide_model(x)
            if isinstance(base, tuple):
                base = base[0]
        correction = self.correction_model(x)
        return base + self.alpha * correction


def train_pass2_residual(base_model, criterion, list_generators, tensor_generators, state_destination,
                         vec_hasher, dtype, device, CFG, trial_idx):
    """
    Residual pass 2.

    Freeze the pass-1 model and train a separate bounded residual model. Targets are
    based on short greedy rollout under the frozen pass-1 model:

        raw_delta = rollout_target - bellman_target
        target_residual = clamp(alpha * (raw_delta - margin), 0, max_correction)

    where rollout_target = L + V_pass1(s_after_L_greedy_steps).

    The original value model is not updated in this pass.
    """
    print("\n--- Pass 2: Bounded Residual Rollout Correction ---")
    t_total = now()

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    max_correction = float(CFG.get('pass2_residual_max_correction', 2.0))
    residual_model = ResidualNet(
        input_size=CFG['n_permutations_length'],
        hidden_dims=CFG['list_layers_sizes'],
        num_classes_for_one_hot=CFG['n_permutations_length'],
        max_correction=max_correction,
    ).to(device)
    residual_optimizer = optim.Adam(residual_model.parameters(), lr=CFG['lr'])

    losses = []
    upward_rates = []
    target_means = []
    pred_means = []
    saturation_rates = []
    corrected_upward_rates = []
    raw_delta_means = []
    margin_adjusted_delta_means = []

    for epoch in range(CFG['pass2_n_epochs']):
        t_epoch = now()

        t0 = now()
        X_train, y_train = random_walks(
            list_generators,
            n_random_walk_length=CFG['pass2_walk_length'],
            n_random_walks_to_generate=CFG['pass2_n_walks'],
            n_random_walks_steps_back_to_ban=CFG['n_random_walks_steps_back_to_ban'],
            random_walks_type=CFG['random_walks_type'],
            state_rw_start=state_destination,
            device=device,
            dtype=dtype,
            vec_hasher=vec_hasher,
        )
        t_rw = elapsed(t0)

        # Bellman and rollout targets are computed from the frozen pass-1 model.
        t0 = now()
        neigb = get_neighbors(X_train, tensor_generators)
        y_bellman = torch.zeros(X_train.shape[0], device=device, dtype=torch.float)
        base_model.eval()
        with torch.no_grad():
            for i_start_batch in range(0, X_train.shape[0], CFG['batch_size']):
                i_end_batch = min(i_start_batch + CFG['batch_size'], X_train.shape[0])
                yp = base_model(neigb[i_start_batch:i_end_batch])
                if isinstance(yp, tuple):
                    yp = yp[0]
                y_bellman[i_start_batch:i_end_batch] = (1 + torch.min(yp, dim=1)[0]).reshape(-1)
        t_bellman = elapsed(t0)

        if CFG['pass2_use_rollout_correction']:
            t0 = now()
            y_rollout = torch.zeros(X_train.shape[0], device=device, dtype=torch.float)
            with torch.no_grad():
                for i_start_batch in range(0, X_train.shape[0], CFG['batch_size']):
                    i_end_batch = min(i_start_batch + CFG['batch_size'], X_train.shape[0])
                    y_rollout[i_start_batch:i_end_batch] = greedy_rollout_targets(
                        X_train[i_start_batch:i_end_batch],
                        base_model,
                        tensor_generators,
                        CFG['pass2_rollout_length'],
                        CFG['batch_size'],
                    )
            raw_delta = y_rollout - y_bellman
            residual_margin = float(CFG.get('pass2_residual_margin', 0.0))
            margin_adjusted_delta = raw_delta - residual_margin
            target_residual = torch.clamp(
                CFG['pass2_rollout_mix_alpha'] * margin_adjusted_delta,
                min=0.0,
                max=max_correction,
            )
            t_rollout = elapsed(t0)
        else:
            y_rollout = y_bellman
            raw_delta = torch.zeros_like(y_bellman)
            residual_margin = float(CFG.get('pass2_residual_margin', 0.0))
            margin_adjusted_delta = raw_delta - residual_margin
            target_residual = torch.zeros_like(y_bellman)
            t_rollout = 0.0

        up_mask = (raw_delta > 0).float()
        corrected_up_mask = (margin_adjusted_delta > 0).float()
        upward_rate = up_mask.mean().item()
        corrected_upward_rate = corrected_up_mask.mean().item()
        raw_delta_mean = raw_delta.mean().item()
        margin_adjusted_delta_mean = margin_adjusted_delta.mean().item()
        saturation_rate = (target_residual >= (max_correction - 1e-6)).float().mean().item()

        indices = torch.randperm(X_train.shape[0], device=device)
        X_train = X_train[indices]
        target_residual = target_residual[indices]

        t0 = now()
        residual_model.train()
        n_states_all = X_train.shape[0]
        cc = 0
        train_loss = 0.0
        pred_sum = 0.0
        n_pred = 0
        for i_start_batch in range(0, n_states_all, CFG['batch_size']):
            i_end_batch = min(i_start_batch + CFG['batch_size'], n_states_all)
            pred_residual = residual_model(X_train[i_start_batch:i_end_batch]).view(-1)
            target_batch = target_residual[i_start_batch:i_end_batch]
            loss = criterion(pred_residual, target_batch)
            residual_optimizer.zero_grad()
            loss.backward()
            residual_optimizer.step()
            train_loss += loss.item()
            pred_sum += pred_residual.detach().sum().item()
            n_pred += pred_residual.numel()
            cc += 1
        train_loss /= cc
        pred_mean = pred_sum / max(n_pred, 1)
        target_mean = target_residual.mean().item()
        losses.append(train_loss)
        upward_rates.append(upward_rate)
        corrected_upward_rates.append(corrected_upward_rate)
        raw_delta_means.append(raw_delta_mean)
        margin_adjusted_delta_means.append(margin_adjusted_delta_mean)
        target_means.append(target_mean)
        pred_means.append(pred_mean)
        saturation_rates.append(saturation_rate)

        t_train = elapsed(t0)
        t_epoch_sec = elapsed(t_epoch)

        if epoch % 10 == 0:
            print('pass2 residual epoch:', epoch,
                  'loss:', np.round(train_loss, 4),
                  'Time:', np.round(elapsed(t_total), 2),
                  'RW %.2f' % t_rw,
                  'Bellman %.2f' % t_bellman,
                  'Rollout %.2f' % t_rollout,
                  'Train %.2f' % t_train,
                  'UpwardRaw %.3f' % upward_rate,
                  'UpwardAfterMargin %.3f' % corrected_upward_rate,
                  'RawDeltaMean %.3f' % raw_delta_mean,
                  'MarginDeltaMean %.3f' % margin_adjusted_delta_mean,
                  'TargetMean %.3f' % target_mean,
                  'PredMean %.3f' % pred_mean,
                  'Saturated %.3f' % saturation_rate)

        wandb.log({
            "trial_idx": trial_idx,
            "pass2_residual/epoch": epoch,
            "pass2_residual/train_loss": train_loss,
            "pass2_residual/rw_time_sec": t_rw,
            "pass2_residual/bellman_time_sec": t_bellman,
            "pass2_residual/rollout_time_sec": t_rollout,
            "pass2_residual/train_time_sec": t_train,
            "pass2_residual/epoch_time_sec": t_epoch_sec,
            "pass2_residual/upward_corrections_raw_delta": upward_rate,
            "pass2_residual/upward_corrections_after_margin": corrected_upward_rate,
            "pass2_residual/raw_delta_mean": raw_delta_mean,
            "pass2_residual/margin_adjusted_delta_mean": margin_adjusted_delta_mean,
            "pass2_residual/target_residual_mean": target_mean,
            "pass2_residual/pred_residual_mean": pred_mean,
            "pass2_residual/saturation_rate": saturation_rate,
            "pass2_residual/max_correction": max_correction,
            "pass2_residual/margin": residual_margin,
        })

    corrected_model = CorrectedValueModel(base_model, residual_model).to(device)
    corrected_model.eval()

    total_time = elapsed(t_total)
    print('Pass 2 residual finished. Timing:', np.round(total_time, 1))
    wandb.log({
        "trial_idx": trial_idx,
        "pass2_residual/total_time_sec": total_time,
        "pass2_residual/final_loss": losses[-1] if losses else float('nan'),
        "pass2_residual/final_upward_corrections_raw_delta": upward_rates[-1] if upward_rates else float('nan'),
        "pass2_residual/final_upward_corrections_after_margin": corrected_upward_rates[-1] if corrected_upward_rates else float('nan'),
        "pass2_residual/final_raw_delta_mean": raw_delta_means[-1] if raw_delta_means else float('nan'),
        "pass2_residual/final_margin_adjusted_delta_mean": margin_adjusted_delta_means[-1] if margin_adjusted_delta_means else float('nan'),
        "pass2_residual/final_target_residual_mean": target_means[-1] if target_means else float('nan'),
        "pass2_residual/final_pred_residual_mean": pred_means[-1] if pred_means else float('nan'),
        "pass2_residual/final_saturation_rate": saturation_rates[-1] if saturation_rates else float('nan'),
        "pass2_residual/final_margin": float(CFG.get('pass2_residual_margin', 0.0)),
    })

    return corrected_model, losses, upward_rates, total_time



def _collect_beam_training_states(snapshots, CFG, trial_idx):
    """Collect a CPU tensor of states from pass-1 beam snapshots.

    Frontier means the beam-search states around a pruning decision: the states
    kept by top-k and the states just below the cutoff. We skip unsaturated early
    snapshots unless they occur at/after CFG['pass2_beam_aware_min_step'].
    """
    pieces = []
    max_per = int(CFG.get('pass2_beam_aware_max_states_per_snapshot', 20_000))
    min_step = int(CFG.get('pass2_beam_aware_min_step', 21))

    for snap in snapshots:
        step = int(snap.get('step', 0))
        kind = str(snap.get('kind', ''))
        if step < min_step:
            continue
        if not (
            'selected_top_by_current_model' in kind or
            'just_pruned_by_current_model' in kind or
            'selected_all_unsaturated' in kind
        ):
            continue
        states = snap['states'].detach().cpu()
        if states.numel() == 0:
            continue
        n = int(states.shape[0])
        if n > max_per:
            g = torch.Generator(device='cpu')
            g.manual_seed(12345 + 1000 * int(trial_idx) + step + (17 if 'pruned' in kind else 0))
            idx = torch.randperm(n, generator=g)[:max_per]
            states = states[idx]
        pieces.append(states)

    if not pieces:
        raise RuntimeError('No beam snapshot states were collected. Check snapshot_every / snapshot_store settings.')

    X = torch.cat(pieces, dim=0)
    # Deduplicate by exact rows on CPU to keep training smaller. Unique preserves no meaningful order here.
    try:
        X = torch.unique(X, dim=0)
    except Exception:
        pass
    return X


def _compute_beam_aware_targets(base_model, X_cpu, tensor_generators, device, CFG, batch_size=4096):
    """Compute beam-aware penalty targets for fixed beam-frontier states.

    For a candidate state s, let V1 be the frozen pass-1 model.

        progress(s) = V1(s) - V1(s_after_L_greedy_steps)
        deficit(s)  = expected_progress - progress(s)
        target       = clamp(beta * (deficit - margin), 0, max_penalty)

    expected_progress is deliberately smaller than rollout_length. In early
    beam-aware tests, 8-step greedy rollouts only reduced V by ~3.5-4.1 on
    average, so using L=8 saturated most targets. This target penalizes only
    states whose rollout progress is worse than the expected beam-frontier norm.
    """
    L_roll = int(CFG.get('pass2_beam_aware_rollout_length', 8))
    expected_progress = float(CFG.get('pass2_beam_aware_expected_progress', 4.0))
    beta = float(CFG.get('pass2_beam_aware_beta', 1.0))
    margin = float(CFG.get('pass2_beam_aware_margin', 0.25))
    max_penalty = float(CFG.get('pass2_beam_aware_max_penalty', 1.0))

    n = int(X_cpu.shape[0])
    targets = torch.empty(n, dtype=torch.float32)
    progresses = torch.empty(n, dtype=torch.float32)
    deficits = torch.empty(n, dtype=torch.float32)
    v0_all = torch.empty(n, dtype=torch.float32)

    base_model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            x = X_cpu[i:j].to(device, non_blocking=True)
            v0 = base_model(x)
            if isinstance(v0, tuple):
                v0 = v0[0]
            v0 = v0.view(-1)

            rollout_est = greedy_rollout_targets(
                x,
                base_model,
                tensor_generators,
                L_roll,
                bs=batch_size,
            )
            v_final = rollout_est - float(L_roll)
            progress = v0 - v_final
            deficit = expected_progress - progress
            target = torch.clamp(beta * (deficit - margin), min=0.0, max=max_penalty)

            v0_all[i:j] = v0.detach().cpu()
            progresses[i:j] = progress.detach().cpu()
            deficits[i:j] = deficit.detach().cpu()
            targets[i:j] = target.detach().cpu()

    stats = {
        'n_states': int(n),
        'target_mean': float(targets.mean().item()),
        'target_std': float(targets.std().item()),
        'target_saturation_rate': float((targets >= max_penalty - 1e-6).float().mean().item()),
        'target_positive_rate': float((targets > 0).float().mean().item()),
        'progress_mean': float(progresses.mean().item()),
        'progress_std': float(progresses.std().item()),
        'deficit_mean': float(deficits.mean().item()),
        'expected_progress': float(expected_progress),
        'target_margin': float(margin),
        'target_max_penalty': float(max_penalty),
        'target_beta': float(beta),
        'v1_mean_on_beam_train_states': float(v0_all.mean().item()),
    }
    return targets, stats


def train_pass3_beam_aware_value_correction(guide_model, criterion, tensor_generators,
                                            device, CFG, trial_idx, beam_snapshots):
    """Pass 3 beam-aware correction on top of residual-calibrated value.

    This freezes the guide model (normally Vcal = V1 + residual) and trains
    a bounded positive penalty model on pass1 beam-frontier states only.
    The final score is:
        V_final(s) = Vcal(s) + lambda_beam * B(s)
    """
    print("\n--- Pass 3: Beam-Aware Correction on Residual-Calibrated Value ---")
    t_total = now()

    guide_model.eval()
    for p in guide_model.parameters():
        p.requires_grad = False

    # 1. Collect states from the pass-1 beam frontier.
    X_beam_cpu = _collect_beam_training_states(beam_snapshots, CFG, trial_idx)
    print('Beam-aware train states:', X_beam_cpu.shape)

    # 2. Compute fixed targets using frozen residual-calibrated guide rollout progress.
    t0 = now()
    target_cpu, target_stats = _compute_beam_aware_targets(
        guide_model,
        X_beam_cpu,
        tensor_generators,
        device,
        CFG,
        batch_size=CFG['batch_size'],
    )
    target_time = elapsed(t0)
    print('Beam-aware target stats:', target_stats, 'target_time_sec:', round(target_time, 2))
    wandb.log({
        "trial_idx": trial_idx,
        "pass3_beam_aware/target_time_sec": target_time,
        **{f"pass3_beam_aware/{k}": v for k, v in target_stats.items()},
    })

    max_penalty = float(CFG.get('pass2_beam_aware_max_penalty', 1.0))
    penalty_model = ResidualNet(
        input_size=CFG['n_permutations_length'],
        hidden_dims=CFG['list_layers_sizes'],
        num_classes_for_one_hot=CFG['n_permutations_length'],
        max_correction=max_penalty,
    ).to(device)
    optimizer = optim.Adam(penalty_model.parameters(), lr=CFG['lr'])

    n_states = int(X_beam_cpu.shape[0])
    epochs = int(CFG.get('pass2_beam_aware_epochs', 20))
    losses = []

    # Keep fixed tensors on CPU and move batches to GPU. This avoids holding too
    # much GPU memory when snapshots are large.
    for epoch in range(epochs):
        t_epoch = now()
        g = torch.Generator(device='cpu')
        g.manual_seed(9876 + 1000 * int(trial_idx) + epoch)
        perm = torch.randperm(n_states, generator=g)

        penalty_model.train()
        train_loss = 0.0
        pred_sum = 0.0
        n_pred = 0
        batches = 0
        for start in range(0, n_states, CFG['batch_size']):
            end = min(start + CFG['batch_size'], n_states)
            idx = perm[start:end]
            x = X_beam_cpu[idx].to(device, non_blocking=True)
            y = target_cpu[idx].to(device, non_blocking=True)
            pred = penalty_model(x).view(-1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred_sum += pred.detach().sum().item()
            n_pred += pred.numel()
            batches += 1

        train_loss /= max(batches, 1)
        pred_mean = pred_sum / max(n_pred, 1)
        losses.append(train_loss)
        epoch_time = elapsed(t_epoch)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print('beam-aware epoch:', epoch,
                  'loss:', np.round(train_loss, 4),
                  'PredMean %.3f' % pred_mean,
                  'TargetMean %.3f' % target_stats['target_mean'],
                  'EpochTime %.2f' % epoch_time)
        wandb.log({
            "trial_idx": trial_idx,
            "pass3_beam_aware/epoch": epoch,
            "pass3_beam_aware/train_loss": train_loss,
            "pass3_beam_aware/pred_mean": pred_mean,
            "pass3_beam_aware/target_mean": target_stats['target_mean'],
            "pass3_beam_aware/epoch_time_sec": epoch_time,
        })

    lambda_beam = float(CFG.get('pass3_beam_aware_score_lambda', 0.25))
    corrected_model = WeightedCorrectedValueModel(guide_model, penalty_model, alpha=lambda_beam).to(device)
    corrected_model.eval()
    total_time = elapsed(t_total)
    print('Pass 3 beam-aware finished. Timing:', np.round(total_time, 1), 'lambda:', lambda_beam)
    wandb.log({
        "trial_idx": trial_idx,
        "pass3_beam_aware/total_time_sec": total_time,
        "pass3_beam_aware/final_loss": losses[-1] if losses else float('nan'),
    })
    return corrected_model, penalty_model, losses, target_stats, total_time

# =============================================================================
# 3. TRIAL LOOP
# =============================================================================
N_TRIALS = 30
_all_pass1_path_lengths = []
_all_residual_path_lengths = []
_all_final_path_lengths = []
_all_final_best_lambdas = []
_all_pass1_found = []
_all_residual_found = []
_all_final_found = []
_all_pass1_spearman = []
_all_residual_spearman = []
_all_final_spearman = []
_all_pass1_argmin = []
_all_residual_argmin = []
_all_final_argmin = []
_trial_rows = []
_lambda_path_lengths = {}
_lambda_found = {}

for trial_idx in range(N_TRIALS):
    run_config = dict(CFG)
    run_config["trial_idx"] = trial_idx
    run_config["trial_number"] = trial_idx + 1
    run_config["n_trials"] = N_TRIALS

    run = wandb.init(
        project=WANDB_PROJECT,
        config=run_config,
        name=f"trial_{trial_idx + 1:02d}",
        group=WANDB_GROUP,
        job_type="trial",
        reinit=True,
    )

    trial_t0 = now()
    print("\n" + "=" * 80)
    print(f"=== TRIAL {trial_idx + 1}/{N_TRIALS} ===")
    print("=" * 80 + "\n")

    setup_t0 = now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = CFG['n_permutations_length']
    L, R, X = get_LRX_moves(n)
    dict_generators = {'L': L, 'R': R, 'X': X}
    list_generators = [L, R, X]
    dtype_generators = torch.int64
    tensor_generators = torch.tensor(list_generators, device=device, dtype=dtype_generators)

    n_unique_symbols_in_states = len(list_generators[0])
    dtype = torch.uint8 if n_unique_symbols_in_states <= 256 else torch.uint16

    state_size = len(list_generators[0])
    max_int = int(2**62)
    dtype_for_hash = torch.int64
    vec_hasher = torch.randint(-max_int, max_int + 1, size=(state_size,), device=device, dtype=dtype_for_hash)

    # Notebook uses dtype_generators here; random_walks converts to dtype internally.
    state_destination = torch.arange(len(list_generators[0]), device=device, dtype=dtype_generators)

    # Longest LRX permutation / solve state.
    p = np.arange(n)
    p[0], p[1] = p[1], p[0]
    i = 2
    while i < n - i + 1:
        p[i], p[n - i + 1] = p[n - i + 1], p[i]
        i += 1
    permutation_longest = torch.tensor(p, dtype=dtype, device=device)
    state_start = permutation_longest

    model = Net(input_size=n, hidden_dims=CFG['list_layers_sizes'], num_classes_for_one_hot=n).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])

    setup_time = elapsed(setup_t0)
    print('device:', device)
    print('tensor_generators.shape:', tensor_generators.shape)
    print('state_destination:', state_destination)
    print('state_start:', state_start)
    wandb.log({"trial_idx": trial_idx, "setup/time_sec": setup_time})

    # -------------------------------------------------------------------------
    # Warmup: original n_epochs supervised/random-walk training from notebook
    # -------------------------------------------------------------------------
    warmup_losses, warmup_time = train_warmup(
        model, optimizer, criterion, list_generators, state_destination,
        vec_hasher, dtype, device, CFG, trial_idx
    )

    # -------------------------------------------------------------------------
    # Reference BFS set for error analysis. Not a graph-generation cell; this is
    # needed for the pass1/pass2 diagnostics.
    # -------------------------------------------------------------------------
    bfs_t0 = now()
    radius_max = 30
    stop_threshold_total_states = 10**6
    dict_growth, dict_additional_data = bfs_growth_permutations_torch_simple(
        list_generators,
        center_states=state_destination,
        radius_max=radius_max,
        stop_threshold_total_states=stop_threshold_total_states,
        flag_return_all_states=True,
        flag_return_all_hashes=True,
        flag_return_list_distances=True,
        device=device,
        dtype=dtype,
        vec_hasher=vec_hasher,
        verbose=10,
    )
    bfs_time = elapsed(bfs_t0)
    array_states_all_saved = dict_additional_data['array_states_all']
    list_distances_save = dict_additional_data['list_distances']
    print('Result - dict_growth:', dict_growth)
    print('Total BFS states:', np.sum(list(dict_growth.values())))
    wandb.log({
        "trial_idx": trial_idx,
        "diagnostic_bfs/time_sec": bfs_time,
        "diagnostic_bfs/total_states": int(np.sum(list(dict_growth.values()))),
        "diagnostic_bfs/max_distance_reached": int(max(dict_growth.keys())),
    })

    # Prepare exact-BFS anchor tensors for supervised anchoring.
    bfs_anchor_states = array_states_all_saved
    bfs_anchor_distances = torch.tensor(list_distances_save, device=device, dtype=torch.float32)
    if CFG.get('bfs_anchor_max_states', None) is not None:
        max_bfs_anchor = int(CFG['bfs_anchor_max_states'])
        if bfs_anchor_states.shape[0] > max_bfs_anchor:
            idx_bfs_anchor = torch.randperm(bfs_anchor_states.shape[0], device=device)[:max_bfs_anchor]
            bfs_anchor_states = bfs_anchor_states[idx_bfs_anchor]
            bfs_anchor_distances = bfs_anchor_distances[idx_bfs_anchor]
    wandb.log({
        "trial_idx": trial_idx,
        "bfs_anchor/n_states": int(bfs_anchor_states.shape[0]),
        "bfs_anchor/max_distance": float(bfs_anchor_distances.max().item()),
        "bfs_anchor/mean_distance": float(bfs_anchor_distances.float().mean().item()),
    })

    # Optional cheap exact-BFS supervised pre-pass.
    bfs_anchor_pretrain_losses, bfs_anchor_pretrain_time = train_bfs_anchor_prepass(
        model, optimizer, criterion, bfs_anchor_states, bfs_anchor_distances, device, CFG, trial_idx
    )

    # -------------------------------------------------------------------------
    # Pass 1 mDQN
    # -------------------------------------------------------------------------
    pass1_losses, pass1_train_time = train_mdqn_pass1(
        model, optimizer, criterion, list_generators, tensor_generators,
        state_destination, vec_hasher, dtype, device, CFG, trial_idx,
        bfs_states=bfs_anchor_states,
        bfs_distances=bfs_anchor_distances,
    )

    # Pass 1 beam search: path length is the main metric.
    # Also save sparse frontier snapshots so we can later compare how pass 2
    # changes values on the actual states beam search cared about.
    pass1_beam_snapshots = []
    pass1_path_len, pass1_found, pass1_beam_time = run_beam_search(
        "pass1", model, state_start, state_destination, list_generators,
        tensor_generators, vec_hasher, dtype, device, CFG, trial_idx,
        snapshot_store=pass1_beam_snapshots,
        snapshot_every=50,
        snapshot_max_states=65536,
    )
    wandb.log({
        "trial_idx": trial_idx,
        "pass1_beam_snapshots/n_snapshots": len(pass1_beam_snapshots),
    })
    _all_pass1_path_lengths.append(int(pass1_path_len))
    _all_pass1_found.append(bool(pass1_found))

    # Pass 1 error analysis.
    pass1_diag = run_error_analysis(
        "pass1", model, array_states_all_saved, list_distances_save,
        tensor_generators, vec_hasher, device, trial_idx
    )
    _all_pass1_spearman.append(pass1_diag["spearman_all"])
    _all_pass1_argmin.append(pass1_diag["argmin_match"])

    # -------------------------------------------------------------------------
    # Pass 2: residual rollout calibration
    # -------------------------------------------------------------------------
    if CFG['pass2_enabled']:
        residual_model, residual_losses, residual_upward_rates, residual_train_time = train_pass2_residual(
            model, criterion, list_generators, tensor_generators, state_destination,
            vec_hasher, dtype, device, CFG, trial_idx
        )

        log_value_delta_stats(
            "value_delta_bfs/pass2_residual_vs_pass1",
            model,
            residual_model,
            array_states_all_saved,
            trial_idx,
            device,
            true_d=list_distances_save,
            batch_size=4096,
            max_states=None,
            seed=trial_idx,
        )
        log_beam_snapshot_value_deltas(
            model,
            residual_model,
            pass1_beam_snapshots,
            trial_idx,
            device,
            batch_size=4096,
            max_states_per_snapshot=65536,
        )

        residual_path_len, residual_found, residual_beam_time = run_beam_search(
            "pass2_residual", residual_model, state_start, state_destination, list_generators,
            tensor_generators, vec_hasher, dtype, device, CFG, trial_idx
        )
        _all_residual_path_lengths.append(int(residual_path_len))
        _all_residual_found.append(bool(residual_found))

        residual_diag = run_error_analysis(
            "pass2_residual", residual_model, array_states_all_saved, list_distances_save,
            tensor_generators, vec_hasher, device, trial_idx
        )
        _all_residual_spearman.append(residual_diag["spearman_all"])
        _all_residual_argmin.append(residual_diag["argmin_match"])

        # ---------------------------------------------------------------------
        # Pass 3: beam-aware penalty trained on top of residual-calibrated value,
        # then lambda sweep at inference. The sweep is logged for analysis; report
        # fixed-lambda statistics and best-of-sweep separately.
        # ---------------------------------------------------------------------
        default_lambda = float(CFG.get('pass3_beam_aware_score_lambda', 0.25))
        default_final_model, beam_aware_penalty_model, pass3_losses, pass3_target_stats, pass3_train_time = train_pass3_beam_aware_value_correction(
            residual_model, criterion, tensor_generators,
            device, CFG, trial_idx, pass1_beam_snapshots
        )

        # Log diagnostics for the default lambda only, to keep the sweep cheap.
        log_value_delta_stats(
            "value_delta_bfs/pass3_default_vs_pass2_residual",
            residual_model,
            default_final_model,
            array_states_all_saved,
            trial_idx,
            device,
            true_d=list_distances_save,
            batch_size=4096,
            max_states=None,
            seed=trial_idx,
        )
        log_value_delta_stats(
            "value_delta_bfs/pass3_default_vs_pass1",
            model,
            default_final_model,
            array_states_all_saved,
            trial_idx,
            device,
            true_d=list_distances_save,
            batch_size=4096,
            max_states=None,
            seed=trial_idx,
        )
        log_beam_snapshot_value_deltas(
            residual_model,
            default_final_model,
            pass1_beam_snapshots,
            trial_idx,
            device,
            batch_size=4096,
            max_states_per_snapshot=65536,
        )

        # Lambda sweep: do NOT retrain the penalty. Only change inference score:
        # score(s) = V_residual(s) + lambda * B(s)
        lambdas = CFG.get('pass3_beam_aware_score_lambdas', [default_lambda])
        lambdas = [float(x) for x in lambdas]
        pass3_sweep_results = {}
        lambda_rows = []

        best_pass3_path_len = int(CFG['n_steps_limit'])
        best_pass3_found = False
        best_pass3_lambda = None
        best_pass3_time = None
        best_pass3_model = None

        for lambda_beam in lambdas:
            print(f"\n--- Starting Pass 3 Beam Search | lambda_beam={lambda_beam} ---")
            sweep_model = WeightedCorrectedValueModel(
                residual_model,
                beam_aware_penalty_model,
                alpha=lambda_beam,
            ).to(device)
            sweep_model.eval()

            lambda_label = f"pass3_lambda_{str(lambda_beam).replace('.', '_')}"
            lambda_path_len, lambda_found, lambda_beam_time = run_beam_search(
                lambda_label, sweep_model, state_start, state_destination, list_generators,
                tensor_generators, vec_hasher, dtype, device, CFG, trial_idx
            )

            lambda_key = str(lambda_beam).replace('.', '_')
            pass3_sweep_results[f"pass3_lambda_{lambda_key}/path_length"] = int(lambda_path_len)
            pass3_sweep_results[f"pass3_lambda_{lambda_key}/found"] = int(bool(lambda_found))
            pass3_sweep_results[f"pass3_lambda_{lambda_key}/beam_time_sec"] = float(lambda_beam_time)

            lambda_rows.append({
                "trial": trial_idx + 1,
                "lambda": lambda_beam,
                "path_length": int(lambda_path_len),
                "found": bool(lambda_found),
                "beam_time_sec": float(lambda_beam_time),
            })
            _lambda_path_lengths.setdefault(lambda_key, []).append(int(lambda_path_len))
            _lambda_found.setdefault(lambda_key, []).append(bool(lambda_found))

            # Prefer found solutions; among found solutions, prefer shorter path.
            if bool(lambda_found):
                if (not best_pass3_found) or int(lambda_path_len) < int(best_pass3_path_len):
                    best_pass3_found = True
                    best_pass3_path_len = int(lambda_path_len)
                    best_pass3_lambda = float(lambda_beam)
                    best_pass3_time = float(lambda_beam_time)
                    best_pass3_model = sweep_model
            elif not best_pass3_found and int(lambda_path_len) < int(best_pass3_path_len):
                best_pass3_path_len = int(lambda_path_len)
                best_pass3_lambda = float(lambda_beam)
                best_pass3_time = float(lambda_beam_time)
                best_pass3_model = sweep_model

        if best_pass3_model is None:
            best_pass3_model = WeightedCorrectedValueModel(
                residual_model,
                beam_aware_penalty_model,
                alpha=default_lambda,
            ).to(device)
            best_pass3_model.eval()
            best_pass3_lambda = default_lambda
            best_pass3_path_len = int(CFG['n_steps_limit'])
            best_pass3_found = False
            best_pass3_time = -1.0

        _all_final_path_lengths.append(int(best_pass3_path_len))
        _all_final_found.append(bool(best_pass3_found))
        _all_final_best_lambdas.append(float(best_pass3_lambda))

        # Error analysis only for the best lambda from the sweep.
        final_diag = run_error_analysis(
            f"pass3_best_lambda_{str(best_pass3_lambda).replace('.', '_')}",
            best_pass3_model,
            array_states_all_saved,
            list_distances_save,
            tensor_generators,
            vec_hasher,
            device,
            trial_idx,
        )
        _all_final_spearman.append(final_diag["spearman_all"])
        _all_final_argmin.append(final_diag["argmin_match"])

        improvement_residual = int(pass1_path_len) - int(residual_path_len)
        improvement_final = int(pass1_path_len) - int(best_pass3_path_len)
        improvement_final_vs_residual = int(residual_path_len) - int(best_pass3_path_len)
        best_of_three_path = min(int(pass1_path_len), int(residual_path_len), int(best_pass3_path_len))

        wandb.log({
            "trial_idx": trial_idx,
            **pass3_sweep_results,
            "pass3_sweep/best_path_length": int(best_pass3_path_len),
            "pass3_sweep/best_found": int(bool(best_pass3_found)),
            "pass3_sweep/best_lambda": float(best_pass3_lambda),
            "pass3_sweep/best_beam_time_sec": float(best_pass3_time),
            "trial/pass1_path_length": int(pass1_path_len),
            "trial/pass2_residual_path_length": int(residual_path_len),
            "trial/pass3_best_path_length": int(best_pass3_path_len),
            "trial/improvement_pass1_minus_residual": improvement_residual,
            "trial/improvement_pass1_minus_pass3_best": improvement_final,
            "trial/improvement_residual_minus_pass3_best": improvement_final_vs_residual,
            "trial/best_of_three_path_length": best_of_three_path,
            "trial/pass1_found": bool(pass1_found),
            "trial/pass2_residual_found": bool(residual_found),
            "trial/pass3_best_found": bool(best_pass3_found),
        })

        # Save per-lambda rows too; useful for later analysis.
        lambda_sweep_df = pd.DataFrame(lambda_rows)
        lambda_sweep_df.to_csv(f"trial_{trial_idx + 1:02d}_pass3_lambda_sweep.csv", index=False)

        _trial_rows.append({
            "trial": trial_idx + 1,
            "pass1_path_length": int(pass1_path_len),
            "pass1_found": bool(pass1_found),
            "pass2_residual_path_length": int(residual_path_len),
            "pass2_residual_found": bool(residual_found),
            "pass3_best_path_length": int(best_pass3_path_len),
            "pass3_best_found": bool(best_pass3_found),
            "pass3_best_lambda": float(best_pass3_lambda),
            "improvement_pass1_minus_residual": improvement_residual,
            "improvement_pass1_minus_pass3_best": improvement_final,
            "improvement_residual_minus_pass3_best": improvement_final_vs_residual,
            "best_of_three_path_length": best_of_three_path,
            "pass1_spearman": pass1_diag["spearman_all"],
            "pass2_residual_spearman": residual_diag["spearman_all"],
            "pass3_best_spearman": final_diag["spearman_all"],
            "pass1_mae": pass1_diag["mae"],
            "pass2_residual_mae": residual_diag["mae"],
            "pass3_best_mae": final_diag["mae"],
            "pass3_target_mean": pass3_target_stats.get("target_mean", float("nan")),
            "pass3_target_saturation_rate": pass3_target_stats.get("target_saturation_rate", float("nan")),
            "pass3_target_positive_rate": pass3_target_stats.get("target_positive_rate", float("nan")),
            "pass3_sweep_lambdas": json.dumps(lambdas),
            **{f"lambda_{str(row['lambda']).replace('.', '_')}_path": int(row["path_length"]) for row in lambda_rows},
            **{f"lambda_{str(row['lambda']).replace('.', '_')}_found": bool(row["found"]) for row in lambda_rows},
        })

    trial_time = elapsed(trial_t0)
    wandb.log({"trial_idx": trial_idx, "trial/total_time_sec": trial_time})
    print(f"TRIAL {trial_idx + 1} finished in {trial_time:.2f}s")

    # Finish this W&B run so each trial appears as a separate run.
    wandb.finish()

    # Memory cleanup between trials.
    try:
        del final_model
    except Exception:
        pass
    del model, optimizer
    try:
        del array_states_all_saved, list_distances_save, bfs_anchor_states, bfs_anchor_distances, dict_additional_data, dict_growth, pass1_beam_snapshots
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# =============================================================================
# 4. FINAL SUMMARY
# =============================================================================
summary = {
    "summary/pass1_mean_path_length": float(np.mean(_all_pass1_path_lengths)) if _all_pass1_path_lengths else float('nan'),
    "summary/pass2_residual_mean_path_length": float(np.mean(_all_residual_path_lengths)) if _all_residual_path_lengths else float('nan'),
    "summary/pass3_best_mean_path_length": float(np.mean(_all_final_path_lengths)) if _all_final_path_lengths else float('nan'),
    "summary/pass1_found_rate": float(np.mean(_all_pass1_found)) if _all_pass1_found else float('nan'),
    "summary/pass2_residual_found_rate": float(np.mean(_all_residual_found)) if _all_residual_found else float('nan'),
    "summary/pass3_best_found_rate": float(np.mean(_all_final_found)) if _all_final_found else float('nan'),
    "summary/pass1_mean_spearman": float(np.nanmean(_all_pass1_spearman)) if _all_pass1_spearman else float('nan'),
    "summary/pass2_residual_mean_spearman": float(np.nanmean(_all_residual_spearman)) if _all_residual_spearman else float('nan'),
    "summary/pass3_best_mean_spearman": float(np.nanmean(_all_final_spearman)) if _all_final_spearman else float('nan'),
    "summary/pass1_mean_argmin_match": float(np.nanmean(_all_pass1_argmin)) if _all_pass1_argmin else float('nan'),
    "summary/pass2_residual_mean_argmin_match": float(np.nanmean(_all_residual_argmin)) if _all_residual_argmin else float('nan'),
    "summary/pass3_best_mean_argmin_match": float(np.nanmean(_all_final_argmin)) if _all_final_argmin else float('nan'),
}
if _all_final_best_lambdas:
    summary["summary/pass3_best_lambda_mean"] = float(np.mean(_all_final_best_lambdas))

# Fixed-lambda aggregate summaries. These are the clean numbers to report if you do
# not want to select lambda per seed.
for lambda_key, vals in _lambda_path_lengths.items():
    arr = np.asarray(vals, dtype=float)
    if arr.size > 0:
        prefix = f"summary/pass3_lambda_{lambda_key}"
        summary[f"{prefix}_mean_path_length"] = float(np.mean(arr))
        summary[f"{prefix}_std_path_length"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        summary[f"{prefix}_sem_path_length"] = float(summary[f"{prefix}_std_path_length"] / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        summary[f"{prefix}_ci95_halfwidth_normal"] = float(1.96 * summary[f"{prefix}_sem_path_length"]) if arr.size > 1 else 0.0
        summary[f"{prefix}_median_path_length"] = float(np.median(arr))
        summary[f"{prefix}_found_rate"] = float(np.mean(_lambda_found.get(lambda_key, []))) if lambda_key in _lambda_found else float('nan')

if _trial_rows:
    best_paths = [row["best_of_three_path_length"] for row in _trial_rows]
    summary["summary/best_of_three_mean_path_length"] = float(np.mean(best_paths))
    summary["summary/best_of_three_found_rate"] = float(np.mean([p < CFG['n_steps_limit'] for p in best_paths]))

    # Distribution summaries for 30-trial reporting. Treat n_steps_limit as the failure sentinel.
    for prefix, vals in [
        ("pass1", _all_pass1_path_lengths),
        ("pass2_residual", _all_residual_path_lengths),
        ("pass3_best", _all_final_path_lengths),
        ("best_of_three", best_paths),
    ]:
        arr = np.asarray(vals, dtype=float)
        if arr.size > 0:
            summary[f"summary/{prefix}_std_path_length"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            summary[f"summary/{prefix}_sem_path_length"] = float(summary[f"summary/{prefix}_std_path_length"] / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            summary[f"summary/{prefix}_median_path_length"] = float(np.median(arr))
            summary[f"summary/{prefix}_min_path_length"] = float(np.min(arr))
            summary[f"summary/{prefix}_max_path_length"] = float(np.max(arr))
            summary[f"summary/{prefix}_ci95_halfwidth_normal"] = float(1.96 * summary[f"summary/{prefix}_sem_path_length"]) if arr.size > 1 else 0.0

with open("trial_summary_bfs_anchor_residual_beamaware_30trials.json", "w") as f:
    json.dump(summary, f, indent=2)

if _trial_rows:
    pd.DataFrame(_trial_rows).to_csv("trial_results_bfs_anchor_residual_beamaware_30trials.csv", index=False)

print("\nAll trials complete!")
print(summary)
if _trial_rows:
    print(pd.DataFrame(_trial_rows).to_string(index=False))
print("Saved aggregate summary to trial_summary_bfs_anchor_residual_beamaware_30trials.json")
print("Saved per-trial results to trial_results_bfs_anchor_residual_beamaware_30trials.csv")
