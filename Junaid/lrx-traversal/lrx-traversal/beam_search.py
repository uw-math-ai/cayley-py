"""Beam search for Koltsov3 — Issues #003 + #004.

Vanilla beam search with optional dedup and non-backtracking history.
"""

import time
from collections import namedtuple, deque
from typing import List, Optional

import torch
from state_expand import expand_neighbors

BeamResult = namedtuple('BeamResult', [
    'path_found',       # bool — True if identity was reached
    'path',             # Optional[List[int]] — generator indices (0=I,1=K,2=S)
    'path_length',      # int — length of the returned path
    'steps_taken',      # int — beam search iterations executed
    'states_visited',   # int — total unique states scored
    'runtime_sec',      # float — wall clock time in seconds
])


def beam_search(start_state, model, gen_tensors, beam_width: int,
                step_limit: int, *,
                deduplicate: bool = True,
                history_size: int = 32) -> BeamResult:
    """Beam search using MLP value function as heuristic.

    Args:
        start_state: (n,) array-like permutation
        model: MLP from mlp_model.py
        gen_tensors: (3, n) int64 tensor from state_expand.py
        beam_width: number of states to keep per step
        step_limit: maximum search iterations
        deduplicate: if True, skip states already seen in any previous beam
        history_size: non-backtracking history size (0 = disabled)

    Returns:
        BeamResult with path_found, path (gen indices), path_length,
        steps_taken, states_visited, runtime_sec.
    """
    t0 = time.perf_counter()

    # Convert start state to tensor
    if not isinstance(start_state, torch.Tensor):
        start_state = torch.tensor(start_state, dtype=torch.int64,
                                   device=gen_tensors.device)
    else:
        start_state = start_state.to(dtype=torch.int64, device=gen_tensors.device)

    n = len(start_state)
    device = gen_tensors.device
    identity = torch.arange(n, dtype=torch.int64, device=device)

    # Ensure model is on the same device as gen_tensors
    model_device = next(model.parameters()).device
    if model_device != device:
        model = model.to(device)

    # Early exit: start state is already identity
    if torch.equal(start_state, identity):
        return BeamResult(
            path_found=True,
            path=[],
            path_length=0,
            steps_taken=0,
            states_visited=0,
            runtime_sec=time.perf_counter() - t0,
        )

    # --- Search loop ---
    beam = start_state.unsqueeze(0)  # (1, n)
    beam_paths: List[List[int]] = [[]]
    total_states_scored = 0

    # Dedup state
    visited: Optional[set] = set() if deduplicate else None
    if visited is not None:
        visited.add(tuple(start_state.cpu().tolist()))

    # Non-backtracking history
    history: Optional[deque] = (
        deque(maxlen=history_size) if history_size > 0 else None
    )

    for step in range(step_limit):
        # --- Expand ---
        candidates = expand_neighbors(beam, gen_tensors)  # (W*3, n)
        n_candidates = candidates.shape[0]

        # --- Score ---
        with torch.no_grad():
            scores = model(candidates).squeeze(-1)  # (W*3,)
        total_states_scored += n_candidates

        # --- Dedup: mask already-visited states with inf score ---
        if visited is not None:
            cand_list = candidates.cpu().tolist()
            for i, c in enumerate(cand_list):
                if tuple(c) in visited:
                    scores[i] = float('inf')

        # --- History: mask recently seen states (except identity) ---
        if history is not None:
            cand_list = candidates.cpu().tolist()
            id_list = identity.cpu().tolist()
            for i, c in enumerate(cand_list):
                tup = tuple(c)
                if tup != tuple(id_list) and tup in history:
                    scores[i] = float('inf')

        # --- Select top W ---
        k = min(beam_width, n_candidates)
        _, top_indices = torch.topk(scores, k, largest=False)

        selected = candidates[top_indices]  # (W, n)

        # --- Check for identity ---
        is_identity = (selected == identity.unsqueeze(0)).all(dim=1)
        if is_identity.any():
            match_idx = is_identity.nonzero(as_tuple=True)[0][0].item()
            original_idx = top_indices[match_idx].item()
            parent_idx = original_idx // 3
            gen_idx = original_idx % 3
            path = beam_paths[parent_idx] + [gen_idx]
            return BeamResult(
                path_found=True,
                path=path,
                path_length=len(path),
                steps_taken=step + 1,
                states_visited=total_states_scored,
                runtime_sec=time.perf_counter() - t0,
            )

        # --- Update visited set ---
        if visited is not None:
            sel_list = selected.cpu().tolist()
            for s in sel_list:
                visited.add(tuple(s))

        # --- Update history deque ---
        if history is not None:
            sel_list = selected.cpu().tolist()
            for s in sel_list:
                history.append(tuple(s))

        # --- Update beam and paths ---
        beam = selected
        new_paths = []
        for i in range(k):
            original_idx = top_indices[i].item()
            parent_idx = original_idx // 3
            gen_idx = original_idx % 3
            new_paths.append(beam_paths[parent_idx] + [gen_idx])
        beam_paths = new_paths

    # Step limit exceeded
    return BeamResult(
        path_found=False,
        path=None,
        path_length=0,
        steps_taken=step_limit,
        states_visited=total_states_scored,
        runtime_sec=time.perf_counter() - t0,
    )
