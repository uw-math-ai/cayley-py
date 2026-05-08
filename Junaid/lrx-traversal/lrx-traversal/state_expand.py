"""State expansion for Koltsov3 beam search — Issue #002.

Applies the 3 Koltsov3 generators (I, K, S) to a batch of permutation states
using torch.gather on GPU.
"""

import numpy as np
import torch


def build_koltsov3_generator_tensors(n: int, k: int = 0) -> torch.Tensor:
    """Return (3, n) int64 tensor of I, K, S permutation generators on GPU.

    I: swaps adjacent pairs (0,1), (2,3), ...
    K: swaps adjacent pairs (1,2), (3,4), ...
    S: swaps positions k and k+2
    """
    I = np.arange(n)
    K = np.arange(n)
    S = np.arange(n)

    for i in range(0, n - 1, 2):
        I[i], I[i + 1] = I[i + 1], I[i]
    for i in range(1, n - 1, 2):
        K[i], K[i + 1] = K[i + 1], K[i]
    S[k], S[k + 2] = S[k + 2], S[k]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(np.array([I, K, S]), dtype=torch.int64, device=device)


def expand_neighbors(states: torch.Tensor, gen_tensors: torch.Tensor) -> torch.Tensor:
    """Expand W states into W*3 neighbors by applying all generators.

    Args:
        states: (W, n) int64 tensor on GPU
        gen_tensors: (3, n) int64 tensor on GPU

    Returns:
        (W*3, n) int64 tensor — I(s0), K(s0), S(s0), I(s1), ...
    """
    W, n = states.shape
    n_gen = gen_tensors.shape[0]

    # Repeat each state n_gen times, then gather with corresponding generator
    # states: (W, n) → (W, 1, n) → expand → (W, n_gen, n) → reshape → (W*n_gen, n)
    states_expanded = states.unsqueeze(1).expand(W, n_gen, n).reshape(W * n_gen, n)

    # gen_tensors: (n_gen, n) → repeat W times → (W*n_gen, n)
    moves = gen_tensors.repeat(W, 1)

    # Apply gather: for each state, reorder according to generator
    result = torch.gather(states_expanded, 1, moves)

    return result
