# Koltsov3 Cayley Graph Traversal

Value function learning and beam search for finding shortest paths from any permutation back to the identity element on the Koltsov3 Cayley graph of the symmetric group S<sub>n</sub>.

## Problem

The **Koltsov3 generators** are three involutions on S<sub>n</sub>:

| Generator | Action | Preserves |
|-----------|--------|-----------|
| **I** | Swap adjacent pairs (0,1), (2,3), … | Position parity |
| **K** | Swap adjacent pairs (1,2), (3,4), … | Position parity |
| **S** | Swap positions *k* and *k*+2 (default *k*=0) | Nothing |

Every permutation can be sorted to the identity by some sequence of I, K, S moves. The **diameter** of the graph is the longest shortest path. We want to (a) learn a heuristic value function that predicts the distance from any state to identity, and (b) use it in beam search to find short paths.

## Architecture

```
 state_expand.py          mlp_model.py
 (GPU generators,         (MLP value function,
  batch neighbor           train / save / load)
  expansion)                   │
      │                        │
      └────────┬───────────────┘
               │
          beam_search.py
          (MLP-guided beam search)
               │
               │
      eval_beam_search.py      bfs_results/
      (batch evaluation,       (BFS ground truth:
       CSV output, CLI)         diameters, longest
                                elements, VRAM data)
```

**Dependency graph:**

```
state_expand.py
    ↓
mlp_model.py  ──→  beam_search.py
                        ↓
                eval_beam_search.py
```

Experiment scripts import nothing from this package; they inline the necessary functions for standalone execution.

---

## Core Library

### `state_expand.py` — GPU State Expansion

Applies the three Koltsov3 generators to batches of permutation states using `torch.gather` on GPU.

**Public API:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_koltsov3_generator_tensors` | `(n: int, k: int = 0) → torch.Tensor` | Returns `(3, n)` int64 tensor of I, K, S generator permutations. Auto-places on GPU if CUDA is available. |
| `expand_neighbors` | `(states: torch.Tensor, gen_tensors: torch.Tensor) → torch.Tensor` | Expands `(W, n)` states into `(W*3, n)` neighbors by applying all generators. The output ordering is I(s₀), K(s₀), S(s₀), I(s₁), K(s₁), S(s₁), … |

**Algorithm:** Each generator is a permutation array. Applying generator *g* to state *s* means `s[g]` (reorder *s* according to *g*). `expand_neighbors` does this in one batched `torch.gather` call: repeat states 3×, repeat generators W×, gather. No Python loop over states.

**Generator construction** (for `k=0`):

- **I:** swaps adjacent pairs (0,1), (2,3), … — stops at n−1
- **K:** swaps adjacent pairs (1,2), (3,4), … — stops at n−1
- **S:** swaps positions 0 and 2 (only if n > 2)

---

### `mlp_model.py` — MLP Value Function

One-hot MLP that predicts random-walk distance from a permutation state to the identity. Supports train / save / load / inference.

**Public API:**

| Class/Function | Signature | Description |
|----------------|-----------|-------------|
| `MLP` | `(n: int, hidden_dim: int)` | `nn.Module` subclass. Architecture: one-hot(n×n) → Linear(n², hidden_dim) → ReLU → Linear(hidden_dim, 1). Input `(batch, n)` int64, output `(batch, 1)` scalar. |
| `save_model` | `(model: MLP, path: str) → None` | Saves state dict + metadata (n, hidden_dim) to a `.pth` file. |
| `load_model` | `(path: str, device: str = "auto") → MLP` | Loads model from `.pth`. `device="auto"` picks CUDA if available. Sets `model.eval()`. |
| `train_koltsov3_mlp` | `(n, hidden_dim=512, epochs=25, lr=0.001, device="auto", seed=42) → MLP` | Full training pipeline. Returns trained model in eval mode. |

**Training algorithm:** Each epoch generates fresh random walks from identity (20K walks for n≤40, 10K for larger n). Walk length = 8×n. Every step of every walk becomes a labeled sample (step number = distance label). Trained with MSE loss, Adam optimizer, batch size 1024. No validation split — the full epoch data is shuffled and used for training.

**Model file format (`.pth`):** A Python dict with keys `state_dict` (PyTorch state dict), `n` (int), `hidden_dim` (int).

---

### `beam_search.py` — MLP-Guided Beam Search

Vanilla beam search using an MLP value function as the heuristic. Supports deduplication and non-backtracking history.

**Public API:**

| Name | Type | Description |
|------|------|-------------|
| `BeamResult` | `namedtuple` | Return type with fields: `path_found` (bool), `path` (Optional[List[int]] — generator indices 0=I,1=K,2=S), `path_length` (int), `steps_taken` (int — iterations executed), `states_visited` (int — total unique states scored), `runtime_sec` (float). |
| `beam_search` | function | Signature below. |

**`beam_search` signature:**

```python
def beam_search(
    start_state,           # (n,) array-like permutation
    model,                 # MLP from mlp_model.py
    gen_tensors,           # (3, n) int64 tensor from state_expand.py
    beam_width: int,       # states to keep per step
    step_limit: int,       # max search iterations
    *,
    deduplicate: bool = True,    # skip states seen in any previous beam
    history_size: int = 32,      # non-backtracking history (0 = disabled)
) -> BeamResult:
```

**Algorithm:**

1. **Early exit:** If start state equals identity, return immediately with 0 steps.
2. **Loop (up to `step_limit` iterations):**
   - **Expand:** Call `expand_neighbors(beam, gen_tensors)` — produces `beam_width × 3` candidates.
   - **Score:** Run `model(candidates)` — lower scores are better (closer to identity).
   - **Mask:** Set score to ∞ for states in the visited set (dedup) and recent history (non-backtracking).
   - **Select:** `torch.topk` the lowest `beam_width` scores.
   - **Check:** If any selected state equals identity, reconstruct and return the path.
   - **Update:** Add selected states to visited set and history deque.
3. **Step limit exceeded:** Return `path_found=False`.

**Path reconstruction:** Each candidate's index encodes its parent: `original_idx // 3` is the parent beam index, `original_idx % 3` is the generator (0=I, 1=K, 2=S).

**Non-backtracking history:** A `deque(maxlen=history_size)` of recently visited state tuples. These are masked with ∞ score (unless they are the identity). This prevents the beam from immediately undoing previous moves.

---

### `eval_beam_search.py` — Batch Beam Search Evaluation

Runs beam search across multiple target states and beam widths, producing a CSV of per-run metrics.

**Public API:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `evaluate_beam_search` | `(model_path, target_states, beam_widths, step_limit, *, optimal_lengths=None, device="auto") → pd.DataFrame` | Runs beam search for every (state, beam_width) combination. Returns DataFrame with columns: `n`, `state_idx`, `start_state`, `path_found`, `path_length`, `steps_taken`, `beam_width`, `runtime_sec`, `path`, `states_visited`. If `optimal_lengths` is provided, also includes `optimal_length` and `path_vs_optimal_ratio`. |

**CLI usage:**

```bash
python eval_beam_search.py \
    --model models/mlp_n16.pth \
    --states longest_elements.json \
    --beam-widths 8 16 32 64 \
    --step-limit 50 \
    --output results.csv \
    --device cuda
```

| Flag | Required | Description |
|------|----------|-------------|
| `--model` | Yes | Path to `.pth` model file. |
| `--states` | Yes | JSON file: list of state lists, or `{"longest_elements": [...]}`. |
| `--beam-widths` | Yes | Space-separated beam widths to sweep. |
| `--step-limit` | Yes | Maximum iterations per beam search run. |
| `--output` | Yes | Path for output CSV. |
| `--device` | No | `auto` (default), `cuda`, or `cpu`. |

---

## Experiment Scripts

These scripts are **standalone**: they inline `get_koltsov3_moves`, `extract_features`, `generate_random_walks`, and the `MLP` class rather than importing from the core library. Run them from the repository root.

### `run_lightgbm.py` — LightGBM vs MLP Head-to-Head

Trains and compares LightGBM (engineered features) vs MLP (one-hot) for a single permutation length *n*.

**CLI:**

```bash
python run_lightgbm.py [n] [--walks N] [--max-train N] [--mlp-epochs N] [--lgb-trees N]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `n` (positional) | 16 | Permutation length. |
| `--walks` | 25000 | Number of random walks for training. |
| `--max-train` | 500000 | Maximum training samples (subsampled). |
| `--mlp-epochs` | 15 | MLP training epochs. |
| `--lgb-trees` | 1000 | LightGBM num boost rounds. |

**Output:** Console metrics (R², RMSE, Spearman ρ) for both models on train/val/test, plus PNG files: `feature_importance.png`, `predictions_vs_true.png`, `training_curves.png`.

**LightGBM features:** ~3n + 20 engineered features encoding displacement statistics, parity mismatch, adjacent pair structure, inversions, descents, generator-specific I/K/S features, theoretical lower bounds, inverse permutation positions, and raw position values.

---

### `run_sweep.py` — Full Hyperparameter Sweep

Comprehensive sweep over *n* values and hyperparameters for both LightGBM and MLP. Logs to Weights & Biases. Supports resume-after-interrupt.

**CLI:**

```bash
python run_sweep.py [--n 16 32 48] [--quick] [--model lgb|mlp|both]
                    [--n-walks N] [--max-train N] [--save-models] [--seed N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | 8,16,24,32,40,48,56,64 | Specific n values. |
| `--quick` | False | Reduced grid (n=16,32; 1 config each). |
| `--model` | both | `lgb`, `mlp`, or `both`. |
| `--n-walks` | 25000 | Random walks per training run. |
| `--max-train` | 500000 | Max training samples (subsampled). |
| `--save-models` | False | Save LightGBM models to `models/`. |
| `--seed` | 42 | Random seed. |

**Sweep grids (full mode):**

| Model | Hyperparameters |
|-------|----------------|
| LightGBM | num_leaves ∈ {31,63,127,255}, max_depth ∈ {6,8,10,12}, learning_rate ∈ {0.05,0.1,0.2} |
| MLP | hidden_dim ∈ {128,256,512}, epochs ∈ {15,25}, learning_rate ∈ {0.001} |

**Output:** `sweep_results.csv` (appended incrementally — resumable). Metrics: R², RMSE, Spearman ρ on train/val/test, fit time, predict time, feature count, best iteration. Wandb project: `koltsov3-sweep`.

---

### `run_lgb_extended.py` — Extended LightGBM Tests

Tests ideas to break LightGBM's plateau (R² ≈ 0.67–0.69 in the main sweep): more trees (2000–3000), more leaves (511), DART boosting, lower learning rates, Huber loss.

**CLI:**

```bash
python run_lgb_extended.py [--quick] [--seed N]
```

**Configurations tested:** 12 configs × 4 n values (16, 32, 48, 64) = 48 runs total. Includes baselines (1000 trees, 127 leaves), more-trees configs (2000/3000 trees), 511 leaves, low LR (0.02, 0.01), DART boosting with dropout 0.1, and Huber loss (α=0.9).

**Output:** `lgb_extended_results.csv`. Wandb project: `koltsov3-sweep` (tag: `lgb_extended`).

---

### `run_mlp_extended.py` — Extended MLP Tests

Tests whether more epochs (50, 100), AdamW, or different learning rates improve MLP beyond the sweep best (hidden_dim=512, epochs=25, Adam).

**CLI:**

```bash
python run_mlp_extended.py [--quick] [--seed N]
```

**Configurations tested:** 3 epoch values × 4 optimizer configs × 4 n values = 48 runs total. Optimizer configs: Adam (lr=0.001, 0.0005), AdamW (lr=0.001/0.0005, weight_decay=0.01).

**Output:** `mlp_extended_results.csv`. Wandb project: `koltsov3-sweep` (tag: `mlp_extended`).

---

## BFS Ground Truth (`bfs_results/`)

Scripts that compute exact BFS diameters via `cayleypy` and analyze VRAM/runtime scaling. These live in `bfs_results/` and operate on JSON files in the same directory.

### `bfs_results/run_koltsov3_bfs.py` — BFS Diameter Computation

Runs exhaustive BFS on the Koltsov3 Cayley graph using `cayleypy.CayleyGraph.bfs()`.

```bash
cd bfs_results
python run_koltsov3_bfs.py              # n=5..12
python run_koltsov3_bfs.py --n 7 8 9    # specific n
```

**Output:** One JSON file per *n* (`koltsov3_bfs_n05.json` through `n12.json`). Each contains: generators, total state count, diameter, layer sizes, longest elements (permutations at max distance), and runtime.

### `bfs_results/run_koltsov3_bfs_vram.py` — BFS with VRAM Monitoring

Same as above but additionally measures `torch.cuda.max_memory_allocated`, peak reserved memory, and theoretical peak expand memory (`max_layer_size × 3 × n × 8` bytes).

```bash
cd bfs_results
python run_koltsov3_bfs_vram.py              # n=5..12
python run_koltsov3_bfs_vram.py --n 10 11 12 # specific n
```

**Extra JSON fields:** `peak_vram_allocated_gb`, `peak_vram_reserved_gb`, `peak_expand_theoretical_gb`, `max_layer_size`.

### `bfs_results/analyze_vram.py` — VRAM & Runtime Model Fitting

Loads the BFS JSON data and fits three VRAM scaling models plus two runtime models. Extrapolates to n=13..16.

```bash
cd bfs_results
python analyze_vram.py
```

**Models:**

| Model | Formula | Fit window |
|-------|---------|------------|
| A | log₁₀(VRAM) = a·n + b | n ≥ 9 |
| B | VRAM = a·n! + b | n ≥ 9 |
| C | VRAM = a·max_layer·3·n·8/1e9 + b | n ≥ 9 |
| Time | log₁₀(Time) = a·n + b | all n |
| Time2 | Time = a·n!·n·3 + b | all n |

**Output:** Console tables — raw data, model fits, extrapolation to n=13–16, final consolidated table with GPU class recommendations, memory bandwidth estimates.

### `bfs_results/final_table.py` — Clean VRAM & Time Estimates Table

Prints a single consolidated table: n=5..16 with measured VRAM (n≤12), estimated VRAM (average of Models B and C for n≥13), measured/predicted time, and the GPU class required.

```bash
cd bfs_results
python final_table.py
```

---

## Tests

All tests use `pytest`. Run from the repository root:

```bash
pytest -v
```

### `test_state_expand.py`

Covers `state_expand.py`. Validates: identity expansion produces generator permutations, generators are valid permutations for n∈{5,8,13,32}, GPU placement when CUDA is available, correct output shapes for varying W and n, neighbor validity for random input permutations, and GPU performance (<10ms for 10K states at n=16).

### `test_mlp_model.py`

Covers `mlp_model.py`. Validates: save/load round-trip produces identical outputs, trained model scores identity lower than reversed state, batch and single-state output shapes are correct, model metadata (n, hidden_dim) survives round-trip.

### `test_beam_search.py`

Covers `beam_search.py`. Validates: identity start returns immediately, beam search finds valid paths for n=5 longest elements (acceptance criterion), returned paths transform start state to identity, beam_width=1 doesn't crash, step counting is correct, `states_visited` is monotonic with step_limit, identity scores lowest in candidate beam, dedup reduces states scored, history_size=0 works, all four (dedup, history) toggle combinations run, `states_visited` never exceeds `step_limit × beam_width × 3`, `step_limit=0` returns `path_found=False` (except for identity start).

### `test_eval_beam_search.py`

Covers `eval_beam_search.py`. Validates: returns DataFrame with correct columns and row count, failed searches produce rows with `path_found=False`, `path_vs_optimal_ratio` is computed when `optimal_lengths` is provided, CLI end-to-end writes CSV output.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a small MLP value function
python -c "
from mlp_model import train_koltsov3_mlp, save_model
model = train_koltsov3_mlp(n=8, hidden_dim=128, epochs=10, device='cpu')
save_model(model, 'models/mlp_n8.pth')
print('Model saved to models/mlp_n8.pth')
"

# Run beam search on a target state
python -c "
import torch
from beam_search import beam_search
from mlp_model import load_model
from state_expand import build_koltsov3_generator_tensors

n = 8
model = load_model('models/mlp_n8.pth', device='cpu')
gens = build_koltsov3_generator_tensors(n)
# A state 1 step from identity
state = [1, 0, 3, 2, 5, 4, 7, 6]

result = beam_search(state, model, gens, beam_width=16, step_limit=20)
print(f'Path found: {result.path_found}')
print(f'Path: {result.path}')
print(f'Path length: {result.path_length}')
print(f'Time: {result.runtime_sec:.4f}s')
"

# Run tests
pytest -v
```

## Output Files

| File | Produced by | Description |
|------|-------------|-------------|
| `bfs_results/koltsov3_bfs_n*.json` | `run_koltsov3_bfs.py` / `run_koltsov3_bfs_vram.py` | BFS ground truth per n |
| `bfs_results/koltsov3_vram_estimates.csv` | Manual export | VRAM estimate data |
| `sweep_results.csv` | `run_sweep.py` | Hyperparameter sweep results |
| `lgb_extended_results.csv` | `run_lgb_extended.py` | Extended LightGBM results |
| `mlp_extended_results.csv` | `run_mlp_extended.py` | Extended MLP results |
| `feature_importance.png` | `run_lightgbm.py` | Top 20 LightGBM features by gain |
| `predictions_vs_true.png` | `run_lightgbm.py` | Scatter: predicted vs true distance |
| `training_curves.png` | `run_lightgbm.py` | Loss curves for both models |
| `models/*.pth` | `run_sweep.py --save-models` | Saved MLP model checkpoints |
| `lgb_models/*.txt` | `run_lgb_extended.py` | Saved LightGBM model files |

## Key Dependencies

- **PyTorch** 2.5+ — MLP model, GPU state expansion
- **cayleypy** — BFS on Cayley graphs (ground truth)
- **LightGBM** 4.6+ — tree-based value function baseline
- **scikit-learn** — metrics, train/test split
- **pandas, numpy, scipy** — data handling, statistics
- **matplotlib** — plotting (headless via `Agg` backend)
- **wandb** — experiment tracking (sweep scripts)

## Data Flow

```
Identity permutation
       │
       ├──→ BFS (cayleypy) ──→ bfs_results/*.json (ground truth:
       │                        diameters, longest elements, VRAM)
       │
       └──→ Random walks ──→ Training data (state, distance pairs)
                  │
                  ├──→ LightGBM (engineered features)
                  │
                  └──→ MLP (one-hot) ──→ mlp_model.py (save .pth)
                                              │
                                              ▼
                                     beam_search.py (heuristic search)
                                              │
                                              ▼
                                     eval_beam_search.py (CSV results)
```
