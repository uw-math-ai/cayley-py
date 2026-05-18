# Koltsov3 Gradient Boosting (XGBoost) Random-Walk Sweep

This folder runs Koltsov3 random-walk **gradient boosting** sweeps on UW Hyak
with Slurm. It is the gradient boosting counterpart to Merav's MLP sweep
(`Merav/koltsov3_mlp_sweep/`): same problem, same random-walk data generation,
same output conventions — the MLP is replaced with an **XGBoost regressor**, and
raw permutation states are replaced with **hand-crafted features**.

Files:

- `koltsov3_gb_sweep.py` — main Python experiment script
- `koltsov3_gb_sweep.sbatch` — example Hyak Slurm batch script
- `smoke_test.sh` — tiny 4-config end-to-end smoke test (`bash smoke_test.sh`)
- `setup_env.sh` — one-shot Klone conda env setup (`bash setup_env.sh`)
- `inspect_results.py` — post-sweep diagnostics on a results directory
  (`python inspect_results.py <output_dir>`)
- `README_koltsov3_gb_sweep.md` — this file

The W&B API key lives in `Rithikesh/.env` (one directory up). That file is
gitignored and is loaded automatically — see [W&B setup](#wandb-setup) below.

## What the script does

It trains an XGBoost model to predict **normalized random-walk depth**
(`step / walk_length`, in `[0, 1]`) for states generated from Koltsov3 random
walks.

Pipeline per configuration:

1. construct Koltsov3 generators for permutations of length `n`
2. generate train / validation / test random-walk datasets (each generated
   **once** — XGBoost has no epochs, unlike the MLP script)
3. turn each permutation state into hand-crafted features (see
   [Features](#features))
4. train one XGBoost model with early stopping
5. evaluate train / validation / test RMSE, R², and Spearman correlation
6. record per-boosting-iteration RMSE history and feature importance (gain)
7. optionally log everything to W&B
8. save CSV results and plots

Like the MLP script, it runs a **Cartesian-product sweep** over many
command-line arguments.

## How this differs from the MLP sweep

| Aspect | MLP sweep (Merav) | GB sweep (this folder) |
|---|---|---|
| Model | PyTorch MLP | XGBoost regressor |
| Input | one-hot permutation (`n*n` dims) | hand-crafted features (`extract_features`) |
| Training loop | epochs; training walks regenerated each epoch | trained once; data generated once per config |
| "epoch" CSV | `epoch_results.csv` | `iteration_results.csv` (one row per boosting round) |
| Hyperparameters | width, lr, batch size, epochs | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha` |
| Extra outputs | param count | per-run feature importance (gain) CSV + PNG |

The Koltsov3 generator logic, random-walk logic, label definition
(normalized depth), evaluation metrics, BFS metadata, and output scaffolding
(partial CSVs, `run_args.json`, plots) are kept the same so results are
comparable to the MLP sweep.

## Features

The feature set is ported from Junaid's LightGBM work on this same Koltsov3
problem (`Junaid/lrx-traversal/`). For each permutation state, `extract_features()`
computes ~13 groups:

1. **Displacement** — `|state[i] - i|` per index, plus sum/max/mean/std/median and counts
2. **Parity mismatch** — values sitting at the wrong parity index
3. **Adjacent pairs** — sortedness, adjacent differences, correct adjacent pairs
4. **Inversions** — count, fraction, parity
5. **Descents** — count of `state[i] > state[i+1]`
6. **I-generator features** — sortedness/correctness on the `(0 1)(2 3)...` positions
7. **K-generator features** — same for the `(1 2)(3 4)...` positions
8. **S-generator features** — the `(k, k+2)` swap
9. **Theoretical lower bounds** — displacement- and parity-based distance bounds
10. **Inverse permutation** — where each value currently sits
11. **Longest correct run** — longest streak of already-correct positions
12. **Signed displacement / skew**
13. **Raw positions** — the permutation values themselves

**How to decide which features to keep:** start with the full set. Every run
writes `feature_importance/importance_<run>.csv` (gain) and a PNG, and the
summary CSV stores `top_features_by_gain`. After the first sweep, inspect those
and prune low-gain features if desired.

## Important arguments

Each sweep argument accepts a comma-separated list; the script runs every
combination.

### Data / random-walk axes

- `--n-values` — permutation sizes, e.g. `8,12,16,24` (default)
- `--n-random-walks-values` — number of training random walks (Cartesian-swept)
- `--walks-per-n` — optional per-n walk override, e.g.
  `5:500,16:25000,24:25000`. Any n listed here uses the mapped walk count;
  unlisted n fall back to `--n-random-walks-values`. Useful because the right
  walk count scales with state space (n!): small n saturate at hundreds of
  walks, large n need tens of thousands.
- `--walk-length-multipliers` — walk length is `n * multiplier`
- `--random-walk-types` — `simple` (uniform random) or `non-backtracking-beam`.
  **For training, use `simple`** — beam-search-based walks are designed to stay
  on promising paths, which gives terrible state-space coverage and crushes the
  model with label collision. `non-backtracking-beam` is the *inference-time*
  algorithm in the paper, not a training-data generator.
- `--steps-back-to-ban-values` — for `non-backtracking-beam` only; ignored
  silently for `simple` walks (the script warns when this happens)
- `--dedup-strategy` — `none` (default) or `first-visit`. `first-visit`
  implements the paper's diffusion-distance label spec: keep one row per unique
  state labeled by its earliest visit step. Without dedup the same state
  appears with multiple conflicting labels and the model converges to
  predicting the per-state mean. Applied to train/val/test together (changing
  the strategy changes what the regression target is).
- `--n-val-samples-values` / `--n-test-samples-values` — validation/test walk counts
- `--seed-values` — random seeds

### XGBoost hyperparameter axes

- `--n-estimators-values` — `num_boost_round` ceiling (early stopping may stop sooner)
- `--max-depth-values` — tree depth
- `--learning-rate-values` — boosting learning rate
- `--subsample-values` — row subsample fraction
- `--colsample-bytree-values` — column subsample fraction
- `--min-child-weight-values` — minimum child weight
- `--reg-lambda-values` — L2 regularization
- `--reg-alpha-values` — L1 regularization

### Training controls

- `--early-stopping-rounds` — patience on validation RMSE; `0` disables (default `50`)
- `--max-train-samples` — cap on training rows to bound memory for large `n`; `0` disables (default `500000`)
- `--nthread` — XGBoost threads; `-1` uses all cores
- `--verbose-eval` — XGBoost log period; `0` silences
- `--top-features-to-record` — how many top-gain features to store in the summary CSV

### Device / problem / output

- `--device auto|cpu|cuda` — XGBoost `hist` runs well on CPU; `auto` uses CUDA if available
- `--koltsov3-k` — `k` in the Koltsov3 `S=(k,k+2)` generator
- `--output-dir` — output directory for CSVs and plots
- `--compute-bfs-metadata` — exact BFS diameter/layers; only safe for small `n`
- `--max-bfs-states` — safety cap; BFS is skipped if `n!` exceeds this

### W&B arguments

- `--use-wandb true|false`
- `--wandb-entity` (default `CayleyPy`)
- `--wandb-project` (default `cayley-py`)
- `--wandb-group` — group name for the whole sweep (defaults to a timestamped name)

## W&B setup

The script loads `WANDB_API_KEY` from, in order:

1. the existing environment variable, if set
2. `Rithikesh/.env` (one directory up from this folder) — **gitignored**
3. `.env` in the current working directory

So with `Rithikesh/.env` present, `--use-wandb true` just works — no manual
`wandb login` needed. The `.env` file is in `.gitignore` and must never be
committed.

> Security note: if the API key is ever exposed (shared in chat, committed by
> accident, etc.), rotate it at <https://wandb.ai/settings>.

## Small smoke test

Run this first to check imports, paths, and the environment:

```bash
bash smoke_test.sh
```

It runs 4 tiny configurations (n=5,6 × max_depth=3,5) with simple walks and
first-visit dedup, no W&B. Finishes in seconds. Output goes to `smoke_test/`.
R² will be modest on this tiny config — the smoke test only verifies the
pipeline runs end-to-end, not that the model is well-tuned.

After it finishes, inspect the results:

```bash
python inspect_results.py smoke_test
```

That prints final per-config metrics, per-n label/uniqueness stats, and a
representative training curve.

## Hyak (Klone) environment setup

The `coenv/python/*` modules on Klone are built without `libffi`, so
`import ctypes` fails and torch/xgboost/wandb cannot import. The setup therefore
uses a **conda environment on gscratch** (the home-directory quota is also too
small for torch + CUDA deps).

`setup_env.sh` automates the whole thing — install Miniconda on gscratch, create
the `cayley` env, install all dependencies, verify the imports. Run it once:

```bash
bash /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_sweep/setup_env.sh
```

It is safe to re-run (it skips steps already done).

To use the env manually in a new shell:

```bash
source /gscratch/stf/rmuddana/miniconda3/etc/profile.d/conda.sh
conda activate cayley
```

## Hyak batch job

1. Make sure the repo is on Klone at `/gscratch/stf/rmuddana/cayley-py` and the
   `cayley` conda env exists (run `setup_env.sh` above).

2. The `.sbatch` is already filled in for this setup (`account=stf`,
   `partition=ckpt`, conda env on gscratch). No edits needed unless your paths
   differ.

3. Submit from the sweep folder:

   ```bash
   cd /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_sweep
   sbatch koltsov3_gb_sweep.sbatch
   ```

4. Watch logs and the queue:

   ```bash
   tail -f logs/koltsov3_gb_sweep_<JOBID>.out
   squeue -u rmuddana
   ```

## Output files

Inside `--output-dir`:

- `summary_results.csv` — one row per completed configuration
- `iteration_results.csv` — one row per boosting iteration per configuration
- `summary_results_partial.csv` / `iteration_results_partial.csv` — incremental, written after each config
- `run_args.json` — command-line arguments used
- `plots/rmse_by_config.png`, `plots/spearman_by_config.png` — per-config overviews
- `plots/rmse_vs_max_depth_*.png` — depth trend plots when `max_depth` varies
- `plots/rmse_by_iteration_*.png` — per-boosting-iteration training curves (up to 25 configs)
- `feature_importance/importance_*.csv` and `*.png` — gain importance per configuration

## Tracked metrics (summary CSV)

- **Performance:** `train_rmse`/`val_rmse`/`test_rmse`, `train_r2`/`val_r2`/`test_r2`,
  `train_spearman`/`val_spearman`/`test_spearman`
- **Training history:** `best_iteration`, `n_boosted_rounds`, `final_train_rmse`,
  `final_val_rmse`, `best_val_rmse_during_training`, `best_iteration_by_val_rmse`
- **Gaps:** `train_val_rmse_gap`, `train_test_rmse_gap`, `val_test_rmse_gap`
- **Dataset size / uniqueness:** `n_train_states`/`n_val_states`/`n_test_states`,
  `num_unique_*_states`, `unique_*_fraction`
- **Dedup / state space:** `dedup_strategy`, `raw_train_rows_generated`,
  `train_dedup_factor` (raw rows / unique rows; `1.0` when dedup is off),
  `state_space_size` (`n!`)
- **Labels:** `label_min`, `label_max`, `label_mean`, `label_std`
- **Features:** `n_features`, `top_features_by_gain` (JSON)
- **Timing:** `data_time_sec`, `fit_time_sec`, `predict_time_sec`
- **Optional BFS:** `diameter`, `last_layer_count`, `layer_sizes` (only when
  `--compute-bfs-metadata true` and `n!` is small enough)

## Scaling notes for large n

- The full graph has up to `n!` permutation states.
- Random-walk dataset size is roughly `n_random_walks * walk_length_multiplier * n`.
- The feature count grows roughly linearly with `n` (per-index features).
- `--max-train-samples` caps training rows to bound feature-matrix memory.
- `non-backtracking-beam` samples allowed moves row-by-row, which can be slow for
  very large random-walk counts.
- Keep `--compute-bfs-metadata false` for larger `n`.
- A full sweep size is the product of the lengths of every comma-separated list
  — estimate it before launching, and start with a smoke test.
