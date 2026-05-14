# Koltsov3 Gradient Boosting (XGBoost) Random-Walk Sweep

This folder runs Koltsov3 random-walk **gradient boosting** sweeps on UW Hyak
with Slurm. It is the gradient boosting counterpart to Merav's MLP sweep
(`Merav/koltsov3_mlp_sweep/`): same problem, same random-walk data generation,
same output conventions — the MLP is replaced with an **XGBoost regressor**, and
raw permutation states are replaced with **hand-crafted features**.

Files:

- `koltsov3_gb_sweep.py` — main Python experiment script
- `koltsov3_gb_sweep.sbatch` — example Hyak Slurm batch script
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
- `--n-random-walks-values` — number of training random walks
- `--walk-length-multipliers` — walk length is `n * multiplier`
- `--random-walk-types` — `simple` or `non-backtracking-beam`
- `--steps-back-to-ban-values` — for `non-backtracking-beam`, previous moves to ban
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
python koltsov3_gb_sweep.py \
  --n-values 5,6 \
  --n-random-walks-values 50 \
  --walk-length-multipliers 4 \
  --random-walk-types non-backtracking-beam \
  --steps-back-to-ban-values 2 \
  --n-estimators-values 30 \
  --max-depth-values 3,5 \
  --learning-rate-values 0.1 \
  --subsample-values 0.8 \
  --colsample-bytree-values 0.8 \
  --min-child-weight-values 5 \
  --reg-lambda-values 1.0 \
  --reg-alpha-values 0.0 \
  --n-val-samples-values 20 \
  --n-test-samples-values 20 \
  --seed-values 0 \
  --output-dir smoke_test \
  --compute-bfs-metadata true \
  --use-wandb false
```

Expected outputs appear in `smoke_test/`. (R² will be low on this tiny config —
that is expected; it only checks the pipeline runs.)

## Hyak batch job

1. Keep these files together in one directory on Hyak:

   ```text
   koltsov3_gb_sweep.py
   koltsov3_gb_sweep.sbatch
   README_koltsov3_gb_sweep.md
   ```

   Make sure `Rithikesh/.env` exists one directory up (for W&B).

2. Edit `koltsov3_gb_sweep.sbatch` — replace every `TODO`:

   - `--account=TODO_ACCOUNT` — run `hyakalloc` to find your account
   - `--partition=TODO_PARTITION` — pick a partition you can use
   - the `source /path/to/your/venv/bin/activate` line — point it at your
     Python environment (venv or conda)

3. Submit:

   ```bash
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
