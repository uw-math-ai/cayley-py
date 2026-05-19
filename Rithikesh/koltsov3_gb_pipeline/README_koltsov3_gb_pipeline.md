# Koltsov3 Gradient Boosting (XGBoost) Pipeline — Algorithm 4

This folder implements Chervov & Soibelman's **Algorithm 4 (Modified DQN)** on
the Koltsov3 random-walk problem, with **XGBoost** as the distance estimator
`f_theta(s)`. It runs as a Cartesian-product sweep on UW Hyak with Slurm.

Three phases run per configuration:

1. **Warm-up / diffusion distance.** Generate random walks from the identity,
   label each visited state by normalized random-walk depth, and fit XGBoost.
   This is the only phase the previous `koltsov3_gb_sweep.py` ran.
2. **MDQN refinement** (optional). For each MDQN epoch: regenerate walks,
   expand every visited state's neighbors, compute Bellman targets
   `d(s) = 1 + min_{t in N(s)} f_theta(t)`, clip to `[0, k]` using the
   first-visit step `k`, and refit XGBoost. Enabled when any
   `--n-epochs-dqn-values > 0`.
3. **Guided beam search** (optional). Use the refined model as a heuristic in a
   torch beam search from a scrambled start state back to the identity.
   Reports whether and at what step the identity was reached. Enabled by
   `--run-beam-search true`.

Files:

- `koltsov3_gb_pipeline.py` — main Python pipeline script (sweep + all three phases)
- `koltsov3_gb_pipeline.sbatch` — example Hyak Slurm batch script
- `smoke_test.sh` — small end-to-end smoke test that exercises all three phases (`bash smoke_test.sh`)
- `setup_env.sh` — one-shot Klone conda env setup (`bash setup_env.sh`)
- `inspect_results.py` — post-sweep diagnostics on a results directory
- `README_koltsov3_gb_pipeline.md` — this file

The W&B API key lives in `Rithikesh/.env` (one directory up). That file is
gitignored and is loaded automatically — see [W&B setup](#wandb-setup) below.

## Why MDQN matters and what changed from the previous sweep

The previous `koltsov3_gb_sweep.py` did only Phase 1: it fit XGBoost to
normalized random-walk depth and stopped. That label is a noisy proxy for true
graph distance — a state visited at step `k` of a length-`L` walk is at distance
**at most** `k`, but typically less. Algorithm 4 fixes this iteratively: once
the model is decent at predicting walk depth, it can be used to refine its own
targets via the Bellman update, which converges toward the true distance under
the `[0, k]` clip.

Phase 3 (beam search) is the actual downstream test: a regressor with a low val
RMSE is only useful if it can guide a search. Phase 3 measures that directly.

The script is otherwise the same as the old sweep — Koltsov3 generator
construction, random-walk generation (`simple` and `non-backtracking-beam`),
the hand-crafted feature set, per-iteration CSV logging, optional W&B, optional
exact BFS metadata. The defaults match the old behavior: Phase 2 is on with a
small MDQN budget; Phase 3 is on with one scramble per config.

## How the three phases interact

Phase 1 → Phase 2 → Phase 3 happens **per configuration**. Each phase consumes
the model from the previous one:

- Phase 2 starts from Phase 1's best XGBoost booster and rebuilds it from
  scratch each MDQN epoch with the relabeled targets.
- Phase 3 uses the *final* model (post-MDQN if Phase 2 ran, post-warmup
  otherwise) as the beam-search heuristic.

The val/test metrics are recorded twice when MDQN runs: once with the warm-up
model (`val_rmse`, `test_rmse`, ...) and once with the post-MDQN model
(`post_mdqn_val_rmse`, `post_mdqn_test_rmse`, ...). The val/test labels are
still "normalized RW depth", so a small *increase* in `post_mdqn_*_rmse` is
expected and is not a failure — MDQN moves predictions toward true distance,
which is generally not equal to RW depth. The real Phase 2 success signal is
whether `beam_found_rate` and `beam_mean_steps` improve.

## Features

The feature set is ported from Junaid's LightGBM work on this same Koltsov3
problem. For each permutation state, `extract_features()` computes ~13 groups:
displacement, parity mismatch, adjacent pairs, inversions, descents, per-generator
features (I, K, S), theoretical lower bounds, inverse permutation, longest correct
run, signed displacement, raw positions. Every run writes
`feature_importance/importance_<run>.csv` and a PNG; the summary CSV stores
`top_features_by_gain`.

## CLI arguments

Each sweep argument accepts a comma-separated list; the script runs every
combination.

### Data / random-walk axes (Phase 1)

- `--n-values` — permutation sizes, e.g. `8,12,16,24`
- `--n-random-walks-values` — number of training random walks (Cartesian-swept)
- `--walks-per-n` — optional per-n walk override, e.g.
  `5:500,16:25000,24:25000`. Listed `n` use the mapped walk count; unlisted
  `n` fall back to `--n-random-walks-values`.
- `--walk-length-multipliers` — walk length is `n * multiplier`
- `--random-walk-types` — `simple` (uniform random) or `non-backtracking-beam`.
  **For training, use `simple`** — beam-search walks are the inference-time
  algorithm and give bad coverage as a training-data source.
- `--steps-back-to-ban-values` — used only by `non-backtracking-beam`
- `--dedup-strategy` — `none` or `first-visit`. **MDQN requires `first-visit`**
  so the `[0, k]` clip in Algorithm 4 line 7 has a well-defined `k`.
- `--n-val-samples-values` / `--n-test-samples-values` — validation/test walk counts
- `--seed-values` — random seeds

### XGBoost hyperparameter axes (Phase 1)

`--n-estimators-values`, `--max-depth-values`, `--learning-rate-values`,
`--subsample-values`, `--colsample-bytree-values`, `--min-child-weight-values`,
`--reg-lambda-values`, `--reg-alpha-values`.

### Training controls

`--early-stopping-rounds` (default 50), `--max-train-samples` (default 500k),
`--nthread` (-1 = all cores), `--verbose-eval`, `--top-features-to-record`,
`--val-metric-every`.

### Phase 2 — MDQN axes

- `--n-epochs-dqn-values` — MDQN epochs. `0` disables Phase 2.
- `--dqn-n-random-walks-values` — walks regenerated per MDQN epoch.
- `--dqn-clip-values` — comma-separated `true`/`false` for the `[0, k]` clip.
- `--verbose-mdqn` — print per-epoch lines (default on).

### Phase 3 — beam-search axes

- `--run-beam-search` — master switch (default `false`).
- `--beam-width-values` — beam widths.
- `--n-steps-limit-mult-values` — beam step limit is `mult * n^2`.
- `--beam-steps-back-to-ban-values` — recent-state ban depth for the beam.
- `--n-scrambles-values` — independent beam rollouts per config, each from a
  freshly scrambled start.
- `--verbose-beam` — print per-rollout lines (default on).

### Device / problem / output

`--device auto|cpu|cuda` (XGBoost `hist` runs well on CPU),
`--koltsov3-k`, `--output-dir`, `--compute-bfs-metadata`, `--max-bfs-states`.

### W&B

`--use-wandb`, `--wandb-entity`, `--wandb-project`, `--wandb-group`.

## W&B setup

The script loads `WANDB_API_KEY` from, in order:

1. the existing environment variable, if set
2. `Rithikesh/.env` (one directory up from this folder) — **gitignored**
3. `.env` in the current working directory

So with `Rithikesh/.env` present, `--use-wandb true` just works.

> Security note: if the API key is ever exposed, rotate it at <https://wandb.ai/settings>.

## Small smoke test

```bash
bash smoke_test.sh
```

Runs 4 tiny configurations (n=5,6 × max_depth=3,5) with simple walks, first-visit
dedup, **3 MDQN epochs**, and a **single beam rollout** at beam_width 64. Finishes in
seconds, no W&B. Output goes to `smoke_test/`. Performance numbers will be poor —
this only verifies all three phases wire up end-to-end.

```bash
python inspect_results.py smoke_test
```

prints final per-config metrics.

## Hyak (Klone) environment setup

`coenv/python/*` on Klone is built without `libffi`, so torch/xgboost/wandb can't
import. We therefore use a conda env on gscratch. Run once:

```bash
bash /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_pipeline/setup_env.sh
```

Safe to re-run. To use the env in a new shell:

```bash
source /gscratch/stf/rmuddana/miniconda3/etc/profile.d/conda.sh
conda activate cayley
```

## Hyak batch job

1. Make sure the repo is on Klone at `/gscratch/stf/rmuddana/cayley-py` and the
   `cayley` env exists (`setup_env.sh`).

2. The `.sbatch` is filled in (`account=stf`, `partition=ckpt`, conda env on
   gscratch, CPU-only, 6h time limit).

3. Submit:

   ```bash
   cd /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_pipeline
   sbatch koltsov3_gb_pipeline.sbatch
   ```

4. Watch:

   ```bash
   tail -f logs/koltsov3_gb_pipeline_<JOBID>.out
   squeue -u rmuddana
   ```

## Output files

Inside `--output-dir`:

- `summary_results.csv` — one row per completed configuration (Phase 1 + 2 + 3 columns)
- `iteration_results.csv` — one row per Phase-1 boosting iteration per config
- `mdqn_results.csv` — one row per MDQN epoch per config (only when Phase 2 ran)
- `summary_results_partial.csv` / `iteration_results_partial.csv` / `mdqn_results_partial.csv` — incremental, written after each config

### Resuming a preempted run

The `ckpt` partition can preempt the job. To pick up where it left off, point
`--output-dir` at the existing run directory and add `--resume`:

```bash
python koltsov3_gb_pipeline.py --output-dir koltsov3_gb_pipeline_<JOBID> --resume ...same other args...
```

Resume loads `summary_results_partial.csv`, treats each row whose `test_rmse`
is non-null as a finished config (keyed by `config_id`, which equals
`SweepConfig.run_name`), skips those in the main loop, and appends new rows to
the same partial CSVs. The final non-partial CSVs are written once everything
finishes. Pass the *same* sweep flags as the original submission so the
configs match.
- `run_args.json` — command-line arguments
- `plots/rmse_by_config.png`, `plots/spearman_by_config.png` — per-config overviews
- `plots/rmse_vs_max_depth_*.png` — depth trend plots when `max_depth` varies
- `plots/rmse_by_iteration_*.png` — per-boosting-iteration curves (up to 25 configs)
- `feature_importance/importance_*.csv` and `*.png` — gain importance per config

## Tracked metrics (summary CSV)

### Phase 1 (warm-up)

`train_rmse`/`val_rmse`/`test_rmse`, `train_r2`/`val_r2`/`test_r2`,
`train_spearman`/`val_spearman`/`test_spearman`, `best_iteration`,
`n_boosted_rounds`, `final_train_rmse`, `final_val_rmse`,
`best_val_rmse_during_training`, gaps, dataset sizes, label stats,
`top_features_by_gain`, timing.

### Phase 2 (MDQN)

`mdqn_ran`, `mdqn_time_sec`, `mdqn_final_train_rmse`, plus post-MDQN metrics:
`post_mdqn_{train,val,test}_{rmse,r2,spearman}`. Per-epoch detail in
`mdqn_results.csv`.

### Phase 3 (beam)

`beam_ran`, `beam_time_sec`, `beam_n_runs`, `beam_found_rate`,
`beam_mean_steps`, `beam_median_steps`, `beam_min_steps`, `beam_max_steps`,
`beam_n_steps_limit`, `beam_scramble_steps`, `beam_per_run_json` (full per-run
detail as JSON).

### Optional BFS

`diameter`, `last_layer_count`, `layer_sizes`.

## Scaling notes for large n

- The full graph has up to `n!` states.
- Phase 1 dataset size is roughly `n_random_walks * walk_length_multiplier * n`.
- Phase 2 dataset size per epoch is roughly `dqn_n_random_walks *
  walk_length_multiplier * n` after dedup; the neighbor-expansion step
  multiplies the per-epoch prediction call by `n_generators`.
- Phase 3 cost per beam step is one feature-extraction pass + one XGBoost
  prediction on `beam_width * n_generators` candidate states. With XGBoost on
  CPU, that dominates wall time for wide beams.
- `--max-train-samples` caps Phase 1 training rows.
- Keep `--compute-bfs-metadata false` for larger `n`.
