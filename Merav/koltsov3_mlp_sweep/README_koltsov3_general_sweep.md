# Koltsov3 General Random-Walk MLP Sweep

This folder contains a cleaner, more general experiment script for running Koltsov3 random-walk MLP sweeps on UW Hyak with Slurm.

Generated files:

- `koltsov3_general_sweep.py`: main Python experiment script
- `koltsov3_general_sweep.sbatch`: example Hyak Slurm batch script

## What the script does

The script trains a small MLP to predict normalized random-walk depth labels for states generated from Koltsov3 random walks.

Main logic:

- construct Koltsov3 generators for permutations of length `n`
- generate validation and test random-walk datasets for each configuration
- regenerate training random walks at every epoch
- train one MLP per sweep configuration
- evaluate train/validation/test RMSE, R², and Spearman correlation
- optionally log runs to W&B
- save CSV results and plots

The main change is that the script now runs a Cartesian-product sweep over many command-line arguments, not only `n` and width.

## Important arguments

### Sweep arguments

Each of these accepts a comma-separated list. The script runs every combination.

- `--n-values`: permutation sizes, for example `5,6,7,8`
- `--widths`: hidden layer widths, for example `32,64,128`
- `--n-random-walks-values`: number of training random walks generated each epoch
- `--walk-length-multipliers`: walk length is `n * walk_length_multiplier`
- `--random-walk-types`: currently supports `simple` and `non-backtracking-beam`
- `--steps-back-to-ban-values`: for `non-backtracking-beam`, how many previous move choices to ban
- `--n-epochs-values`: number of epochs per model
- `--lr-values`: Adam learning rates
- `--batch-size-values`: mini-batch sizes
- `--n-val-samples-values`: validation random-walk counts
- `--n-test-samples-values`: test random-walk counts
- `--seed-values`: random seeds

Example:

```bash
python koltsov3_general_sweep.py \
  --n-values 5,6,7,8 \
  --widths 32,64,128 \
  --n-random-walks-values 500,1000,2000 \
  --walk-length-multipliers 4,8,12 \
  --steps-back-to-ban-values 0,1,2
```

This runs every combination of those values, so the number of jobs grows quickly.

### Device arguments

- `--device auto`: use CUDA if available, otherwise CPU
- `--device cuda`: require CUDA
- `--device cpu`: force CPU

For Hyak GPU jobs, `--device auto` is usually fine.

### BFS metadata arguments

Exact BFS metadata is disabled by default because it can scale like `n!`.

- `--compute-bfs-metadata false`: default and recommended for larger sweeps
- `--compute-bfs-metadata true`: attempt exact BFS metadata
- `--max-bfs-states 50000`: safety cap; BFS is skipped if `n!` exceeds this

When computed safely, the script adds:

- `diameter`
- `last_layer_count`
- `layer_sizes`

For larger `n`, leave BFS metadata disabled.

### W&B arguments

- `--use-wandb true/false`: enable or disable W&B
- `--wandb-entity CayleyPy`: W&B entity/team
- `--wandb-project cayley-py`: W&B project
- `--wandb-group NAME`: group name for the whole sweep
- `--wandb-login true`: optionally call `wandb.login()` at the start

Each configuration becomes one W&B run. The run name includes:

- `n`
- width
- training random-walk count
- walk-length multiplier
- random-walk type
- steps back to ban
- epochs
- learning rate
- batch size
- seed

The script uses `resume="never"`, so it should not overwrite previous W&B runs.

## Small smoke test

Run this first on a login node or short interactive session to check that imports, paths, and the environment work:

```bash
python koltsov3_general_sweep.py \
  --n-values 5 \
  --widths 8 \
  --n-random-walks-values 20 \
  --walk-length-multipliers 4 \
  --random-walk-types non-backtracking-beam \
  --steps-back-to-ban-values 2 \
  --n-epochs-values 2 \
  --lr-values 0.001 \
  --batch-size-values 8 \
  --n-val-samples-values 10 \
  --n-test-samples-values 10 \
  --seed-values 0 \
  --output-dir smoke_test \
  --device auto \
  --use-wandb false
```

Expected outputs appear in `smoke_test/`.

## Medium Hyak batch job

1. Put these files in the same directory:

   ```text
   koltsov3_general_sweep.py
   koltsov3_general_sweep.sbatch
   README.md
   ```

2. Edit the `.sbatch` file:

   - change `--account=stf` if needed
   - change the partition if needed
   - update the virtual environment path:

     ```bash
     source /path/to/your/venv/bin/activate
     ```

3. Submit:

   ```bash
   sbatch koltsov3_general_sweep.sbatch
   ```

4. Watch logs:

   ```bash
   tail -f logs/koltsov3_sweep_<JOBID>.out
   ```

## Running with W&B

First make sure you are logged in on Hyak:

```bash
wandb login
```

Then run:

```bash
python koltsov3_general_sweep.py \
  --n-values 5,6,7 \
  --widths 32,64 \
  --n-random-walks-values 500 \
  --walk-length-multipliers 8 \
  --random-walk-types non-backtracking-beam \
  --steps-back-to-ban-values 2 \
  --n-epochs-values 25 \
  --lr-values 0.001 \
  --batch-size-values 64 \
  --n-val-samples-values 300 \
  --n-test-samples-values 300 \
  --seed-values 0 \
  --output-dir wandb_test \
  --use-wandb true \
  --wandb-entity CayleyPy \
  --wandb-project cayley-py \
  --wandb-group koltsov3_wandb_test
```

## Output files

Inside `--output-dir`, the script writes:

- `summary_results.csv`: one row per completed configuration
- `epoch_results.csv`: one row per epoch per configuration
- `summary_results_partial.csv`: incrementally updated partial summary
- `epoch_results_partial.csv`: incrementally updated partial epoch results
- `run_args.json`: command-line arguments used for the run
- `plots/rmse_by_config.png`: train/val/test RMSE overview
- `plots/spearman_by_config.png`: train/val/test Spearman overview
- `plots/rmse_vs_width_*.png`: width trend plots when width varies
- `plots/spearman_vs_width_*.png`: width trend plots when width varies
- `plots/val_rmse_by_epoch_*.png`: epoch plots for up to 25 configs
- `plots/val_spearman_by_epoch_*.png`: epoch plots for up to 25 configs

## Tracked metrics

### Dataset size and uniqueness

- `n_train_states`: number of training states in the final epoch
- `n_val_states`: number of validation states
- `n_test_states`: number of test states
- `num_unique_train_states`: unique training states in the final epoch
- `num_unique_val_states`: unique validation states
- `num_unique_test_states`: unique test states
- `unique_train_fraction`: `num_unique_train_states / n_train_states`
- `unique_val_fraction`: `num_unique_val_states / n_val_states`
- `unique_test_fraction`: `num_unique_test_states / n_test_states`

These help you see whether random walks are producing diverse states or repeatedly sampling the same states.

### Label statistics

- `label_min`
- `label_max`
- `label_mean`
- `label_std`

Labels are normalized random-walk depths, so they typically range from near `0` to `1`.

### Model diagnostics

- `num_parameters`: trainable parameter count
- `layer_sizes`: hidden layer sizes as JSON, currently a single width like `[64]`

### Performance metrics

- `train_rmse`, `val_rmse`, `test_rmse`
- `train_r2`, `val_r2`, `test_r2`
- `train_spearman`, `val_spearman`, `test_spearman`

Spearman is useful because it measures whether the model ranks states by depth well, even if exact numeric predictions are imperfect.

### Gap metrics

- `train_val_rmse_gap = val_rmse - train_rmse`
- `train_test_rmse_gap = test_rmse - train_rmse`
- `val_test_rmse_gap = test_rmse - val_rmse`

Large train/test or train/validation gaps may indicate overfitting or distribution mismatch.

### Best epoch metrics

- `best_epoch_by_val_rmse`: epoch with lowest validation RMSE
- `best_epoch_by_val_spearman`: epoch with highest validation Spearman
- `best_val_rmse_during_training`
- `best_val_spearman_during_training`

### Timing metrics

- `fit_time_sec`: total training time for the configuration
- `predict_time_sec`: final train/validation/test prediction time

## Scaling warnings for large n

Increasing `n` can become expensive quickly.

Important scaling factors:

- The full graph has up to `n!` permutation states.
- Random-walk dataset size is roughly `num_random_walks * walk_length_multiplier * n`.
- Sweeping many comma-separated values creates a Cartesian product.
- `non-backtracking-beam` currently samples allowed moves row-by-row, which is easy to read but may become slow for very large random-walk counts.
- Exact BFS metadata should stay disabled for larger `n`.

Before running a large sweep, estimate the number of configurations:

```text
len(n-values)
* len(widths)
* len(n-random-walks-values)
* len(walk-length-multipliers)
* len(random-walk-types)
* len(steps-back-to-ban-values)
* len(n-epochs-values)
* len(lr-values)
* len(batch-size-values)
* len(n-val-samples-values)
* len(n-test-samples-values)
* len(seed-values)
```

Start with one small smoke test, then scale gradually.

## Notes on changes from the original script

The Koltsov3 generator logic and random-walk state update logic are preserved.

The main experiment behavior is also preserved: validation/test data are generated once per configuration, while training random walks are regenerated each epoch.

The main intentional changes are:

1. The sweep now runs every combination of many parameters, not only `n` and width.
2. The output CSVs use generic names: `summary_results.csv` and `epoch_results.csv`.
3. The script writes partial CSVs after each completed configuration to reduce the risk of losing all results from a long Slurm job.
4. BFS metadata is optional and guarded by `--max-bfs-states` because exact BFS is unsafe for large `n`.
5. Plots are summarized by configuration index and by width where possible, because a high-dimensional sweep cannot always be cleanly represented by a single width plot.
