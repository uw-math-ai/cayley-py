#!/bin/bash
#SBATCH --job-name=koltsov3_gb_sweep
#SBATCH --account=stf
#SBATCH --partition=ckpt                # checkpoint partition: thousands of idle CPUs, but jobs can be
                                        # preempted. For a guaranteed (non-preemptible) slot use: compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # XGBoost 'hist' is CPU-parallel; more cores = faster training
#SBATCH --mem=32G
#SBATCH --time=04:00:00                 # ckpt jobs may be preempted; the script writes partial CSVs so
                                        # progress is not lost. Bump --time if a sweep needs longer.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This job runs the Koltsov3 random-walk XGBoost (gradient boosting) sweep.
# It is the gradient boosting counterpart to Merav's MLP sweep .sbatch.
# XGBoost 'hist' runs well on CPU, so no GPU is requested by default. If you
# want GPU, add `#SBATCH --gres=gpu:1` above and pass `--device cuda`.

set -euo pipefail

# Run from the directory where you submitted the job.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Environment setup for Klone. We use a conda env on gscratch:
#   - the coenv/python modules are built without libffi (import ctypes fails),
#     so a conda Python is required instead
#   - the home-directory quota is too small for conda + torch + CUDA deps
# Run Rithikesh/koltsov3_gb_sweep/setup_env.sh once to create this env.
export PYTHONNOUSERSITE=1   # ignore ~/.local packages; keep the conda env isolated
source /gscratch/stf/rmuddana/miniconda3/etc/profile.d/conda.sh
conda activate cayley

# W&B credentials: the sweep script loads WANDB_API_KEY from Rithikesh/.env
# automatically (that file is gitignored). If you used `wandb login` once on
# Klone instead, that also works -- the script falls back to the cached login.

# Sweep: n = 5,6,7,8 (Merav's MLP n's, for direct GB-vs-MLP comparison)
#        + 12,16,24 (larger n, our extensions)
# max_depth = 4,6,8 -> 7 n x 3 depth = 21 configs.
# BFS metadata is on: it auto-skips when n! > --max-bfs-states (default 50000),
# so n=5..8 get exact diameters and n>=12 are auto-skipped at zero extra cost.
python koltsov3_gb_sweep.py \
  --n-values 5,6,7,8,12,16,24 \
  --n-random-walks-values 2000 \
  --walk-length-multipliers 8 \
  --random-walk-types non-backtracking-beam \
  --steps-back-to-ban-values 2 \
  --n-estimators-values 600 \
  --max-depth-values 4,6,8 \
  --learning-rate-values 0.05 \
  --subsample-values 0.8 \
  --colsample-bytree-values 0.8 \
  --min-child-weight-values 5 \
  --reg-lambda-values 1.0 \
  --reg-alpha-values 0.0 \
  --n-val-samples-values 300 \
  --n-test-samples-values 300 \
  --seed-values 0 \
  --early-stopping-rounds 50 \
  --device auto \
  --output-dir koltsov3_gb_sweep_${SLURM_JOB_ID} \
  --compute-bfs-metadata true \
  --use-wandb true \
  --wandb-entity CayleyPy \
  --wandb-project cayley-py \
  --wandb-group koltsov3_gb_sweep_${SLURM_JOB_ID}

# To run without W&B, change the last four lines to:
#   --use-wandb false
