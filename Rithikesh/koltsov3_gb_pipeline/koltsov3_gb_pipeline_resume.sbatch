#!/bin/bash
#SBATCH --job-name=koltsov3_gb_pipeline_resume
#SBATCH --account=stf
#SBATCH --partition=ckpt                # checkpoint partition: thousands of idle CPUs, but jobs can be
                                        # preempted. For a guaranteed (non-preemptible) slot use: compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # XGBoost 'hist' is CPU-parallel; more cores = faster training
#SBATCH --mem=32G
#SBATCH --time=24:00:00                 # 160-config grid; ckpt may preempt, partial CSVs written each config.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This job runs the Koltsov3 random-walk XGBoost pipeline:
#   Phase 1 (warm-up): diffusion-distance XGBoost
#   Phase 2 (MDQN): Algorithm 4 Bellman refinement (optional)
#   Phase 3 (beam): guided beam search rollouts (optional)
# XGBoost 'hist' runs well on CPU, so no GPU is requested by default.

set -euo pipefail

# Run from the directory where you submitted the job.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Environment setup for Klone. We use a conda env on gscratch:
#   - the coenv/python modules are built without libffi (import ctypes fails),
#     so a conda Python is required instead
#   - the home-directory quota is too small for conda + torch + CUDA deps
# Run Rithikesh/koltsov3_gb_pipeline/setup_env.sh once to create this env.
export PYTHONNOUSERSITE=1   # ignore ~/.local packages; keep the conda env isolated
source /gscratch/stf/rmuddana/miniconda3/etc/profile.d/conda.sh
conda activate cayley

# W&B credentials: the pipeline loads WANDB_API_KEY from Rithikesh/.env
# automatically (that file is gitignored). If you used `wandb login` once on
# Klone instead, that also works -- the script falls back to the cached login.

# Sweep grid matches Merav's MLP best-config data settings so Phase 1 metrics
# are directly comparable: n = 8, 16, 24, 32, 64 (her three benchmark n's plus
# two of ours); n_random_walks = 500, 1000 (covers her 500/1000); walk_length
# multipliers 4, 8 (covers her 4/8); simple walks, ban=0. We then add MDQN
# (Phase 2) and beam search (Phase 3) on top -- her work stops at Phase 1.
#   5 n's x 2 walk counts x 2 mults x 4 max_depths x 2 lrs = 160 configurations.
python koltsov3_gb_pipeline.py \
  --n-values 8,16,24,32,64 \
  --n-random-walks-values 500,1000 \
  --walk-length-multipliers 4,8 \
  --walk-length-mode multiplier \
  --random-walk-types simple \
  --steps-back-to-ban-values 0 \
  --n-estimators-values 1000 \
  --max-depth-values 6,8,10,12 \
  --learning-rate-values 0.05,0.1 \
  --subsample-values 0.8 \
  --colsample-bytree-values 0.8 \
  --min-child-weight-values 5 \
  --reg-lambda-values 1.0 \
  --reg-alpha-values 0.0 \
  --n-val-samples-values 300 \
  --n-test-samples-values 300 \
  --seed-values 0 \
  --dedup-strategy first-visit \
  --early-stopping-rounds 50 \
  --n-epochs-dqn-values 50 \
  --dqn-n-random-walks-values 2000 \
  --dqn-clip-values true \
  --run-beam-search true \
  --beam-width-values 65536 \
  --n-steps-limit-mult-values 4 \
  --beam-steps-back-to-ban-values 32 \
  --n-scrambles-values 3 \
  --beam-scramble-depth-mult-values 100 \
  --device auto \
  --output-dir koltsov3_gb_pipeline_35341286 \
  --resume \
  --compute-bfs-metadata true \
  --use-wandb true \
  --wandb-entity CayleyPy \
  --wandb-project cayley-py \
  --wandb-group koltsov3_gb_pipeline_35341286

# To run Phase 1 only (matches the original sweep behavior):
#   --n-epochs-dqn-values 0 --run-beam-search false
