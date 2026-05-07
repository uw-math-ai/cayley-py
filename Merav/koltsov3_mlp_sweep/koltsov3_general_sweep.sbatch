#!/bin/bash
#SBATCH --job-name=koltsov3_sweep
#SBATCH --account=stf                    # TODO: change if your Hyak account is different
#SBATCH --partition=gpu-l40s             # TODO: change partition if needed, e.g. gpu-a40, gpu-l40s, compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1                     # TODO: remove this line if running CPU-only
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Run from the directory where you submitted the job.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# TODO: customize this environment setup for your Hyak install.
# Examples:
#   module load python/3.11
#   source /gscratch/stf/YOUR_UWNETID/venvs/cayley/bin/activate
source /path/to/your/venv/bin/activate

# Optional but often useful for PyTorch CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Medium sweep defaults. Start smaller first if you are testing a new environment.
python koltsov3_general_sweep.py \
  --n-values 5,6,7,8 \
  --widths 32,64,128 \
  --n-random-walks-values 500,1000 \
  --walk-length-multipliers 4,8 \
  --random-walk-types non-backtracking-beam \
  --steps-back-to-ban-values 1,2 \
  --n-epochs-values 25 \
  --lr-values 0.001 \
  --batch-size-values 64 \
  --n-val-samples-values 300 \
  --n-test-samples-values 300 \
  --seed-values 0 \
  --device auto \
  --output-dir koltsov3_general_sweep_${SLURM_JOB_ID} \
  --compute-bfs-metadata false \
  --use-wandb false

# To enable W&B, change the last line above to:
#   --use-wandb true \
#   --wandb-entity CayleyPy \
#   --wandb-project cayley-py \
#   --wandb-group koltsov3_general_sweep_${SLURM_JOB_ID}
