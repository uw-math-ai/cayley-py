#!/bin/bash
#SBATCH --job-name=lrx_ppo_n16_ckpt_longcmp
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "/mmfs1/gscratch/stf/zengrf/cayley-py"
mkdir -p logs checkpoints

if [[ ! -f "Michael/.venv/bin/activate" ]]; then
  echo "Missing Michael/.venv. Create it before submitting this job." >&2
  exit 1
fi

source "Michael/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python --version
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

checkpoint_path="checkpoints/lrx_ppo_n16_best_${SLURM_JOB_ID}.pt"

python Michael/lrx_koltsov3_ppo.py \
  --n 16 \
  --total-steps 524288 \
  --rollout-steps 1024 \
  --hidden-dim 512 \
  --policy-lr 3e-4 \
  --value-lr 1e-3 \
  --value-coef 0.5 \
  --heuristic-coef 1.0 \
  --entropy-coef 0.01 \
  --clip-coef 0.2 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --grad-clip-norm 0.5 \
  --target-kl 0.02 \
  --anneal-learning-rate \
  --success-bonus 5.0 \
  --step-penalty 1.0 \
  --warmup-epochs 30 \
  --warmup-walks 4096 \
  --warmup-walk-length 128 \
  --warmup-batch-size 1024 \
  --warmup-history-size 32 \
  --bellman-epochs 20 \
  --bellman-batch-size 1024 \
  --policy-warmstart-epochs 10 \
  --policy-warmstart-batch-size 1024 \
  --policy-head-lr 3e-4 \
  --eval-every-updates 16 \
  --eval-beam-width 256 \
  --eval-step-limit 256 \
  --eval-history-size 32 \
  --eval-policy-alpha 0.5 \
  --eval-apply-x-trick \
  --checkpoint-path "$checkpoint_path" \
  --log-every-updates 8 \
  --device auto