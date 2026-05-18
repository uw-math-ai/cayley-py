#!/bin/bash
#SBATCH --job-name=lrx_cmp_n16_ppo_dqn
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "/mmfs1/gscratch/stf/zengrf/cayley-py"
mkdir -p logs reports

if [[ ! -f "Michael/.venv/bin/activate" ]]; then
  echo "Missing Michael/.venv. Create it before submitting this job." >&2
  exit 1
fi

source "Michael/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${PPO_CKPT:?Set PPO_CKPT to the longer-run PPO checkpoint path}"
: "${DQN_CKPT:?Set DQN_CKPT to the DQN baseline checkpoint path}"

report_json="${REPORT_JSON:-reports/ppo_dqn_compare_${SLURM_JOB_ID}.json}"
report_csv="${REPORT_CSV:-reports/ppo_dqn_compare_${SLURM_JOB_ID}.csv}"
previous_ppo_ckpt="${PPO_PREV_CKPT:-checkpoints/lrx_ppo_n16_best_35329717.pt}"

model_args=(
  --model "ppo_long=${PPO_CKPT}"
  --model "dqn=${DQN_CKPT}"
)

if [[ -f "$previous_ppo_ckpt" ]]; then
  model_args+=(--model "ppo_prev=${previous_ppo_ckpt}")
fi

python Michael/compare_lrx_koltsov3_models.py \
  "${model_args[@]}" \
  --profiles heuristic_only checkpoint_default \
  --beam-width 256 \
  --step-limit 256 \
  --history-size 32 \
  --apply-x-trick \
  --holdout-num-states 8 \
  --holdout-walk-length 128 \
  --holdout-seed 999 \
  --holdout-history-size 32 \
  --output-json "$report_json" \
  --output-csv "$report_csv" \
  --device auto

echo "Wrote comparison report to $report_json"
echo "Wrote comparison summary CSV to $report_csv"