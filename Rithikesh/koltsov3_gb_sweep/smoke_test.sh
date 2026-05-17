#!/bin/bash
# Quick smoke test for the Koltsov3 GB sweep -- tiny config, runs in seconds.
# Verifies the script runs end-to-end in this environment before submitting a
# real sbatch job. Does NOT use W&B.
#
# Usage (from anywhere on Klone, with the `cayley` conda env active):
#   bash /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_sweep/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Running smoke test from ${SCRIPT_DIR}"
echo

python koltsov3_gb_sweep.py \
  --n-values 5,6 \
  --n-random-walks-values 50 \
  --walk-length-multipliers 4 \
  --random-walk-types simple \
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

echo
echo "=============================================="
echo " Smoke test finished. Check smoke_test/ for:"
echo "   summary_results.csv, iteration_results.csv,"
echo "   plots/, feature_importance/"
echo "=============================================="
