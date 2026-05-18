#!/bin/bash
# Quick smoke test for the Koltsov3 GB pipeline -- tiny config, runs in seconds.
# Exercises all three phases (warm-up, MDQN, beam search) so a successful run
# proves the wiring end-to-end. Does NOT use W&B.
#
# Usage (from anywhere on Klone, with the `cayley` conda env active):
#   bash /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_pipeline/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Running smoke test from ${SCRIPT_DIR}"
echo

python koltsov3_gb_pipeline.py \
  --n-values 5,6 \
  --n-random-walks-values 50 \
  --walk-length-multipliers 4 \
  --random-walk-types simple \
  --steps-back-to-ban-values 0 \
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
  --dedup-strategy first-visit \
  --n-epochs-dqn-values 3 \
  --dqn-n-random-walks-values 50 \
  --dqn-clip-values true \
  --run-beam-search true \
  --beam-width-values 64 \
  --n-steps-limit-mult-values 2 \
  --beam-steps-back-to-ban-values 2 \
  --n-scrambles-values 1 \
  --beam-scramble-depth-mult-values 1 \
  --output-dir smoke_test \
  --compute-bfs-metadata true \
  --use-wandb false

echo
echo "=============================================="
echo " Smoke test finished. Check smoke_test/ for:"
echo "   summary_results.csv, iteration_results.csv,"
echo "   mdqn_results.csv, plots/, feature_importance/"
echo "=============================================="
