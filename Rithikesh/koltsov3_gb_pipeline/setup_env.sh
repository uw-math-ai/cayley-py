#!/bin/bash
# One-shot environment setup for the Koltsov3 GB pipeline on Klone (UW Hyak).
#
# Why this script exists:
#   - The `coenv/python/3.11.9` and `3.13.11` modules are built without libffi,
#     so `import ctypes` fails -> torch/xgboost/wandb won't import. Unusable.
#   - Conda ships its own libffi, so a conda Python works.
#   - The home directory quota is too small for conda + torch + CUDA deps, so
#     everything goes on gscratch.
#
# Usage (from anywhere on Klone):
#   bash /gscratch/stf/rmuddana/cayley-py/Rithikesh/koltsov3_gb_pipeline/setup_env.sh
#
# Safe to re-run: it skips steps that are already done.

set -euo pipefail

# Never let packages in ~/.local/lib leak into the conda env. An earlier failed
# install left a corrupted matplotlib there; user-site packages would shadow the
# clean copies installed into the conda env below.
export PYTHONNOUSERSITE=1

# ---- Paths (everything on gscratch, not home) ----
GSCRATCH_BASE="/gscratch/stf/rmuddana"
CONDA_DIR="${GSCRATCH_BASE}/miniconda3"
ENV_NAME="cayley"
REPO_DIR="${GSCRATCH_BASE}/cayley-py"
INSTALLER="${GSCRATCH_BASE}/Miniconda3-latest-Linux-x86_64.sh"

echo "=============================================="
echo " Koltsov3 GB pipeline - environment setup"
echo "=============================================="
echo "conda dir : ${CONDA_DIR}"
echo "env name  : ${ENV_NAME}"
echo "repo dir  : ${REPO_DIR}"
echo

# ---- 1. Install Miniconda on gscratch (skip if already there) ----
if [ -d "${CONDA_DIR}" ]; then
    echo "[1/5] Miniconda already installed at ${CONDA_DIR} - skipping."
else
    echo "[1/5] Downloading Miniconda installer..."
    # Klone needs the http proxy for outbound downloads.
    export https_proxy="${https_proxy:-http://klone-dip1:3128}"
    export http_proxy="${http_proxy:-http://klone-dip1:3128}"
    curl -fsSL -o "${INSTALLER}" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    echo "[1/5] Installing Miniconda to ${CONDA_DIR} (batch mode)..."
    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"
fi

# ---- 2. Activate conda for this shell ----
echo "[2/5] Activating conda..."
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Recent conda requires explicitly accepting the Anaconda channel Terms of
# Service before the default channels can be used. This is idempotent.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ---- 3. Create the cayley env (skip if it exists) ----
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "[3/5] Conda env '${ENV_NAME}' already exists - skipping create."
else
    echo "[3/5] Creating conda env '${ENV_NAME}' with Python 3.11..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi

# ---- 4. Install dependencies into the env ----
echo "[4/5] Installing dependencies into '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# A previous failed install left broken packages in ~/.local/lib/python3.11
# (notably a half-installed matplotlib). Those would shadow the conda env's
# copies, so remove them. This only touches leftover pip --user packages.
if [ -d "${HOME}/.local/lib/python3.11" ]; then
    echo "      Removing stale ~/.local/lib/python3.11 packages..."
    rm -rf "${HOME}/.local/lib/python3.11"
fi

# Keep pip's cache on gscratch, not the small home quota.
export PIP_CACHE_DIR="${GSCRATCH_BASE}/.pip-cache"
# PYTHONNOUSERSITE (set at the top) keeps everything inside the conda env.
python -m pip install --upgrade pip
python -m pip install \
    -r "${REPO_DIR}/requirements.txt" \
    -r "${REPO_DIR}/Rithikesh/requirements.txt"

# ---- 5. Verify the imports that actually matter ----
echo "[5/5] Verifying imports..."
python -c "import ctypes; print('  ctypes OK')"
python -c "import xgboost; print('  xgboost', xgboost.__version__)"
python -c "import dotenv, wandb, matplotlib; print('  dotenv / wandb / matplotlib OK')"
python -c "import numpy, pandas, sklearn, scipy, torch; print('  numpy / pandas / sklearn / scipy / torch OK')"

echo
echo "=============================================="
echo " DONE. Environment is ready."
echo "=============================================="
echo
echo "To use this env in a new shell:"
echo "  source ${CONDA_DIR}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
