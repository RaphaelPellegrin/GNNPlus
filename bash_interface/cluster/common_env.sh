#!/usr/bin/env bash
# Shared Harvard FASRC / mweber_gpu setup for GNNPlus cluster jobs.
# Source from SLURM scripts:  source "$(dirname "$0")/common_env.sh"

set -euo pipefail

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task ${SLURM_ARRAY_TASK_ID:-local}] $1"
}

# --- Weights & Biases (Harvard GeoML team) ---
export WANDB_API_KEY="${WANDB_API_KEY:-ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0}"
export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
_wandb_job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"
_wandb_task="${SLURM_ARRAY_TASK_ID:-local}"
WANDB_TMP_DIR="${TMPDIR:-/tmp}/wandb_${_wandb_job}_${_wandb_task}"
export WANDB_DIR="${WANDB_TMP_DIR}"
export WANDB_CACHE_DIR="${WANDB_TMP_DIR}/.cache"
export WANDB_DISABLE_CODE=true
export WANDB_SYNC_MODE="now"
mkdir -p "${WANDB_TMP_DIR}" logs logs_gnnplus
export PYTHONNOUSERSITE=1
export PIP_USER=0
if command -v module &> /dev/null; then
    module load cuda/12.9.1-fasrc01 2>/dev/null || module load cuda 2>/dev/null || true
    module load python/3.10.12-fasrc01 2>/dev/null || true
fi

# --- Conda env (dedicated GNNPlus env on lab storage) ---
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs}"
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
fi
ENV_NAME="${ENV_NAME:-gnnplus}"
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ]; then
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
else
    log_message "WARNING: conda env '${ENV_NAME}' not found at ${CONDA_ENVS_PATH}/${ENV_NAME}"
    log_message "Run: bash bash_interface/cluster/create_gnnplus_env.sh (interactive salloc first)"
fi

# --- Repo root ---
PROJECT_ROOT="${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
cd "$PROJECT_ROOT" || {
    log_message "Cannot cd to PROJECT_ROOT=${PROJECT_ROOT}"
    exit 1
}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Optional persistent dataset cache (recommended on cluster)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_DATASET_DIR}"
fi

# Editable install (no model changes; ensures imports resolve)
python -m pip install -e . --no-deps --quiet 2>/dev/null || true
python -c "
import GNNPlus
from GNNPlus.network.custom_gnn import CustomGNN
import torch
prefix = '${CONDA_ENVS_PATH}/${ENV_NAME}'
assert torch.__file__.startswith(prefix), f'torch not in env: {torch.__file__}'
print('GNNPlus + torch import OK from', prefix)
" || {
    log_message "GNNPlus import check failed (is gnnplus env installed in holylabs, not ~/.local?)"
    exit 1
}
