#!/usr/bin/env bash
# Shared AWS / Docker environment for GNNPlus training jobs.
# Source from other scripts:  source "$(dirname "$0")/common_env.sh"

set -euo pipefail

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [task=${TASK_ID:-local}] $1"
}

# --- Repo root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GNNPLUS_PROJECT_ROOT="${GNNPLUS_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${GNNPLUS_PROJECT_ROOT}"
export PYTHONPATH="${GNNPLUS_PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Persistent storage (mount EBS at /data on EC2) ---
export GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-/data/datasets}"
export GNNPLUS_RESULTS_DIR="${GNNPLUS_RESULTS_DIR:-/data/results}"
mkdir -p "${GNNPLUS_DATASET_DIR}" "${GNNPLUS_RESULTS_DIR}" logs logs_gnnplus

# --- Weights & Biases (set WANDB_API_KEY before running) ---
if [ -z "${WANDB_API_KEY:-}" ]; then
    log_message "WARNING: WANDB_API_KEY is not set — W&B logging will fail if wandb.use True"
fi
export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export WANDB_DIR="${WANDB_DIR:-/data/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
export WANDB_DISABLE_CODE=true
export WANDB_SYNC_MODE="${WANDB_SYNC_MODE:-now}"
mkdir -p "${WANDB_DIR}"

export PYTHONNOUSERSITE=1
export PIP_USER=0

# Results land under mounted volume when out_dir is relative
export OUT_DIR_BASE="${GNNPLUS_RESULTS_DIR}"

# Editable install when running from a bind-mounted repo (dev on EC2)
python -m pip install -e . --no-deps --quiet 2>/dev/null || true

python -c "
import GNNPlus
import torch
print('GNNPlus OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
" || {
    log_message "GNNPlus import check failed"
    exit 1
}
