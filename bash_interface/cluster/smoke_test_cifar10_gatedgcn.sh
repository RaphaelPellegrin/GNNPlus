#!/usr/bin/env bash
# =============================================================================
# Smoke test: single GatedGCN run on CIFAR10 with W&B logging.
# Good first job after create_gnnplus_env.sh.
#
# Usage (login node):
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets   # optional
#   sbatch bash_interface/cluster/smoke_test_cifar10_gatedgcn.sh
# =============================================================================

#SBATCH --job-name=gnnplus_smoke
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs_gnnplus/smoke_cifar10_gatedgcn_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

# sbatch copies this script to /var/slurmd/spool/... — use submit dir, not BASH_SOURCE.
REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

REPEAT="${REPEAT:-1}"
SEED="${SEED:-0}"
MAX_EPOCH="${MAX_EPOCH:-5}"

log_message "Smoke test: gatedgcn / cifar10 / repeat=${REPEAT} / max_epoch=${MAX_EPOCH}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

python main.py \
    --cfg configs/gatedgcn/cifar10.yaml \
    --repeat "${REPEAT}" \
    seed "${SEED}" \
    optim.max_epoch "${MAX_EPOCH}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "smoke_cifar10_gatedgcn_r${REPEAT}_ep${MAX_EPOCH}" \
    "${extra_args[@]}"

log_message "Smoke test finished OK"
