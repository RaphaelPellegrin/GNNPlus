#!/usr/bin/env bash
# Smoke test: hybrid_gnn on MNIST (short run).
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   sbatch bash_interface/cluster/smoke_test_hybrid_mnist.sh

#SBATCH --job-name=gnnplus_hybrid_smoke
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs_gnnplus/smoke_hybrid_mnist_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

REPEAT="${REPEAT:-1}"
MAX_EPOCH="${MAX_EPOCH:-5}"

log_message "Smoke test: hybrid_gnn / mnist / repeat=${REPEAT} / max_epoch=${MAX_EPOCH}"

export WANDB_PROJECT="${WANDB_PROJECT:-MOE_6}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

python main.py \
    --cfg configs/gated_hybrid/mnist.yaml \
    --repeat "${REPEAT}" \
    seed 0 \
    optim.max_epoch "${MAX_EPOCH}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    gnn.hybrid.log_gate_stats True \
    wandb.name "smoke_hybrid_mnist_ep${MAX_EPOCH}_gnnplus" \
    "${extra_args[@]}"

log_message "Smoke test finished OK"
