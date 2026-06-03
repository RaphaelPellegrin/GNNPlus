#!/usr/bin/env bash
# =============================================================================
# CIFAR10 — GNNPlus paper baselines (gcn / gine / gatedgcn), 2 seeds each.
# Mirrors local run.sh:  sh run.sh <gpu> cifar10 2
#
# Array layout (6 tasks):
#   1-2: gcn      seeds 0,1
#   3-4: gine     seeds 0,1
#   5-6: gatedgcn seeds 0,1
#
# Usage:
#   sbatch bash_interface/cluster/cifar10_paper_baselines.sh
#   sbatch --array=5-6 bash_interface/cluster/cifar10_paper_baselines.sh   # gatedgcn only
# =============================================================================

#SBATCH --job-name=gnnplus_cifar10
#SBATCH --array=1-6%6
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/cifar10_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

GNN_LIST=(gcn gine gatedgcn)
NUM_GNN=${#GNN_LIST[@]}
NUM_SEEDS=2

task_id=${SLURM_ARRAY_TASK_ID:-1}
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt $((NUM_GNN * NUM_SEEDS)) ]; then
    log_message "task_id=${task_id} out of range"
    exit 1
fi

idx=$((task_id - 1))
gnn_idx=$((idx / NUM_SEEDS))
seed_idx=$((idx % NUM_SEEDS))
gnn="${GNN_LIST[$gnn_idx]}"
seed="${seed_idx}"

log_message "Task ${task_id}: ${gnn} / cifar10 / seed=${seed}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

wandb_name="cifar10_${gnn}_seed${seed}_cluster"

python main.py \
    --cfg "configs/${gnn}/cifar10.yaml" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    "${extra_args[@]}"

log_message "Finished ${gnn} seed=${seed}"
