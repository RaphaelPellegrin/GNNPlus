#!/usr/bin/env bash
# CIFAR10 paper baselines on AWS: gcn / gine / gatedgcn × 2 seeds.
# Mirrors bash_interface/cluster/cifar10_paper_baselines.sh (without SLURM).
#
# Run all 6 tasks on one instance (sequential):
#   export WANDB_API_KEY=...
#   bash bash_interface/aws/cifar10_paper_baselines.sh
#
# Run one task per instance (parallel — launch 6 EC2 nodes):
#   TASK_ID=1 bash bash_interface/aws/cifar10_paper_baselines.sh   # gcn seed 0
#   TASK_ID=6 bash bash_interface/aws/cifar10_paper_baselines.sh   # gatedgcn seed 1
#
# Docker:
#   docker run --gpus all --rm -v /data/gnnplus:/data -e WANDB_API_KEY=... \
#     -e TASK_ID=1 gnnplus:gpu bash bash_interface/aws/cifar10_paper_baselines.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

GNN_LIST=(gcn gine gatedgcn)
NUM_GNN=${#GNN_LIST[@]}
NUM_SEEDS=2
NUM_TASKS=$((NUM_GNN * NUM_SEEDS))

run_task() {
    local task_id="$1"
    if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${NUM_TASKS}" ]; then
        log_message "task_id=${task_id} out of range (1-${NUM_TASKS})"
        exit 1
    fi

    local idx=$((task_id - 1))
    local gnn_idx=$((idx / NUM_SEEDS))
    local seed_idx=$((idx % NUM_SEEDS))
    local gnn="${GNN_LIST[$gnn_idx]}"
    local seed="${seed_idx}"

    log_message "Task ${task_id}: ${gnn} / cifar10 / seed=${seed}"

    local extra_args=()
    if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
        extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
    fi
    extra_args+=(out_dir "${GNNPLUS_RESULTS_DIR}")

    python main.py \
        --cfg "configs/${gnn}/cifar10.yaml" \
        --repeat 1 \
        seed "${seed}" \
        wandb.use True \
        wandb.entity "${WANDB_ENTITY}" \
        wandb.project "${WANDB_PROJECT}" \
        wandb.name "cifar10_${gnn}_seed${seed}_aws" \
        "${extra_args[@]}"

    log_message "Finished ${gnn} seed=${seed}"
}

if [ -n "${TASK_ID:-}" ]; then
    run_task "${TASK_ID}"
else
    log_message "Running all ${NUM_TASKS} CIFAR10 paper tasks sequentially"
    for task_id in $(seq 1 "${NUM_TASKS}"); do
        TASK_ID="${task_id}" run_task "${task_id}"
    done
fi
