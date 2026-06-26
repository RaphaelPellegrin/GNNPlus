#!/usr/bin/env bash
# =============================================================================
# CLUSTER hybrid LR × d_h grid on anchor o6owwoqp (a1g1 GATEDGCN, seed 1).
#
# 8 tasks = 2 × hybrid_d_h {64, 128} × 4 log-spaced optim.base_lr
#   {0.0003, 0.000669, 0.001492, 0.003}
#
# Task 1 → d_h=64,  lr=0.0003   (includes near-original o6owwoqp lr 3.36e-4)
# Task 5 → d_h=128, lr=0.0003
#
# Submit:
#   sbatch --job-name=cluster_lr_dh --array=1-8%4 --mem=128GB --time=120:00:00 \
#     --export=ALL,SEED=1,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_cluster_hybrid_o6owwoqp_lr_dh_sweep.sh
# =============================================================================

#SBATCH --job-name=cluster_lr_dh
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=128GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

SEED="${SEED:-1}"
CFG="configs/gated_hybrid/cluster-hybrid-o6owwoqp-anchor.yaml"
task_id=${SLURM_ARRAY_TASK_ID:-1}

DHS=(64 128)
LRS=(0.0003 0.000669 0.001492 0.003)

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt 8 ]; then
    log_message "task_id=${task_id} out of range (expected 1-8)"
    exit 1
fi

dh_idx=$(( (task_id - 1) / 4 ))
lr_idx=$(( (task_id - 1) % 4 ))
d_h="${DHS[$dh_idx]}"
base_lr="${LRS[$lr_idx]}"

lr_tag="${base_lr//./p}"
wandb_name="cluster_hybrid_a1g1_dh${d_h}_lr${lr_tag}_seed${SEED}"

log_message "CLUSTER LR×d_h task ${task_id}/8: d_h=${d_h} base_lr=${base_lr} seed=${SEED}"

python main.py \
    --cfg "${CFG}" \
    --repeat 1 \
    seed "${SEED}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    gnn.hybrid.d_h "${d_h}" \
    optim.base_lr "${base_lr}" \
    "${extra_args[@]}"

log_message "Finished ${wandb_name}"
