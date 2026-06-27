#!/usr/bin/env bash
# =============================================================================
# CLUSTER hybrid refine grid on anchor ht9bntg2 (a1g1 GATEDGCN, seed 1).
#
# 32 tasks = hybrid_d_h {48,64,96,128} × lr {8e-4…2e-3} × attn_mask {full,graph_restricted}
# Center: d_h=64, lr=0.001492, full (task 7 reproduces ht9bntg2).
#
# Submit:
#   sbatch --job-name=cluster_refine --array=1-32%4 --mem=128GB --time=120:00:00 \
#     --export=ALL,SEED=1,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_cluster_hybrid_ht9bntg2_refine_sweep.sh
# =============================================================================

#SBATCH --job-name=cluster_refine
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
CFG="configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml"
task_id=${SLURM_ARRAY_TASK_ID:-1}

MASKS=(full graph_restricted)
DHS=(48 64 96 128)
LRS=(0.0008 0.001 0.001492 0.002)

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt 32 ]; then
    log_message "task_id=${task_id} out of range (expected 1-32)"
    exit 1
fi

idx=$((task_id - 1))
lr_idx=$((idx % 4))
dh_idx=$(((idx / 4) % 4))
mask_idx=$((idx / 16))
attn_mask="${MASKS[$mask_idx]}"
d_h="${DHS[$dh_idx]}"
base_lr="${LRS[$lr_idx]}"

if [ "${attn_mask}" = "graph_restricted" ]; then
    mask_tag="gr"
else
    mask_tag="full"
fi
lr_tag="${base_lr//./p}"
wandb_name="cluster_hybrid_a1g1_dh${d_h}_lr${lr_tag}_${mask_tag}_seed${SEED}"

log_message "CLUSTER refine task ${task_id}/32: d_h=${d_h} lr=${base_lr} attn_mask=${attn_mask} seed=${SEED}"

python main.py \
    --cfg "${CFG}" \
    --repeat 1 \
    seed "${SEED}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    gnn.hybrid.d_h "${d_h}" \
    gnn.hybrid.attn_mask "${attn_mask}" \
    optim.base_lr "${base_lr}" \
    "${extra_args[@]}"

log_message "Finished ${wandb_name}"
