#!/usr/bin/env bash
# =============================================================================
# Peptides-func UniGCN hybrid: 2i5psq22 a5g3 × 3 LR × 10 seeds = 30 tasks.
#
# Source run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2i5psq22
# Sweep: bq62chmz
#
# Array task layout (task_id = lr_idx * 10 + seed + 1):
#   Tasks  1–10: seeds 0–9 @ base_lr=0.000455  (≈ sweep exact 4.546e-4)
#   Tasks 11–20: seeds 0–9 @ base_lr=0.00045
#   Tasks 21–30: seeds 0–9 @ base_lr=0.0005
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_hybrid_2i5psq22_a5g3_lr_seeds.sh
# =============================================================================

#SBATCH --job-name=pf_2i5psq22_lr
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=64GB
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

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_lr="${PF_2I5_NUM_LR:-3}"
num_seeds="${PF_2I5_NUM_SEEDS:-10}"
num_tasks="${PF_2I5_NUM_TASKS:-$((num_lr * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
lr_idx=$((idx / num_seeds))
seed=$((idx % num_seeds))

case "${lr_idx}" in
    0)
        base_lr="0.000455"
        lr_tag="b455"
        ;;
    1)
        base_lr="0.00045"
        lr_tag="b45"
        ;;
    2)
        base_lr="0.0005"
        lr_tag="b5"
        ;;
    *)
        log_message "unknown lr_idx=${lr_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

cfg="configs/gated_hybrid/peptides-func-hybrid-2i5psq22-a5g3-unigcn-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PF_2I5_WANDB_GROUP:-peptides_func_2i5psq22_a5g3_lr_seeds}"
wandb_group="${wandb_group_prefix}_${lr_tag}"
wandb_name="peptides_func_hybrid_2i5psq22_a5g3_seed${seed}_${lr_tag}_job${job_tag}_${task_id}"
wandb_tags="unigcn,hybrid_gnn,peptides_func,anchor_2i5psq22,hybrid_a5g3,sweep_bq62chmz,${lr_tag},seed${seed}"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

log_message "peptides-func 2i5psq22 LR×seed task ${task_id}/${num_tasks}: lr_idx=${lr_idx} seed=${seed} base_lr=${base_lr}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

export WANDB_EXTRA_TAGS="${wandb_tags}"

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    model.type hybrid_gnn \
    gnn.hybrid.log_gate_stats True \
    gnn.hybrid.identity_proj False \
    gnn.hybrid.residual True \
    optim.base_lr "${base_lr}" \
    "${extra_args[@]}"
