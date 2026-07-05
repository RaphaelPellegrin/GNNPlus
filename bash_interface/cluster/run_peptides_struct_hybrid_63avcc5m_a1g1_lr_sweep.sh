#!/usr/bin/env bash
# =============================================================================
# Peptides-struct LR ablation: 63avcc5m a1g1 anchor × 5 LR settings × 10 seeds.
#
# Source run (seed 3 baseline): https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/gzzcctaf
#
# Array task layout (50 tasks = 5 LR × 10 seeds):
#   task_id = (lr_idx * 10) + seed + 1   with lr_idx ∈ {0..4}, seed ∈ {0..9}
#
# LR settings (lr_idx):
#   0  base_lr=5.5835e-4  min_lr=0
#   1  base_lr=7.0e-4      min_lr=0
#   2  base_lr=9.0e-4      min_lr=0
#   3  base_lr=6.0e-4      min_lr=1e-6
#   4  base_lr=7.0e-4      min_lr=1e-6
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_63avcc5m_a1g1_lr_sweep.sh
# =============================================================================

#SBATCH --job-name=peptides_struct_63avcc5m_lr
#SBATCH --ntasks=1
#SBATCH --time=240:00:00
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
num_lr="${LR_SWEEP_NUM_LR:-5}"
num_seeds="${LR_SWEEP_NUM_SEEDS:-10}"
num_tasks="${LR_SWEEP_NUM_TASKS:-$((num_lr * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
lr_idx=$((idx / num_seeds))
seed=$((idx % num_seeds))

case "${lr_idx}" in
    0)
        base_lr="0.0005583531449037443"
        min_lr="0.0"
        lr_tag="b558_m0"
        ;;
    1)
        base_lr="0.0007"
        min_lr="0.0"
        lr_tag="b7_m0"
        ;;
    2)
        base_lr="0.0009"
        min_lr="0.0"
        lr_tag="b9_m0"
        ;;
    3)
        base_lr="0.0006"
        min_lr="0.000001"
        lr_tag="b6_m1e-6"
        ;;
    4)
        base_lr="0.0007"
        min_lr="0.000001"
        lr_tag="b7_m1e-6"
        ;;
    *)
        log_message "unknown lr_idx=${lr_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

cfg="configs/gated_hybrid/peptides-struct-hybrid-63avcc5m-a1g1-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${LR_SWEEP_WANDB_GROUP:-lr_ablation_peptides_struct_63avcc5m_a1g1}"
wandb_group="${wandb_group_prefix}_${lr_tag}"
wandb_name="peptides_struct_hybrid_63avcc5m_a1g1_seed${seed}_${lr_tag}_job${job_tag}_${task_id}"

log_message "peptides-struct 63avcc5m LR sweep task ${task_id}/${num_tasks}: lr_idx=${lr_idx} seed=${seed} base_lr=${base_lr} min_lr=${min_lr}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    gnn.hybrid.log_gate_stats True \
    optim.base_lr "${base_lr}" \
    optim.min_lr "${min_lr}" \
    "${extra_args[@]}"
