#!/usr/bin/env bash
# =============================================================================
# MalNet-Tiny hybrid LR ablation: 9h3jqzkm a0g2 × 5 LR settings × 10 seeds.
#
# Source run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u
#
# Array task layout (50 tasks = 5 LR × 10 seeds):
#   task_id = (lr_idx * 10) + seed + 1   with lr_idx ∈ {0..4}, seed ∈ {0..9}
#
# LR settings (lr_idx), all min_lr=1e-6, max_epoch=250:
#   0  base_lr=1.914236e-3  (exact 9h3jqzkm / apiw6l3u)
#   1  base_lr=1.7e-3
#   2  base_lr=2.1e-3
#   3  base_lr=2.3e-3
#   4  base_lr=2.5e-3
#
# Submit:
#   bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_lr_sweep.sh
# =============================================================================

#SBATCH --job-name=malnet_9h3jqzkm_lr
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
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
max_epoch="${LR_SWEEP_MAX_EPOCH:-250}"
min_lr="${LR_SWEEP_MIN_LR:-1e-6}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
lr_idx=$((idx / num_seeds))
seed=$((idx % num_seeds))

case "${lr_idx}" in
    0)
        base_lr="0.0019142361730964056"
        lr_tag="b1914_m1e6"
        ;;
    1)
        base_lr="0.0017"
        lr_tag="b17_m1e6"
        ;;
    2)
        base_lr="0.0021"
        lr_tag="b21_m1e6"
        ;;
    3)
        base_lr="0.0023"
        lr_tag="b23_m1e6"
        ;;
    4)
        base_lr="0.0025"
        lr_tag="b25_m1e6"
        ;;
    *)
        log_message "unknown lr_idx=${lr_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

cfg="configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${LR_SWEEP_WANDB_GROUP:-lr_ablation_malnet_hybrid_9h3jqzkm_a0g2}"
wandb_group="${wandb_group_prefix}_${lr_tag}"
wandb_name="malnet_hybrid_9h3jqzkm_a0g2_seed${seed}_${lr_tag}_job${job_tag}_${task_id}"

log_message "MalNet 9h3jqzkm LR sweep task ${task_id}/${num_tasks}: lr_idx=${lr_idx} seed=${seed} base_lr=${base_lr} min_lr=${min_lr} max_epoch=${max_epoch}"

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
    optim.max_epoch "${max_epoch}" \
    "${extra_args[@]}"
