#!/usr/bin/env bash
# =============================================================================
# Peptides-func Level 1 ONLY (custom_gnn_gated): 10 seeds × 3 LRs = 30 tasks.
#
# Config: configs/gcn/peptides-func-gated.yaml
#
# Task layout (seed cycles 0–9 within each LR block):
#   Tasks  1–10: seeds 0–9 @ base_lr=1e-3
#   Tasks 11–20: seeds 0–9 @ base_lr × 1.25  (= 1.25e-3)
#   Tasks 21–30: seeds 0–9 @ base_lr × 0.75  (= 7.5e-4)
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_level1_seed_lr.sh
# =============================================================================

#SBATCH --job-name=peptides_l1_sweep
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
num_seeds="${LEVEL1_SWEEP_NUM_SEEDS:-10}"
num_lr="${LEVEL1_SWEEP_NUM_LR:-3}"
num_tasks="${LEVEL1_SWEEP_NUM_TASKS:-$((num_seeds * num_lr))}"
base_lr="${LEVEL1_SWEEP_BASE_LR:-0.001}"
lr_high_mult="${LEVEL1_SWEEP_LR_HIGH_MULT:-1.25}"
lr_low_mult="${LEVEL1_SWEEP_LR_LOW_MULT:-0.75}"
wandb_group="${LEVEL1_SWEEP_WANDB_GROUP:-peptides_func_level1_seed_lr_30}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

seed=$(( (task_id - 1) % num_seeds ))
lr_idx=$(( (task_id - 1) / num_seeds ))

case "${lr_idx}" in
    0)
        lr="${base_lr}"
        lr_tag="lr_base"
        ;;
    1)
        lr="$(python -c "print(${base_lr} * ${lr_high_mult})")"
        lr_tag="lr_${lr_high_mult}x"
        ;;
    2)
        lr="$(python -c "print(${base_lr} * ${lr_low_mult})")"
        lr_tag="lr_${lr_low_mult}x"
        ;;
    *)
        log_message "unknown lr_idx=${lr_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

cfg="configs/gcn/peptides-func-gated.yaml"
variant_tag="seed${seed}_${lr_tag}"
wandb_tags="level_1,custom_gnn_gated,seed_lr_grid,${lr_tag}"

extra_args=()
if [ "${lr_idx}" -ne 0 ]; then
    extra_args+=(optim.base_lr "${lr}")
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="peptides_func_l1_${variant_tag}_job${job_tag}_${task_id}"

log_message "Level-1 grid task ${task_id}/${num_tasks}: seed=${seed} lr=${lr} (${lr_tag}) cfg=${cfg}"

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
    "${extra_args[@]}"
