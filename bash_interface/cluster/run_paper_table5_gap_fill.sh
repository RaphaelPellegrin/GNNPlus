#!/usr/bin/env bash
# =============================================================================
# Table 5 gap-fill retries (vanilla anchors — not gritvn4).
#
# Task map (1-based):
#   1–5   CIFAR10 MP_only          seeds 0–4   (prior array 35720034 crashed)
#   6–9   COCO SiGMA_ungated_attn  seeds 1–4   (seed 0 still running on 36605829)
#
# W&B groups (same as main campaign):
#   paper_T5_cifar10_MP_only
#   paper_T5_coco_SiGMA_ungated_attn
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_gap_fill.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_gap
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

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_tasks="${PAPER_T5_GAP_NUM_TASKS:-9}"
name_suffix="${PAPER_T5_GAP_NAME_SUFFIX:-_gapfill}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

extra_args=()

if [ "$task_id" -le 5 ]; then
    # CIFAR10 MP_only — a8g4 → 12× GATEDGCN
    ds_tag="cifar10"
    cfg="configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml"
    source_run="3tx560wq"
    variant="MP_only"
    seed=$((task_id - 1))
    total_heads=12
    mp_types="GATEDGCN"
    for ((i = 1; i < total_heads; i++)); do
        mp_types="${mp_types},GATEDGCN"
    done
    extra_args+=(
        gnn.hybrid.num_attn_heads 0
        gnn.hybrid.num_gnn_heads "${total_heads}"
        "gnn.hybrid.gnn_types" "${mp_types}"
    )
else
    # COCO ungated_attn — seeds 1–4 (tasks 6–9)
    ds_tag="coco"
    cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
    source_run="xgjakrz0"
    variant="SiGMA_ungated_attn"
    seed=$((task_id - 5))  # task6→seed1 … task9→seed4
    extra_args+=(
        gnn.hybrid.gate none
        gnn.hybrid.mp_gate headwise
    )
fi

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_GAP_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"
wandb_tags="paper_table5,paper_table6,${variant},${ds_tag},seed${seed},source_${source_run},gapfill"

log_message "T5 gap task ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
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
    "${extra_args[@]}"
