#!/usr/bin/env bash
# =============================================================================
# Relaunch Table 5 COCO Attn_only only (5 seeds) — same recipe as tasks 71–75
# of job 32232124, but as a fresh array on gpu_h200.
#
# Does NOT cancel the old mweber_gpu jobs. New W&B runs still land in
# paper_T5_coco_Attn_only (aggregate with --state finished; prefer newest).
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_coco_attn_only_h200.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_coco_attn
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${PAPER_T5_COCO_ATTN_NUM_SEEDS:-5}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

# Array task 1..5 → seed 0..4 (same seeds as old 32232124_71..75).
seed=$((task_id - 1))

ds_tag="coco"
variant="Attn_only"
cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
source_run="xgjakrz0"
# COCO SiGMA anchor is a1g1 → Attn_only replaces the 1 MP head → 2 attn heads.
na=1
ng=1
total_heads=$((na + ng))

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_COCO_ATTN_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}_h200"
wandb_tags="paper_table5,${variant},${ds_tag},seed${seed},source_${source_run},relaunch_h200"

log_message "Table5 COCO Attn_only H200 task ${task_id}/${num_seeds}: seed=${seed} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

extra_args=(
    gnn.hybrid.num_attn_heads "${total_heads}"
    gnn.hybrid.num_gnn_heads 0
    "gnn.hybrid.gnn_types" ""
)
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
    "${extra_args[@]}"
