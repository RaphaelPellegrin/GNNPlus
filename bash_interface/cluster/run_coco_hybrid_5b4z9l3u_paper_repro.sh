#!/usr/bin/env bash
# =============================================================================
# COCO-SP paper repro v2: 5b4z9l3u / q57ng7d2 anchor (a1g1) × seeds 0–4.
#
# Parent: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q57ng7d2
# Submit:
#   bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u_paper_repro.sh
# =============================================================================

#SBATCH --job-name=coco_paper_v2
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
num_seeds="${PAPER_NUM_SEEDS:-5}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="coco_hybrid_5b4z9l3u_a1g1_seed${seed}_job${job_tag}_${task_id}"

log_message "COCO v2 (5b4z9l3u a1g1) paper repro task ${task_id}/${num_seeds}: seed=${seed}"

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
    wandb.name "${wandb_name}" \
    "${extra_args[@]}"
