#!/usr/bin/env bash
# =============================================================================
# CIFAR10 paper repro v2: t8prvgqr anchor (a4g4, d_h=128) × seeds 0–4.
#
# Parent: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/t8prvgqr
# Submit (gpu_h200): bash bash_interface/cluster/submit_cifar10_hybrid_t8prvgqr_paper_repro.sh
# =============================================================================

#SBATCH --job-name=cifar10_paper_v2
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
cfg="configs/gated_hybrid/cifar10-hybrid-t8prvgqr-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="cifar10_hybrid_t8prvgqr_a4g4_seed${seed}_job${job_tag}_${task_id}"

log_message "CIFAR10 v2 (t8prvgqr a4g4) paper repro task ${task_id}/${num_seeds}: seed=${seed}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    gnn.hybrid.log_gate_stats True \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.name "${wandb_name}" \
    "${extra_args[@]}"
