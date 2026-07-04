#!/usr/bin/env bash
# =============================================================================
# Peptides-func paper repro: 3g180qle GCN lineage, a1g8 (1 attn + 8× GCN) × seeds 0–4.
#
# MP baseline: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3g180qle
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_hybrid_3g180qle_a1g8_paper_repro.sh
# =============================================================================

#SBATCH --job-name=peptides_func_3g180qle_a1g8
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
num_seeds="${PAPER_NUM_SEEDS:-5}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/peptides-func-hybrid-3g180qle-a1g8-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="peptides_func_hybrid_3g180qle_a1g8_seed${seed}_job${job_tag}_${task_id}"

log_message "peptides-func (3g180qle a1g8 GCN×8) paper repro task ${task_id}/${num_seeds}: seed=${seed}"

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
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
