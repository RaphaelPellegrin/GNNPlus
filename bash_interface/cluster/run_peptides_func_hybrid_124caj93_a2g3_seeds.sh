#!/usr/bin/env bash
# =============================================================================
# Peptides-func UniGCN hybrid: 124caj93 a2g3 × 10 seeds.
#
# Source run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/124caj93
# Sweep: bq62chmz
#
# Array: task_id 1–10 → seeds 0–9
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_hybrid_124caj93_a2g3_seeds.sh
# =============================================================================

#SBATCH --job-name=pf_124caj93_seeds
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
num_seeds="${PF_124_NUM_SEEDS:-10}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/peptides-func-hybrid-124caj93-a2g3-unigcn-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="${PF_124_WANDB_GROUP:-peptides_func_124caj93_a2g3_seeds}"
wandb_name="peptides_func_hybrid_124caj93_a2g3_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="unigcn,hybrid_gnn,peptides_func,anchor_124caj93,hybrid_a2g3,sweep_bq62chmz,seed${seed}"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

log_message "peptides-func 124caj93 a2g3 seed task ${task_id}/${num_seeds}: seed=${seed}"

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
    "${extra_args[@]}"
