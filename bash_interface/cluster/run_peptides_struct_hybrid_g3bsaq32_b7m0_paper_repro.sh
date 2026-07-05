#!/usr/bin/env bash
# =============================================================================
# Peptides-struct paper repro: g3bsaq32 b7_m0 (lr=7e-4) × 10 seeds.
#
# Source: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/g3bsaq32
#
# Env:
#   PAPER_NUM_SEEDS=10 (default)
#   PAPER_MAX_EPOCH=250 or 350
#   PAPER_WANDB_GROUP=...
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_g3bsaq32_b7m0_ep250_paper_repro.sh
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_g3bsaq32_b7m0_ep350_paper_repro.sh
# =============================================================================

#SBATCH --job-name=peptides_struct_g3bsaq32_b7m0
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
num_seeds="${PAPER_NUM_SEEDS:-10}"
max_epoch="${PAPER_MAX_EPOCH:-250}"
wandb_group="${PAPER_WANDB_GROUP:-paper_bestmodel_v2_peptides_struct_g3bsaq32_b7m0_ep250}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="peptides_struct_hybrid_g3bsaq32_b7m0_seed${seed}_ep${max_epoch}_job${job_tag}_${task_id}"

log_message "peptides-struct g3bsaq32 b7_m0 paper repro task ${task_id}/${num_seeds}: seed=${seed} max_epoch=${max_epoch}"

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
    optim.max_epoch "${max_epoch}" \
    "${extra_args[@]}"
