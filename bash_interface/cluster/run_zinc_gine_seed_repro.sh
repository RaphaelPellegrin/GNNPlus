#!/usr/bin/env bash
# =============================================================================
# ZINC GINE baseline seed repro × 5 (seeds 0–4).
#
# Source runs (same cfg configs/gine/zinc.yaml; different seeds/commits):
#   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/gbdpt2gc
#   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/x6a7vgim
#   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/jb8domtt
#
# Submit:
#   bash bash_interface/cluster/submit_zinc_gine_seed_repro.sh
# =============================================================================

#SBATCH --job-name=zinc_gine_seed_repro
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
num_seeds="${ZINC_GINE_NUM_SEEDS:-5}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gine/zinc.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="${ZINC_GINE_WANDB_GROUP:-seed_repro_zinc_gine}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"

log_message "ZINC GINE seed repro task ${task_id}/${num_seeds}: seed=${seed} group=${wandb_group}"

# YACS cannot override yaml list fields via CLI (wandb.tags); use env instead.
export WANDB_EXTRA_TAGS="seed_repro,zinc,gine"

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
    "${extra_args[@]}"
