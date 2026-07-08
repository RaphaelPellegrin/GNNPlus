#!/usr/bin/env bash
# =============================================================================
# Peptides-func Level 2ter: hybrid_gnn a0g1 collapsed to Level-1-like (identity proj).
#
# Config: configs/gated_hybrid/peptides-func-gcn-repro-a0g1-identity-proj.yaml
#   still hybrid_gnn / SiGMA-shaped API
#   identity_proj + no LN + no residual + d_h=275 + 3 layers
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_level2ter_seeds.sh
# =============================================================================

#SBATCH --job-name=peptides_l2ter
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
num_tasks="${LEVEL2TER_NUM_TASKS:-10}"
wandb_group="${LEVEL2TER_WANDB_GROUP:-peptides_func_level2ter_seeds}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/peptides-func-gcn-repro-a0g1-identity-proj.yaml"
wandb_tags="level_2ter,hybrid_gnn,hybrid_a0g1,no_ln,no_residual,identity_proj,seed_sweep"
variant_tag="seed${seed}"

extra_args=(gnn.hybrid.log_gate_stats True)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="peptides_func_l2ter_${variant_tag}_job${job_tag}_${task_id}"

log_message "Level-2ter seed sweep task ${task_id}/${num_tasks}: seed=${seed} cfg=${cfg}"

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
