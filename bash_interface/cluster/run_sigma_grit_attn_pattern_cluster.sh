#!/usr/bin/env bash
# =============================================================================
# SiGMA + GRIT attention heads — PATTERN + CLUSTER seed grids
#
# 2 datasets × 5 seeds = 10 tasks (default).
#
# W&B group:  paper_sigma_grit_attn_pattern | paper_sigma_grit_attn_cluster
# W&B tags:   sigma_grit_attn, attn_type_grit, grit_attn, <dataset>, seed<k>
#
# Anchors:
#   pattern  configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml
#   cluster  configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml
#
# Submit:
#   bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh
# =============================================================================

#SBATCH --job-name=sigma_grit_attn
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
num_seeds="${SIGMA_GRIT_ATTN_NUM_SEEDS:-5}"
num_datasets="${SIGMA_GRIT_ATTN_NUM_DATASETS:-2}"
num_tasks="${SIGMA_GRIT_ATTN_NUM_TASKS:-$((num_datasets * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
dataset_idx=$((idx / num_seeds))

case "${dataset_idx}" in
    0)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
        wandb_group="paper_sigma_grit_attn_pattern"
        source_run="ta9qtxb9"
        ;;
    1)
        ds_tag="cluster"
        cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml"
        wandb_group="paper_sigma_grit_attn_cluster"
        source_run="ht9bntg2"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="sigma_grit_attn_${ds_tag}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="sigma_grit_attn,attn_type_grit,grit_attn,${ds_tag},seed${seed},source_${source_run}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "sigma_grit_attn task ${task_id}/${num_tasks}: ds=${ds_tag} seed=${seed} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"
log_message "Force override: gnn.hybrid.attn_type=grit"

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
    gnn.hybrid.attn_type grit \
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
