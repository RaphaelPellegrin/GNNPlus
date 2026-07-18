#!/usr/bin/env bash
# =============================================================================
# ENZYMES ogpkubk9 a4g4 — 2 scheduler variants × 5 seeds = 10 tasks.
#
# Source: https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/ogpkubk9
#   HybridGated a4g4 GCN,GIN,SAGE,GAT L12/H64/dh16 headwise LN + skip/FFN
#
# Variants:
#   0 plateau — reduce_on_plateau (matches source lr_scheduler=plateau)
#   1 cosine  — cosine_with_warmup
#
# Submit:
#   bash bash_interface/cluster/submit_enzymes_ogpkubk9_seed_grids.sh
# =============================================================================

#SBATCH --job-name=enz_ogpkubk9
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
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
num_seeds="${ENZ_OGPK_NUM_SEEDS:-5}"
num_variants="${ENZ_OGPK_NUM_VARIANTS:-2}"
num_tasks="${ENZ_OGPK_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
variant_idx=$((idx / num_seeds))

case "${variant_idx}" in
    0)
        variant="plateau"
        cfg="configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml"
        wandb_group="enzymes_ogpkubk9_a4g4_plateau_seeds"
        ;;
    1)
        variant="cosine"
        cfg="configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-cosine-anchor.yaml"
        wandb_group="enzymes_ogpkubk9_a4g4_cosine_seeds"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="enzymes_ogpkubk9_a4g4_${variant}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="enzymes,ogpkubk9,a4g4,${variant},seed${seed},source_ogpkubk9"

log_message "ENZYMES ogpkubk9 task ${task_id}/${num_tasks}: variant=${variant} seed=${seed} cfg=${cfg}"

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
    "${extra_args[@]}"
