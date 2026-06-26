#!/usr/bin/env bash
# =============================================================================
# Fair repro: CLUSTER GatedGCN+ standard vs +1 attention (same seed as anchor).
#
# Task 1 → configs/gatedgcn/cluster.yaml (custom_gnn, seed 1)
# Task 2 → configs/gated_hybrid/cluster-gatedgcn-repro-a1.yaml (hybrid a1g1)
#
# Anchor: W&B n4unldzn / cluster_gatedgcn_seed1_cluster
#
# Submit:
#   sbatch --job-name=cluster_gatedgcn_fair --array=1-2%2 \
#     --mem=128GB --time=120:00:00 \
#     --export=ALL,SEED=1,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_cluster_gatedgcn_fair_repro.sh
# =============================================================================

#SBATCH --job-name=cluster_gatedgcn_fair
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

SEED="${SEED:-1}"
task_id=${SLURM_ARRAY_TASK_ID:-1}

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

case "${task_id}" in
    1)
        cfg="configs/gatedgcn/cluster.yaml"
        wandb_name="cluster_gatedgcn_seed${SEED}_cluster"
        log_message "Fair repro task 1/2: standard GatedGCN+ seed=${SEED}"
        python main.py \
            --cfg "${cfg}" \
            --repeat 1 \
            seed "${SEED}" \
            wandb.use True \
            wandb.entity "${WANDB_ENTITY}" \
            wandb.project "${WANDB_PROJECT}" \
            wandb.name "${wandb_name}" \
            "${extra_args[@]}"
        ;;
    2)
        cfg="configs/gated_hybrid/cluster-gatedgcn-repro-a1.yaml"
        wandb_name="cluster_gatedgcn_seed${SEED}_repro_hybrid_attn1"
        log_message "Fair repro task 2/2: hybrid a1g1 GATEDGCN+ seed=${SEED}"
        python main.py \
            --cfg "${cfg}" \
            --repeat 1 \
            seed "${SEED}" \
            wandb.use True \
            wandb.entity "${WANDB_ENTITY}" \
            wandb.project "${WANDB_PROJECT}" \
            wandb.name "${wandb_name}" \
            gnn.hybrid.log_gate_stats True \
            "${extra_args[@]}"
        ;;
    *)
        log_message "task_id=${task_id} out of range (expected 1-2)"
        exit 1
        ;;
esac

log_message "Finished ${wandb_name}"
