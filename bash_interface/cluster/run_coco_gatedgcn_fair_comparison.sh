#!/usr/bin/env bash
# =============================================================================
# COCO-SP fair comparison vs 5b4z9l3u GatedGCN+ baseline (seed 1).
#
# Baseline MP stack: configs/gatedgcn/coco.yaml → custom_gnn layer_type=gatedgcn
#   (layers_mp=20, dim_inner=52, batch=16, lr=0.001) — reproduces 5b4z9l3u.
#
# Hybrid fair repro: same outer hyperparams + 1×GATEDGCN MP head + {1,2} attn.
#
# 5 tasks (seed 1):
#   1  baseline gatedgcn MP-only
#   2  hybrid a1g1  lr=0.001
#   3  hybrid a2g1  lr=0.001
#   4  hybrid a1g1  lr=0.002
#   5  hybrid a2g1  lr=0.002
#
# Submit:
#   sbatch --job-name=coco_fair --array=1-5%2 --mem=128GB --time=192:00:00 \
#     --export=ALL,SEED=1,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_coco_gatedgcn_fair_comparison.sh
# =============================================================================

#SBATCH --job-name=coco_fair
#SBATCH --ntasks=1
#SBATCH --time=192:00:00
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
        cfg="configs/gatedgcn/coco.yaml"
        wandb_name="coco_gatedgcn_seed${SEED}_repro_baseline"
        hybrid_args=()
        ;;
    2)
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1.yaml"
        wandb_name="coco_gatedgcn_seed${SEED}_repro_hybrid_a1g1_lr0p001"
        hybrid_args=(optim.base_lr 0.001)
        ;;
    3)
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a2g1.yaml"
        wandb_name="coco_gatedgcn_seed${SEED}_repro_hybrid_a2g1_lr0p001"
        hybrid_args=(optim.base_lr 0.001)
        ;;
    4)
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1.yaml"
        wandb_name="coco_gatedgcn_seed${SEED}_repro_hybrid_a1g1_lr0p002"
        hybrid_args=(optim.base_lr 0.002)
        ;;
    5)
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a2g1.yaml"
        wandb_name="coco_gatedgcn_seed${SEED}_repro_hybrid_a2g1_lr0p002"
        hybrid_args=(optim.base_lr 0.002)
        ;;
    *)
        log_message "task_id=${task_id} out of range (expected 1-5)"
        exit 1
        ;;
esac

log_message "COCO fair compare task ${task_id}/5: ${wandb_name}"

python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${SEED}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    "${hybrid_args[@]}" \
    "${extra_args[@]}"

log_message "Finished ${wandb_name}"
