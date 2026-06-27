#!/usr/bin/env bash
# =============================================================================
# PATTERN fair comparison vs qcz7umtl GCNE baseline (seed 2).
#
# Baseline MP stack: configs/gcn/pattern.yaml → custom_gnn layer_type=gcne
#   (layers_mp=12, dim_inner=90, ffn=True, lr=0.001) — reproduces qcz7umtl.
#
# Hybrid fair repro: same outer hyperparams + 1×GCNE MP head + {1,2} attention heads.
#
# 5 tasks (seed 2):
#   1  baseline gcne MP-only
#   2  hybrid a1g1  lr=0.001
#   3  hybrid a2g1  lr=0.001
#   4  hybrid a1g1  lr=0.002
#   5  hybrid a2g1  lr=0.002
#
# Submit:
#   sbatch --job-name=pattern_fair --array=1-5%2 --mem=128GB --time=120:00:00 \
#     --export=ALL,SEED=2,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_pattern_gcne_fair_comparison.sh
# =============================================================================

#SBATCH --job-name=pattern_fair
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

SEED="${SEED:-2}"
task_id=${SLURM_ARRAY_TASK_ID:-1}

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

case "${task_id}" in
    1)
        cfg="configs/gcn/pattern.yaml"
        wandb_name="pattern_gcne_seed${SEED}_repro_baseline"
        hybrid_args=()
        ;;
    2)
        cfg="configs/gated_hybrid/pattern-gcne-repro-a1.yaml"
        wandb_name="pattern_gcne_seed${SEED}_repro_hybrid_a1g1_lr0p001"
        hybrid_args=(optim.base_lr 0.001 gnn.hybrid.log_gate_stats True)
        ;;
    3)
        cfg="configs/gated_hybrid/pattern-gcne-repro-a2.yaml"
        wandb_name="pattern_gcne_seed${SEED}_repro_hybrid_a2g1_lr0p001"
        hybrid_args=(optim.base_lr 0.001 gnn.hybrid.log_gate_stats True)
        ;;
    4)
        cfg="configs/gated_hybrid/pattern-gcne-repro-a1.yaml"
        wandb_name="pattern_gcne_seed${SEED}_repro_hybrid_a1g1_lr0p002"
        hybrid_args=(optim.base_lr 0.002 gnn.hybrid.log_gate_stats True)
        ;;
    5)
        cfg="configs/gated_hybrid/pattern-gcne-repro-a2.yaml"
        wandb_name="pattern_gcne_seed${SEED}_repro_hybrid_a2g1_lr0p002"
        hybrid_args=(optim.base_lr 0.002 gnn.hybrid.log_gate_stats True)
        ;;
    *)
        log_message "task_id=${task_id} out of range (expected 1-5)"
        exit 1
        ;;
esac

log_message "PATTERN fair compare task ${task_id}/5: ${wandb_name}"

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
