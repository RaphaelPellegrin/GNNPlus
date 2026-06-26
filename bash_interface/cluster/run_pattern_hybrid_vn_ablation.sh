#!/usr/bin/env bash
# =============================================================================
# PATTERN virtual-node ablation on anchor ta9qtxb9 (a2g2, d_h=90, seed 0).
#
# Task 1 → baseline (no virtual nodes)
# Task 2 → +1 virtual node
# Task 3 → +2 virtual nodes
# Task 4 → +4 virtual nodes
#
# Submit:
#   sbatch --job-name=pattern_hybrid_vn --array=1-4%4 --mem=128GB --time=120:00:00 \
#     --export=ALL,SEED=0,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_pattern_hybrid_vn_ablation.sh
# =============================================================================

#SBATCH --job-name=pattern_hybrid_vn
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

SEED="${SEED:-0}"
CFG="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-anchor.yaml"
task_id=${SLURM_ARRAY_TASK_ID:-1}

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

case "${task_id}" in
    1)
        num_vn=0
        add_vn="False"
        wandb_name="pattern_hybrid_a2g2_dh90_anchor_seed${SEED}"
        ;;
    2)
        num_vn=1
        add_vn="True"
        wandb_name="pattern_hybrid_a2g2_dh90_vn1_seed${SEED}"
        ;;
    3)
        num_vn=2
        add_vn="True"
        wandb_name="pattern_hybrid_a2g2_dh90_vn2_seed${SEED}"
        ;;
    4)
        num_vn=4
        add_vn="True"
        wandb_name="pattern_hybrid_a2g2_dh90_vn4_seed${SEED}"
        ;;
    *)
        log_message "task_id=${task_id} out of range (expected 1-4)"
        exit 1
        ;;
esac

log_message "PATTERN VN ablation task ${task_id}/4: add_vn=${add_vn} num_vn=${num_vn} seed=${SEED}"

python main.py \
    --cfg "${CFG}" \
    --repeat 1 \
    seed "${SEED}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    dataset.add_virtual_nodes "${add_vn}" \
    dataset.num_virtual_nodes "${num_vn}" \
    "${extra_args[@]}"

log_message "Finished ${wandb_name}"
