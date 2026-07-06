#!/usr/bin/env bash
# =============================================================================
# MalNet-Tiny hybrid architecture ablation (apiw6l3u / 9h3jqzkm lineage).
#
# Baseline anchor: d_h=110, GCNE, graph_restricted, L8, ep=250.
# Source: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u
#
# Array layout (default 8 tasks = 4 variants × 2 seeds):
#   task_id = (variant_idx * num_seeds) + seed + 1
#
# Variants (variant_idx), all base_lr=4e-3:
#   0  a5g5    — 5 attn + 5 GCNE MP
#   1  a0g10   — 0 attn + 10 GCNE MP
#   2  a0g50   — 0 attn + 50 GCNE MP
#   3  a50g50  — 50 attn + 50 GCNE MP  (may need 128GB GPU mem)
#
# Submit:
#   bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_arch_ablation.sh
# =============================================================================

#SBATCH --job-name=malnet_9h3jqzkm_arch
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
num_variants="${ARCH_ABLATION_NUM_VARIANTS:-4}"
num_seeds="${ARCH_ABLATION_NUM_SEEDS:-2}"
num_tasks="${ARCH_ABLATION_NUM_TASKS:-$((num_variants * num_seeds))}"
max_epoch="${ARCH_ABLATION_MAX_EPOCH:-250}"
min_lr="${ARCH_ABLATION_MIN_LR:-1e-6}"
base_lr="${ARCH_ABLATION_BASE_LR:-0.004}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
variant_idx=$((idx / num_seeds))
seed=$((idx % num_seeds))

case "${variant_idx}" in
    0)
        num_attn=5
        num_gnn=5
        variant_tag="a5g5_b40"
        ;;
    1)
        num_attn=0
        num_gnn=10
        variant_tag="a0g10_b40"
        ;;
    2)
        num_attn=0
        num_gnn=50
        variant_tag="a0g50_b40"
        ;;
    3)
        num_attn=50
        num_gnn=50
        variant_tag="a50g50_b40"
        ;;
    *)
        log_message "unknown variant_idx=${variant_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

cfg="configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${ARCH_ABLATION_WANDB_GROUP:-malnet_arch_ablation_9h3jqzkm}"
wandb_group="${wandb_group_prefix}_${variant_tag}"
wandb_name="malnet_hybrid_${variant_tag}_seed${seed}_job${job_tag}_${task_id}"

log_message "MalNet arch ablation task ${task_id}/${num_tasks}: variant=${variant_tag} attn=${num_attn} gnn=${num_gnn} seed=${seed} base_lr=${base_lr}"

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
    gnn.hybrid.num_attn_heads "${num_attn}" \
    gnn.hybrid.num_gnn_heads "${num_gnn}" \
    gnn.hybrid.log_gate_stats True \
    optim.base_lr "${base_lr}" \
    optim.min_lr "${min_lr}" \
    optim.max_epoch "${max_epoch}" \
    "${extra_args[@]}"
