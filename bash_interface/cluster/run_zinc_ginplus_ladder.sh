#!/usr/bin/env bash
# =============================================================================
# ZINC GIN+ staged ladder: levels 0–4 × seeds {0,1,2,3,4} (25 tasks).
#
# Paper baseline: configs/gine/zinc.yaml (ICML 2025 GNN+ default, arXiv:2502.09263)
#   GIN+: 12×GINE @ dh=80, RWSE 20 steps, FFN+residual, lr=1e-3, 2000 epochs.
#
# Level 0 — Paper single head GIN+ baseline (custom_gnn, 12×GINE @ 80, no gate)
# Level 1 — Single head + gating: hybrid a0g1 @ dh=80 (1×GINE MP + headwise gate)
# Level 2 — + 1 attention head: hybrid a1g1 @ dh=80 (1×attn + 1×GINE MP, gated)
# Level 3 — + 2nd MP head: hybrid a0g2 @ dh=80 (GINE + GatedGCN, gated, no attn)
# Level 4 — Full hybrid: hybrid a1g2 @ dh=80 (1×attn + GINE + GatedGCN, gated)
#
# Submit:
#   bash bash_interface/cluster/submit_zinc_ginplus_ladder.sh
# =============================================================================

#SBATCH --job-name=zinc_gin_ladder
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_tasks="${ZINC_GINPLUS_LADDER_NUM_TASKS:-25}"
num_seeds="${ZINC_GINPLUS_LADDER_NUM_SEEDS:-5}"
wandb_group="${ZINC_GINPLUS_LADDER_WANDB_GROUP:-zinc_ginplus_ladder}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

level_idx=$(( (task_id - 1) / num_seeds ))
seed=$(( (task_id - 1) % num_seeds ))

case "${level_idx}" in
    0)
        level="0"
        cfg="configs/gine/zinc.yaml"
        variant_tag="level0_ginplus_paper"
        wandb_tags="ginplus_ladder,zinc,level_0,ginplus_paper"
        extra_args=()
        ;;
    1)
        level="1"
        cfg="configs/gated_hybrid/zinc-ginplus-ladder-l1-a0g1.yaml"
        variant_tag="level1_ginplus_gated_a0g1"
        wandb_tags="ginplus_ladder,zinc,level_1,hybrid_a0g1"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    2)
        level="2"
        cfg="configs/gated_hybrid/zinc-ginplus-ladder-l2-a1g1.yaml"
        variant_tag="level2_ginplus_attn_a1g1"
        wandb_tags="ginplus_ladder,zinc,level_2,hybrid_a1g1"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    3)
        level="3"
        cfg="configs/gated_hybrid/zinc-ginplus-ladder-l3-a0g2.yaml"
        variant_tag="level3_ginplus_2mp_a0g2"
        wandb_tags="ginplus_ladder,zinc,level_3,hybrid_a0g2"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    4)
        level="4"
        cfg="configs/gated_hybrid/zinc-ginplus-ladder-l4-a1g2.yaml"
        variant_tag="level4_ginplus_full_a1g2"
        wandb_tags="ginplus_ladder,zinc,level_4,hybrid_a1g2"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    *)
        log_message "unknown level_idx=${level_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="zinc_ginplus_l${level}_s${seed}_${variant_tag}_job${job_tag}_${task_id}"

log_message "ZINC GIN+ ladder task ${task_id}/${num_tasks}: level=${level} seed=${seed} cfg=${cfg}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

# YACS cannot override yaml list fields via CLI (wandb.tags); use env instead.
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
