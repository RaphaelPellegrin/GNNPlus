#!/usr/bin/env bash
# =============================================================================
# MalNet-Tiny GatedGCN+ staged ladder: levels 0–4 × seeds {0,1,2,3,4} (25 tasks).
#
# Paper baseline: configs/gatedgcn/mal.yaml (ICML 2025 GNN+ default, arXiv:2502.09263)
#   GatedGCN+: 6×GatedGCN @ dh=100, DummyEdge, FFN+residual, lr=5e-4, 150 epochs.
#
# Level 0 — Paper single head GatedGCN+ baseline (custom_gnn, 6×GatedGCN @ 100, no gate)
# Level 1 — Single head + gating: hybrid a0g1 @ dh=100 (1×GatedGCN MP + headwise gate)
# Level 2 — + 1 attention head: hybrid a1g1 @ dh=100 (1×attn + 1×GatedGCN MP, gated, graph_restricted)
# Level 3 — + 2nd MP head: hybrid a0g2 @ dh=100 (GatedGCN + GINE, gated, no attn)
# Level 4 — Full hybrid: hybrid a1g2 @ dh=100 (1×attn + GatedGCN + GINE, gated, graph_restricted)
#
# Submit:
#   bash bash_interface/cluster/submit_mal_gatedgcnplus_ladder.sh
# =============================================================================

#SBATCH --job-name=mal_gated_ladder
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
num_tasks="${MAL_GATEDGCNPLUS_LADDER_NUM_TASKS:-25}"
num_seeds="${MAL_GATEDGCNPLUS_LADDER_NUM_SEEDS:-5}"
wandb_group="${MAL_GATEDGCNPLUS_LADDER_WANDB_GROUP:-mal_gatedgcnplus_ladder}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

level_idx=$(( (task_id - 1) / num_seeds ))
seed=$(( (task_id - 1) % num_seeds ))

case "${level_idx}" in
    0)
        level="0"
        cfg="configs/gatedgcn/mal.yaml"
        variant_tag="level0_gatedgcnplus_paper"
        wandb_tags="gatedgcnplus_ladder,malnet,level_0,gatedgcnplus_paper"
        extra_args=()
        ;;
    1)
        level="1"
        cfg="configs/gated_hybrid/mal-gatedgcnplus-ladder-l1-a0g1.yaml"
        variant_tag="level1_gatedgcnplus_gated_a0g1"
        wandb_tags="gatedgcnplus_ladder,malnet,level_1,hybrid_a0g1"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    2)
        level="2"
        cfg="configs/gated_hybrid/mal-gatedgcnplus-ladder-l2-a1g1.yaml"
        variant_tag="level2_gatedgcnplus_attn_a1g1"
        wandb_tags="gatedgcnplus_ladder,malnet,level_2,hybrid_a1g1"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    3)
        level="3"
        cfg="configs/gated_hybrid/mal-gatedgcnplus-ladder-l3-a0g2.yaml"
        variant_tag="level3_gatedgcnplus_2mp_a0g2"
        wandb_tags="gatedgcnplus_ladder,malnet,level_3,hybrid_a0g2"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    4)
        level="4"
        cfg="configs/gated_hybrid/mal-gatedgcnplus-ladder-l4-a1g2.yaml"
        variant_tag="level4_gatedgcnplus_full_a1g2"
        wandb_tags="gatedgcnplus_ladder,malnet,level_4,hybrid_a1g2"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    *)
        log_message "unknown level_idx=${level_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="mal_gatedgcnplus_l${level}_s${seed}_${variant_tag}_job${job_tag}_${task_id}"

log_message "MalNet-Tiny GatedGCN+ ladder task ${task_id}/${num_tasks}: level=${level} seed=${seed} cfg=${cfg}"

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
