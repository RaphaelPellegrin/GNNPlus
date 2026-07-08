#!/usr/bin/env bash
# =============================================================================
# ZINC fairness ladder: levels 0–4 × seeds {0,1,2} (15 tasks).
#
# Paper baseline: configs/gcn/zinc.yaml (custom_gnn, 12×GCNE @ 64, RWSE, 2000 ep)
#
# Level 0 — baseline GCNE @ 64 (no gate)
# Level 1 — GCNE @ 64 + headwise γ gating (custom_gnn)
# Level 2 — hybrid a0g1 @ d_h=64 (gated MP only)
# Level 3 — hybrid a0g2 @ d_h=64 (2× gated MP, no attention)
# Level 4 — hybrid a1g1 @ d_h=64 (1× attn + 1× GCNE MP)
#
# Submit:
#   bash bash_interface/cluster/submit_zinc_fairness_ladder.sh
# =============================================================================

#SBATCH --job-name=zinc_ladder
#SBATCH --ntasks=1
#SBATCH --time=240:00:00
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
num_tasks="${ZINC_FAIRNESS_LADDER_NUM_TASKS:-15}"
num_seeds="${ZINC_FAIRNESS_LADDER_NUM_SEEDS:-3}"
wandb_group="${ZINC_FAIRNESS_LADDER_WANDB_GROUP:-zinc_fairness_ladder}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

level_idx=$(( (task_id - 1) / num_seeds ))
seed=$(( (task_id - 1) % num_seeds ))

case "${level_idx}" in
    0)
        level="0"
        cfg="configs/gcn/zinc.yaml"
        variant_tag="level0_gcne_baseline"
        wandb_tags="fairness_ladder,zinc,level_0,gcne_baseline"
        extra_args=()
        ;;
    1)
        level="1"
        cfg="configs/gcn/zinc-gated.yaml"
        variant_tag="level1_gcne_gated"
        wandb_tags="fairness_ladder,zinc,level_1,gcne_gated"
        extra_args=()
        ;;
    2)
        level="2"
        cfg="configs/gated_hybrid/zinc-gcn-repro-a0g1.yaml"
        variant_tag="level2_hybrid_a0g1_dh64"
        wandb_tags="fairness_ladder,zinc,level_2,hybrid_a0g1"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    3)
        level="3"
        cfg="configs/gated_hybrid/zinc-gcn-repro-a0g2.yaml"
        variant_tag="level3_hybrid_a0g2_dh64"
        wandb_tags="fairness_ladder,zinc,level_3,hybrid_a0g2"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    4)
        level="4"
        cfg="configs/gated_hybrid/zinc-gcn-repro-a1.yaml"
        variant_tag="level4_hybrid_a1g1_dh64"
        wandb_tags="fairness_ladder,zinc,level_4,hybrid_a1g1"
        extra_args=(gnn.hybrid.log_gate_stats True)
        ;;
    *)
        log_message "unknown level_idx=${level_idx} for task_id=${task_id}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="zinc_l${level}_seed${seed}_repro_${variant_tag}_job${job_tag}_${task_id}"

log_message "ZINC fairness ladder task ${task_id}/${num_tasks}: level=${level} seed=${seed} cfg=${cfg}"

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
    wandb.tags "${wandb_tags}" \
    "${extra_args[@]}"
