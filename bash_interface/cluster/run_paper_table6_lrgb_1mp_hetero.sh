#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 6 — LRGB datasets whose best SiGMA has exactly 1 MP head.
#
# Datasets: peptides_func, peptides_struct, coco
# 3 datasets × 5 variants × 5 seeds = 75 tasks (default).
#
# Variants (W&B group suffix + tag — keep names stable):
#   0  SiGMA               — best gated hybrid as-is (ng=1)
#   1  Homog_MP            — +1 MP head, same type (ng=2), gated
#   2  Hetero_MP           — +1 MP head, different type (ng=2), gated
#   3  Homog_MP_ungated    — +1 same-type MP head, gate=none
#   4  Hetero_MP_ungated   — +1 different-type MP head, gate=none
#
# Different-type second heads (chosen for complementarity / prior sweeps):
#   peptides_func:   GCN      → GCN,GINE
#   peptides_struct: GINE     → GINE,GGNN
#   coco:            GATEDGCN → GATEDGCN,GCN
#
# Anchors / source runs:
#   peptides_func   o5cdk766 a1g1 GCN       l31u4b3k
#   peptides_struct g3bsaq32 a1g1 GINE      bqkect9l
#   coco            5b4z9l3u a1g1 GATEDGCN  xgjakrz0
#
# W&B group:  paper_T6_<dataset>_<Variant>
# W&B tags:   paper_table6, <Variant>, <dataset>, seed<k>
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table6_lrgb_1mp_hetero.sh
# =============================================================================

#SBATCH --job-name=sigma_T6_1mp
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
num_seeds="${PAPER_T6_1MP_NUM_SEEDS:-5}"
num_variants="${PAPER_T6_1MP_NUM_VARIANTS:-5}"
num_datasets="${PAPER_T6_1MP_NUM_DATASETS:-3}"
num_tasks="${PAPER_T6_1MP_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
dataset_idx=$((rest / num_variants))

case "${dataset_idx}" in
    0)
        ds_tag="peptides_func"
        cfg="configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml"
        source_run="l31u4b3k"
        base_type="GCN"
        hetero_types="GCN,GINE"
        ;;
    1)
        ds_tag="peptides_struct"
        cfg="configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
        source_run="bqkect9l"
        base_type="GINE"
        hetero_types="GINE,GGNN"
        ;;
    2)
        ds_tag="coco"
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
        source_run="xgjakrz0"
        base_type="GATEDGCN"
        hetero_types="GATEDGCN,GCN"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

homog_types="${base_type},${base_type}"
extra_args=()

case "${variant_idx}" in
    0)
        variant="SiGMA"
        # Anchor as-is (ng=1, gated).
        ;;
    1)
        variant="Homog_MP"
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${homog_types}"
        )
        ;;
    2)
        variant="Hetero_MP"
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${hetero_types}"
        )
        ;;
    3)
        variant="Homog_MP_ungated"
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${homog_types}"
            gnn.hybrid.gate none
        )
        ;;
    4)
        variant="Hetero_MP_ungated"
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${hetero_types}"
            gnn.hybrid.gate none
        )
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T6_1MP_WANDB_PREFIX:-paper_T6}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table6,${variant},${ds_tag},seed${seed},source_${source_run}"

log_message "Table6 1MP task ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} seed=${seed} source=${source_run}"
log_message "W&B group=${wandb_group} name=${wandb_name} types_homog=${homog_types} types_hetero=${hetero_types}"

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
    gnn.hybrid.log_gate_stats True \
    gnn.hybrid.identity_proj False \
    "${extra_args[@]}"
