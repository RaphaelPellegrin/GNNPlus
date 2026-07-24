#!/usr/bin/env bash
# =============================================================================
# Paper Table 6 (code: T5) COCO Attn_only a3 + MP_only a0g3 — 10 jobs.
#
# Baseline SiGMA COCO is a1g1 GATEDGCN. Previous Attn/MP ablations used
# total_heads=2 (a2g0 / a0g2). This campaign uses 3 heads:
#   tasks 1–5  Attn_only  a3g0                 (3 attention heads)
#   tasks 6–10 MP_only    a0g3 GATEDGCN×3      (3 MP heads)
#
# Distinct W&B groups (do not mix with old a2 runs):
#   paper_T5_coco_Attn_only_a3
#   paper_T5_coco_MP_only_a0g3
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_coco_attn_mp_a3.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_coco_a3
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
num_seeds="${PAPER_T5_COCO_A3_NUM_SEEDS:-5}"
num_variants=2
num_tasks=$((num_variants * num_seeds))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
variant_idx=$((idx / num_seeds))

ds_tag="coco"
cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
source_run="xgjakrz0"
n_heads="${PAPER_T5_COCO_A3_HEADS:-3}"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

extra_args=()
case "${variant_idx}" in
    0)
        variant="Attn_only"
        group_suffix="Attn_only_a${n_heads}"
        extra_args+=(
            gnn.hybrid.num_attn_heads "${n_heads}"
            gnn.hybrid.num_gnn_heads 0
            "gnn.hybrid.gnn_types" ""
        )
        head_tag="a${n_heads}g0"
        ;;
    1)
        variant="MP_only"
        group_suffix="MP_only_a0g${n_heads}"
        mp_types="GATEDGCN"
        for ((i = 1; i < n_heads; i++)); do
            mp_types="${mp_types},GATEDGCN"
        done
        extra_args+=(
            gnn.hybrid.num_attn_heads 0
            gnn.hybrid.num_gnn_heads "${n_heads}"
            "gnn.hybrid.gnn_types" "${mp_types}"
        )
        head_tag="a0g${n_heads}"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_COCO_A3_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${group_suffix}"
name_suffix="${PAPER_T5_COCO_A3_NAME_SUFFIX:-}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"
wandb_tags="paper_table5,paper_table6,${variant},${ds_tag},seed${seed},source_${source_run},${head_tag},coco_a3_relaunch"

log_message "COCO Table6 ${variant} ${head_tag} task ${task_id}/${num_tasks}: seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name} cfg=${cfg}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi
if [ -n "${PAPER_T5_COCO_A3_MAX_EPOCH:-}" ]; then
    extra_args+=(optim.max_epoch "${PAPER_T5_COCO_A3_MAX_EPOCH}")
    log_message "max_epoch override: ${PAPER_T5_COCO_A3_MAX_EPOCH}"
fi
# Default: 150 epochs for this a3/a0g3 COCO campaign (insurance / faster fill).
if [ -z "${PAPER_T5_COCO_A3_MAX_EPOCH:-}" ]; then
    extra_args+=(optim.max_epoch 150)
    log_message "max_epoch default: 150"
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
