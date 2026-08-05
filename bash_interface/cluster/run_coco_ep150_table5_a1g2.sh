#!/usr/bin/env bash
# =============================================================================
# COCO-SP Table 5 — full a1g2 family @ ep150 (extra twin).
#
# Based on frozen SiGMA anchor coco-hybrid-5b4z9l3u-a1g1-anchor.yaml
# (paper SiGMA ~0.42 F1, a1g1). Extra twin expands to a1g2 (1 attn + 2 MP),
# total_heads=3 → Attn_only a3 / MP_only a0g3.
#
# Variants (tasks = variant_idx * 5 + seed + 1):
#   0  SiGMA                 a1g2 gated
#   1  SiGMA_ungated         gate=none
#   2  SiGMA_attn_gate       mp_gate=none          ("Hybrid, ungated MP")
#   3  SiGMA_ungated_attn    gate=none, mp_gate=hw ("Hybrid, ungated Att")
#   4  Attn_only             a3g0
#   5  MP_only               a0g3 GATEDGCN×3
#
# W&B: paper_T5_ep150_coco_{SiGMA,SiGMA_ungated,...}_a1g2
#      (Attn_only / MP_only → _Attn_only_a3 / _MP_only_a0g3)
#
# Submit:
#   bash bash_interface/cluster/submit_coco_ep150_table5_a1g2.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_coco_a1g2_ep150
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
num_seeds="${PAPER_T5_COCO_A1G2_EP150_NUM_SEEDS:-5}"
num_variants="${PAPER_T5_COCO_A1G2_EP150_NUM_VARIANTS:-6}"
num_tasks=$((num_variants * num_seeds))
max_epoch="${PAPER_T5_COCO_A1G2_EP150_MAX_EPOCH:-150}"

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
total_heads=3  # a1g2 → Attn_only a3 / MP_only a0g3

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

extra_args=()
case "${variant_idx}" in
    0)
        variant="SiGMA"
        group_suffix="SiGMA_a1g2"
        extra_args+=(
            gnn.hybrid.num_attn_heads 1
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "GATEDGCN,GATEDGCN"
        )
        ;;
    1)
        variant="SiGMA_ungated"
        group_suffix="SiGMA_ungated_a1g2"
        extra_args+=(
            gnn.hybrid.num_attn_heads 1
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "GATEDGCN,GATEDGCN"
            gnn.hybrid.gate none
        )
        ;;
    2)
        variant="SiGMA_attn_gate"
        group_suffix="SiGMA_attn_gate_a1g2"
        extra_args+=(
            gnn.hybrid.num_attn_heads 1
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "GATEDGCN,GATEDGCN"
            gnn.hybrid.mp_gate none
        )
        ;;
    3)
        variant="SiGMA_ungated_attn"
        group_suffix="SiGMA_ungated_attn_a1g2"
        extra_args+=(
            gnn.hybrid.num_attn_heads 1
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "GATEDGCN,GATEDGCN"
            gnn.hybrid.gate none
            gnn.hybrid.mp_gate headwise
        )
        ;;
    4)
        variant="Attn_only"
        group_suffix="Attn_only_a3"
        extra_args+=(
            gnn.hybrid.num_attn_heads "${total_heads}"
            gnn.hybrid.num_gnn_heads 0
            "gnn.hybrid.gnn_types" ""
        )
        ;;
    5)
        variant="MP_only"
        group_suffix="MP_only_a0g3"
        mp_types="GATEDGCN"
        for ((i = 1; i < total_heads; i++)); do
            mp_types="${mp_types},GATEDGCN"
        done
        extra_args+=(
            gnn.hybrid.num_attn_heads 0
            gnn.hybrid.num_gnn_heads "${total_heads}"
            "gnn.hybrid.gnn_types" "${mp_types}"
        )
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_prefix="${PAPER_T5_COCO_A1G2_EP150_WANDB_PREFIX:-paper_T5_ep150}"
name_suffix="${PAPER_T5_COCO_A1G2_EP150_NAME_SUFFIX:-_ep150_a1g2}"
wandb_group="${wandb_prefix}_${ds_tag}_${group_suffix}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"
wandb_tags="paper_table5,${variant},${ds_tag},seed${seed},source_${source_run},a1g2_family,ep${max_epoch}"

log_message "COCO T5 a1g2 ep${max_epoch} task ${task_id}/${num_tasks}: ${variant} seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
fi
extra_args+=(optim.max_epoch "${max_epoch}")

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
