#!/usr/bin/env bash
# =============================================================================
# Paper Table 6 (code: T5) COCO a1g2 twins — ungated + attn_gate · 10 jobs.
#
# Baseline SiGMA COCO is a1g1. Main Table 6 ungated / attn_gate keep a1g1.
# This campaign adds +1 GATEDGCN MP head (a1g2), matching the Table 7 Homog
# head count, as an "extra twin" (same idea as Attn_only_a3 / MP_only_a0g3):
#   tasks 1–5  SiGMA_ungated_a1g2     gate=none, a1g2 GATEDGCN×2
#   tasks 6–10 SiGMA_attn_gate_a1g2   attn gated, mp_gate=none, a1g2
#
# W&B groups (distinct from a1g1 and from paper_T6 Homog_*):
#   paper_T5_coco_SiGMA_ungated_a1g2
#   paper_T5_coco_SiGMA_attn_gate_a1g2
#
# Default: full 300 epochs (anchor length).
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_coco_ungated_a1g2.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_coco_a1g2
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

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${PAPER_T5_COCO_A1G2_NUM_SEEDS:-5}"
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
mp_types="GATEDGCN,GATEDGCN"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

extra_args=(
    gnn.hybrid.num_attn_heads 1
    gnn.hybrid.num_gnn_heads 2
    "gnn.hybrid.gnn_types" "${mp_types}"
)

case "${variant_idx}" in
    0)
        variant="SiGMA_ungated"
        group_suffix="SiGMA_ungated_a1g2"
        extra_args+=(gnn.hybrid.gate none)
        head_tag="a1g2_ungated"
        ;;
    1)
        variant="SiGMA_attn_gate"
        group_suffix="SiGMA_attn_gate_a1g2"
        # Keep yaml gate (headwise); disable MP gates only.
        extra_args+=(gnn.hybrid.mp_gate none)
        head_tag="a1g2_attn_gate"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_COCO_A1G2_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${group_suffix}"
name_suffix="${PAPER_T5_COCO_A1G2_NAME_SUFFIX:-}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"
wandb_tags="paper_table5,paper_table6,${variant},${ds_tag},seed${seed},source_${source_run},${head_tag},coco_a1g2_twin"

log_message "COCO Table6 ${variant} a1g2 task ${task_id}/${num_tasks}: seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name} cfg=${cfg}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi
max_epoch="${PAPER_T5_COCO_A1G2_MAX_EPOCH:-300}"
extra_args+=(optim.max_epoch "${max_epoch}")
log_message "max_epoch: ${max_epoch}"

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
