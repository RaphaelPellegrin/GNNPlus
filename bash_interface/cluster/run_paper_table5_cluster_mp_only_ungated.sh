#!/usr/bin/env bash
# =============================================================================
# Table 5 CLUSTER — MP_only ungated (match gated MP_only head count).
#
# Gated MP_only (paper_T5_cluster_MP_only): 79.087±0.158%
#   a0g2  GATEDGCN,GATEDGCN  · gate=headwise (from yaml)
#
# This run: same 2 MP heads, gnn.hybrid.gate=none.
#
#   bash bash_interface/cluster/submit_paper_table5_cluster_mp_only_ungated.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_cluster_mpung
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
num_seeds="${PAPER_T5_CLUSTER_MPUNG_NUM_SEEDS:-5}"
seed_offset="${PAPER_T5_CLUSTER_MPUNG_SEED_OFFSET:-0}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((seed_offset + task_id - 1))
cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml"
# Same head budget as gated MP_only: SiGMA a1g1 → a0g2 GATEDGCN×2
num_gnn_heads=2
gnn_types="GATEDGCN,GATEDGCN"
variant="MP_only_ungated"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_CLUSTER_MPUNG_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_cluster_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,paper_table6,${variant},cluster,seed${seed},source_ht9bntg2,a0g2"

log_message "T5 CLUSTER ${variant} task ${task_id}/${num_seeds}: seed=${seed} a0g${num_gnn_heads} ${gnn_types} gate=none"
log_message "W&B group=${wandb_group} name=${wandb_name}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
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
    gnn.hybrid.num_attn_heads 0 \
    gnn.hybrid.num_gnn_heads "${num_gnn_heads}" \
    "gnn.hybrid.gnn_types" "${gnn_types}" \
    gnn.hybrid.gate none \
    "${extra_args[@]}"
