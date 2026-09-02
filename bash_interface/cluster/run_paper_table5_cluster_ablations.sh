#!/usr/bin/env bash
# =============================================================================
# Table 5/6 architecture ablations — CLUSTER on paper SiGMA ht9bntg2.
#
# Anchor (Paper_final_runs):
#   W&B   paper_bestmodel_v1_cluster_ht9bntg2
#   Acc.  78.956±0.112%  (seeds 0–4)
#   Cfg   configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml
#   Arch  a1g1 GATEDGCN, headwise, LN, full mask, L16/H56/d_h64,
#         lr=1.492e-3, warmup5 / ep100, bs16  (vanilla attn — not grit)
#
# Variants → paper_T5_cluster_<Variant>:
#   0 SiGMA (skip by default) | 1 ungated | 2 attn_gate | 3 ungated_attn
#   4 Attn_only (2 attn) | 5 MP_only (2×GATEDGCN)
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_cluster_ablations.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_cluster
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
num_seeds="${PAPER_T5_CLUSTER_NUM_SEEDS:-5}"
seed_offset="${PAPER_T5_CLUSTER_SEED_OFFSET:-0}"
include_sigma="${PAPER_T5_CLUSTER_INCLUDE_SIGMA:-0}"

if [ -n "${PAPER_T5_CLUSTER_VARIANT_INDICES:-}" ]; then
    IFS=',' read -r -a variant_list <<< "${PAPER_T5_CLUSTER_VARIANT_INDICES}"
elif [ "${include_sigma}" = "1" ]; then
    variant_list=(0 1 2 3 4 5)
else
    variant_list=(1 2 3 4 5)
fi
num_variants=${#variant_list[@]}
num_tasks="${PAPER_T5_CLUSTER_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
variant_idx="${variant_list[$((idx / num_seeds))]}"

ds_tag="cluster"
cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml"
source_run="ht9bntg2"
na=1
ng=1
gnn_types="GATEDGCN"
total_heads=$((na + ng))
first_type="${gnn_types%%,*}"
extra_args=()

case "${variant_idx}" in
    0) variant="SiGMA" ;;
    1)
        variant="SiGMA_ungated"
        extra_args+=(gnn.hybrid.gate none)
        ;;
    2)
        variant="SiGMA_attn_gate"
        extra_args+=(gnn.hybrid.mp_gate none)
        ;;
    3)
        variant="SiGMA_ungated_attn"
        extra_args+=(
            gnn.hybrid.gate none
            gnn.hybrid.mp_gate headwise
        )
        ;;
    4)
        variant="Attn_only"
        extra_args+=(
            gnn.hybrid.num_attn_heads "${total_heads}"
            gnn.hybrid.num_gnn_heads 0
            "gnn.hybrid.gnn_types" ""
        )
        ;;
    5)
        variant="MP_only"
        mp_types="${first_type}"
        for ((i = 1; i < total_heads; i++)); do
            mp_types="${mp_types},${first_type}"
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

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_CLUSTER_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,paper_table6,${variant},${ds_tag},seed${seed},source_${source_run}"

log_message "T5 CLUSTER task ${task_id}/${num_tasks}: variant=${variant} seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

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
    "${extra_args[@]}"
