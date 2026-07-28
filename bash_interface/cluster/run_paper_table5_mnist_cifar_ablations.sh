#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 6 (code: paper_T5) — MNIST + CIFAR10 + PATTERN
#
# 3 datasets × 5 variants × 5 seeds = 75 tasks (default).
#
# Variants (W&B group suffix + tag — same names as LRGB Table 6 / paper_T5):
#   0  SiGMA             — best gated hybrid (paper Table 3 baseline)
#   1  SiGMA_ungated     — same architecture, gnn.hybrid.gate=none
#   2  SiGMA_attn_gate   — yaml gate on attention; mp_gate=none
#   3  Attn_only         — all MP heads replaced by attention
#   4  MP_only           — all attention heads replaced by same MP type(s)
#
# W&B group:  paper_T5_<dataset>_<Variant>
# W&B tags:   paper_table5, paper_table6, <Variant>, <dataset>, seed<k>
#
# Source anchors / best exemplar runs (hyperparams frozen in yaml):
#   mnist    lcvbyyss a2g2 GATEDGCN×2   seed0  uh7nxm4e
#   cifar10  ulij45a2 a8g4 GATEDGCN×4   seed1  3tx560wq
#   pattern  ta9qtxb9 a2g2 GCNE×2       seed0  ta9qtxb9
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_mnist_cifar_ablations.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_mc
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=96GB
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
num_seeds="${PAPER_T5_MC_NUM_SEEDS:-5}"
num_variants="${PAPER_T5_MC_NUM_VARIANTS:-5}"
num_datasets="${PAPER_T5_MC_NUM_DATASETS:-3}"
num_tasks="${PAPER_T5_MC_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

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
        ds_tag="mnist"
        cfg="configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml"
        source_run="uh7nxm4e"
        na=2; ng=2; gnn_types="GATEDGCN,GATEDGCN"
        ;;
    1)
        ds_tag="cifar10"
        cfg="configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml"
        source_run="3tx560wq"
        na=8; ng=4; gnn_types="GATEDGCN"
        ;;
    2)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-gcne-best-hybrid.yaml"
        source_run="ta9qtxb9"
        na=2; ng=2; gnn_types="GCNE,GCNE"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

total_heads=$((na + ng))
first_type="${gnn_types%%,*}"
extra_args=()

case "${variant_idx}" in
    0)
        variant="SiGMA"
        # Anchor yaml unchanged (gated hybrid).
        ;;
    1)
        variant="SiGMA_ungated"
        extra_args+=(gnn.hybrid.gate none)
        ;;
    2)
        variant="SiGMA_attn_gate"
        # Keep yaml ``gate`` on attention; disable MP gating only.
        extra_args+=(gnn.hybrid.mp_gate none)
        ;;
    3)
        variant="Attn_only"
        extra_args+=(
            gnn.hybrid.num_attn_heads "${total_heads}"
            gnn.hybrid.num_gnn_heads 0
            "gnn.hybrid.gnn_types" ""
        )
        ;;
    4)
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
wandb_group_prefix="${PAPER_T5_MC_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,paper_table6,${variant},${ds_tag},seed${seed},source_${source_run}"

log_message "Table6 MC task ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} seed=${seed} source=${source_run} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
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
