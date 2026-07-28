#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 7 (code: paper_T6) — MNIST + CIFAR10 + PATTERN
#
# Baselines already have multiple homogeneous MP heads. Table 7 only needs:
#   - Homog_MP_ungated  (same types, gate=none)
#   - Hetero_MP         (swap ONE MP head to a different type, gated)
#   - Hetero_MP_ungated (same one-head swap, gate=none)
#
# SiGMA / Homog_MP gated cells → reuse paper_bestmodel anchors (do not relaunch).
#
# 3 datasets × 3 variants × 5 seeds = 45 tasks (default).
#
# Anchors / source runs (Paper_final_runs.md):
#   mnist    lcvbyyss a2g2 GATEDGCN×2   uh7nxm4e
#   cifar10  ulij45a2 a8g4 GATEDGCN×4   3tx560wq
#   pattern  ta9qtxb9 a2g2 GCNE×2       ta9qtxb9
#
# Hetero (swap last MP head only):
#   mnist    GATEDGCN,GATEDGCN  →  GATEDGCN,GCN
#   cifar10  GATEDGCN×4         →  GATEDGCN,GATEDGCN,GATEDGCN,GCN
#   pattern  GCNE,GCNE          →  GCNE,GINE
#
# W&B group:  paper_T6_<dataset>_<Variant>
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table6_mnist_cifar_pattern.sh
# =============================================================================

#SBATCH --job-name=sigma_T6_mc
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
num_seeds="${PAPER_T6_MC_NUM_SEEDS:-5}"
num_variants="${PAPER_T6_MC_NUM_VARIANTS:-3}"
num_datasets="${PAPER_T6_MC_NUM_DATASETS:-3}"
num_tasks="${PAPER_T6_MC_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

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
        homog_types="GATEDGCN,GATEDGCN"
        # Swap one (last) MP head only.
        hetero_types="GATEDGCN,GCN"
        ng=2
        ;;
    1)
        ds_tag="cifar10"
        cfg="configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml"
        source_run="3tx560wq"
        homog_types="GATEDGCN,GATEDGCN,GATEDGCN,GATEDGCN"
        # Swap one (last) MP head only — not alternating.
        hetero_types="GATEDGCN,GATEDGCN,GATEDGCN,GCN"
        ng=4
        ;;
    2)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-gcne-best-hybrid.yaml"
        source_run="ta9qtxb9"
        homog_types="GCNE,GCNE"
        # Swap one (last) MP head only.
        hetero_types="GCNE,GINE"
        ng=2
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

extra_args=()

case "${variant_idx}" in
    0)
        variant="Homog_MP_ungated"
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
            "gnn.hybrid.gnn_types" "${homog_types}"
            gnn.hybrid.gate none
        )
        ;;
    1)
        variant="Hetero_MP"
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
            "gnn.hybrid.gnn_types" "${hetero_types}"
        )
        ;;
    2)
        variant="Hetero_MP_ungated"
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
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
wandb_group_prefix="${PAPER_T6_MC_WANDB_PREFIX:-paper_T6}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
name_suffix="${PAPER_T6_MC_NAME_SUFFIX:-}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"
wandb_tags="paper_table6,paper_table7,${variant},${ds_tag},seed${seed},source_${source_run},one_mp_swap"
if [ -n "${name_suffix}" ]; then
    tag_suffix="${name_suffix#_}"
    wandb_tags="${wandb_tags},relaunch_${tag_suffix}"
fi

log_message "Table7 MC task ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} seed=${seed} source=${source_run}"
log_message "W&B group=${wandb_group} name=${wandb_name} homog=${homog_types} hetero=${hetero_types}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi
if [ -n "${PAPER_T6_MC_MAX_EPOCH:-}" ]; then
    extra_args+=(optim.max_epoch "${PAPER_T6_MC_MAX_EPOCH}")
    log_message "max_epoch override: ${PAPER_T6_MC_MAX_EPOCH}"
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
