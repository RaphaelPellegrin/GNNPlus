#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 6 (PascalVOC-SP only) — homogeneous vs heterogeneous MP.
#
# 3 variants × 5 seeds = 15 tasks (default).
#
# Variants (W&B group suffix + tag — keep these names stable):
#   0  SiGMA              — best gated hybrid (GATEDGCN,GATEDGCN)
#   1  Hetero_MP          — same as SiGMA but gnn_types=GATEDGCN,GCN (still gated)
#   2  Hetero_MP_ungated  — GATEDGCN,GCN + gnn.hybrid.gate=none
#
# Anchor: configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml
# Source run: vyt7hjj5 (paper_bestmodel_v1_voc_j7ukyzdm)
#
# W&B group:  paper_T6_voc_<Variant>
# W&B tags:   paper_table6, voc, <Variant>, seed<k>
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table6_voc_hetero.sh
# =============================================================================

#SBATCH --job-name=sigma_T6_voc
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
num_seeds="${PAPER_T6_VOC_NUM_SEEDS:-5}"
num_variants="${PAPER_T6_VOC_NUM_VARIANTS:-3}"
num_tasks="${PAPER_T6_VOC_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
variant_idx=$((idx / num_seeds))

ds_tag="voc"
cfg="configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml"
source_run="vyt7hjj5"

extra_args=()
case "${variant_idx}" in
    0)
        variant="SiGMA"
        # Anchor as-is: GATEDGCN,GATEDGCN + headwise gate.
        ;;
    1)
        variant="Hetero_MP"
        extra_args+=("gnn.hybrid.gnn_types" "GATEDGCN,GCN")
        ;;
    2)
        variant="Hetero_MP_ungated"
        extra_args+=(
            "gnn.hybrid.gnn_types" "GATEDGCN,GCN"
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
wandb_group_prefix="${PAPER_T6_VOC_WANDB_PREFIX:-paper_T6}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table6,${variant},${ds_tag},seed${seed},source_${source_run}"

log_message "Table6 VOC task ${task_id}/${num_tasks}: variant=${variant} seed=${seed} source=${source_run} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

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
