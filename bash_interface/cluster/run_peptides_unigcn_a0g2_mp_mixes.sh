#!/usr/bin/env bash
# =============================================================================
# Peptides func+struct: a0g2 UniGCN mixes (no attention), best-SiGMA HPs.
#
# 2 datasets × 2 MP mixes × 5 seeds = 20 tasks.
#
# Variants (W&B group suffix):
#   0  UNIGCN_GINE       — a0g2 types UNIGCN,GINE (gated)
#   1  UNIGCN_GATEDGCN   — a0g2 types UNIGCN,GATEDGCN (gated)
#
# HP anchors (attn removed; only heads change):
#   peptides_func   Homog_MP / o5cdk766 lineage (AP 0.7080 a1g2 best)
#   peptides_struct g3bsaq32 a1g1 GINE (paper SiGMA struct)
#
# W&B group:  paper_peptides_<ds>_a0g2_<Variant>
# Submit:
#   bash bash_interface/cluster/submit_peptides_unigcn_a0g2_mp_mixes.sh
# =============================================================================

#SBATCH --job-name=pep_unigcn_a0g2
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
num_seeds="${PEP_UNIGCN_NUM_SEEDS:-5}"
num_variants="${PEP_UNIGCN_NUM_VARIANTS:-2}"
num_datasets="${PEP_UNIGCN_NUM_DATASETS:-2}"
num_tasks="${PEP_UNIGCN_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

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
        cfg="configs/gated_hybrid/peptides-func-hybrid-homog-a1g2-gcn-anchor.yaml"
        source_note="homog_a1g2_o5cdk766"
        ;;
    1)
        ds_tag="peptides_struct"
        cfg="configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
        source_note="g3bsaq32_a1g1"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

case "${variant_idx}" in
    0)
        variant="UNIGCN_GINE"
        gnn_types="UNIGCN,GINE"
        ;;
    1)
        variant="UNIGCN_GATEDGCN"
        gnn_types="UNIGCN,GATEDGCN"
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
wandb_group_prefix="${PEP_UNIGCN_WANDB_PREFIX:-paper_peptides}"
wandb_group="${wandb_group_prefix}_${ds_tag}_a0g2_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="peptides_unigcn_a0g2,${variant},${ds_tag},seed${seed},a0g2,no_attn,source_${source_note}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "Peptides UniGCN a0g2 task ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} types=${gnn_types} seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

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
    gnn.hybrid.num_attn_heads 0 \
    gnn.hybrid.num_gnn_heads 2 \
    "gnn.hybrid.gnn_types" "${gnn_types}" \
    gnn.hybrid.log_gate_stats True \
    gnn.hybrid.identity_proj False \
    "${extra_args[@]}"
