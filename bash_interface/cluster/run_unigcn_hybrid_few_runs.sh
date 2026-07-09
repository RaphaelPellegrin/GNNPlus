#!/usr/bin/env bash
# =============================================================================
# UniGCN hybrid few-run sweep: 4 datasets × 3 variants × N seeds.
#
# Datasets:  peptides-func, peptides-struct, CLUSTER, PATTERN
# Variants:  a0g1 (gated UniGCN only)
#            a1g1 (1×attn + gated UniGCN)
#            a1g2 (1×attn + gated UniGCN + GINE)
#
# Configs: configs/gated_hybrid/unigcn/<dataset>-<variant>.yaml
#
# Task layout (default 3 seeds per dataset×variant):
#   seed cycles fastest → tasks 1–3 share cfg (s0,s1,s2), then next variant, etc.
#   peptides-func a0g1 s0..s2, a1g1 s0..s2, a1g2 s0..s2,
#   peptides-struct ..., cluster ..., pattern ...
#
# W&B: model.type=hybrid_gnn and gnn.hybrid.gnn_types come from each yaml;
#   tags via cfg.wandb.tags + WANDB_EXTRA_TAGS (never CLI wandb.tags).
#   group = <UNIGCN_WANDB_GROUP>_<dataset>_<variant> (3 seeds per group).
#
# Submit:
#   bash bash_interface/cluster/submit_unigcn_hybrid_few_runs.sh
# =============================================================================

#SBATCH --job-name=unigcn_hybrid
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=64GB
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
num_seeds="${UNIGCN_NUM_SEEDS:-3}"
num_variants="${UNIGCN_NUM_VARIANTS:-3}"
num_datasets="${UNIGCN_NUM_DATASETS:-4}"
num_tasks=$((num_datasets * num_variants * num_seeds))
wandb_group_base="${UNIGCN_WANDB_GROUP:-unigcn_hybrid_few_runs}"

datasets=(peptides-func peptides-struct cluster pattern)
variants=(a0g1 a1g1 a1g2)

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
dataset_idx=$((rest / num_variants))

dataset="${datasets[$dataset_idx]}"
variant="${variants[$variant_idx]}"
cfg="configs/gated_hybrid/unigcn/${dataset}-${variant}.yaml"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

dataset_tag="${dataset//-/_}"
case "${variant}" in
    a0g1|a1g1) gnn_types="UNIGCN" ;;
    a1g2) gnn_types="UNIGCN,GINE" ;;
    *) gnn_types="UNIGCN" ;;
esac
gnn_types_tag="${gnn_types//,/_}"
wandb_group="${wandb_group_base}_${dataset_tag}_${variant}"
wandb_tags="unigcn,hybrid_gnn,model_hybrid_gnn,gnn_types_${gnn_types_tag},${dataset_tag},hybrid_${variant},seed${seed}"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="unigcn_${dataset_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}"

extra_args=(
    model.type hybrid_gnn
    gnn.hybrid.identity_proj False
    gnn.hybrid.residual True
    "gnn.hybrid.gnn_types" "${gnn_types}"
    gnn.hybrid.log_gate_stats True
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "UniGCN hybrid task ${task_id}/${num_tasks}: dataset=${dataset} variant=${variant} seed=${seed} cfg=${cfg}"
log_message "W&B: group=${wandb_group} model.type=hybrid_gnn gnn.hybrid.gnn_types=${gnn_types}"

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
    "${extra_args[@]}"
