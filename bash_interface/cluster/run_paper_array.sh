#!/usr/bin/env bash
# =============================================================================
# Generic SLURM array: paper baselines for one dataset (gcn / gine / gatedgcn).
#
# Paper defaults: configs/{gcn,gine,gatedgcn}/<DATASET>.yaml with no
# hyperparameter overrides (matches README / arXiv:2502.09263).
#
# Submit from repo root (array size = 3 * NUM_SEEDS):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   sbatch --job-name=gnnplus_mnist --array=1-6%6 \
#     --export=ALL,DATASET=mnist,NUM_SEEDS=2 \
#     bash_interface/cluster/run_paper_array.sh
#
# Env:
#   DATASET     — yaml stem: cifar10, mnist, peptides-func, peptides-struct, coco, voc, ...
#   NUM_SEEDS   — number of seeds per model (default 2)
# =============================================================================

#SBATCH --job-name=gnnplus_paper
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
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

DATASET="${DATASET:?Set DATASET (e.g. mnist, peptides-func, coco, voc)}"
NUM_SEEDS="${NUM_SEEDS:-2}"

GNN_LIST=(gcn gine gatedgcn)
NUM_GNN=${#GNN_LIST[@]}
MAX_TASK=$((NUM_GNN * NUM_SEEDS))

task_id=${SLURM_ARRAY_TASK_ID:-1}
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$MAX_TASK" ]; then
    log_message "task_id=${task_id} out of range for NUM_SEEDS=${NUM_SEEDS} (max ${MAX_TASK})"
    exit 1
fi

cfg_path="configs/gcn/${DATASET}.yaml"
if [ ! -f "$cfg_path" ]; then
    log_message "Missing config: ${cfg_path}"
    exit 1
fi

idx=$((task_id - 1))
gnn_idx=$((idx / NUM_SEEDS))
seed_idx=$((idx % NUM_SEEDS))
gnn="${GNN_LIST[$gnn_idx]}"
seed="${seed_idx}"

log_message "Task ${task_id}/${MAX_TASK}: ${gnn} / ${DATASET} / seed=${seed}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

wandb_name="${DATASET}_${gnn}_seed${seed}_cluster"

python main.py \
    --cfg "configs/${gnn}/${DATASET}.yaml" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    "${extra_args[@]}"

log_message "Finished ${gnn} / ${DATASET} seed=${seed}"
