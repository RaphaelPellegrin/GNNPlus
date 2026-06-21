#!/usr/bin/env bash
# =============================================================================
# SLURM array: hybrid_gnn (gated attention + MP heads) for one dataset.
#
# Configs: configs/gated_hybrid/<DATASET>.yaml
#   - Outer dims/optim match GNN+ paper gcne configs per dataset
#   - hybrid: num_attn_heads=2, num_gnn_heads=2, gnn_types GCN,GIN (or GCN,GINE)
#   - W&B project MOE_6, tag gnnplus (set in yaml)
#
# Submit from repo root:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export ENV_NAME=gnnplus
#   sbatch --job-name=gnnplus_hybrid_mnist --array=1-2%2 \
#     --export=ALL,DATASET=mnist,NUM_SEEDS=2,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_hybrid_array.sh
#
# Env:
#   DATASET    — yaml stem under configs/gated_hybrid/ (mnist, cifar10, …)
#   NUM_SEEDS  — seeds for this array task (see submit_hybrid_suite.sh)
# =============================================================================

#SBATCH --job-name=gnnplus_hybrid
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/hybrid_%x_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

DATASET="${DATASET:?Set DATASET (e.g. mnist, cifar10, peptides-func, coco, voc)}"
NUM_SEEDS="${NUM_SEEDS:-2}"

export WANDB_PROJECT="${WANDB_PROJECT:-MOE_6}"

cfg_path="configs/gated_hybrid/${DATASET}.yaml"
if [ ! -f "$cfg_path" ]; then
    log_message "Missing config: ${cfg_path}"
    exit 1
fi

task_id=${SLURM_ARRAY_TASK_ID:-1}
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$NUM_SEEDS" ]; then
    log_message "task_id=${task_id} out of range for NUM_SEEDS=${NUM_SEEDS}"
    exit 1
fi

seed=$((task_id - 1))

log_message "Task ${task_id}/${NUM_SEEDS}: hybrid_gnn / ${DATASET} / seed=${seed}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

wandb_name="${DATASET}_hybrid_a2g2_seed${seed}_gnnplus"

python main.py \
    --cfg "${cfg_path}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${wandb_name}" \
    "${extra_args[@]}"

log_message "Finished hybrid_gnn / ${DATASET} seed=${seed}"
