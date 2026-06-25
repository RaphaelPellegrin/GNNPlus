#!/usr/bin/env bash
# =============================================================================
# SLURM array: fair repro baseline custom_gnn vs hybrid (+1 attn), all paper seeds.
#
# Tasks 1..NUM_SEEDS     → baseline (gcn/*.yaml)
# Tasks NUM_SEEDS+1..2N  → hybrid_attn1 (gated_hybrid/*-repro-a1.yaml)
#
# Submit:
#   sbatch --job-name=fair_repro_pattern --array=1-8%4 --mem=128GB --time=120:00:00 \
#     --export=ALL,DATASET=pattern,NUM_SEEDS=4,ENV_NAME=gnnplus \
#     bash_interface/cluster/run_fair_repro_array.sh
#
# Env:
#   DATASET   — pattern | cluster | mal
#   NUM_SEEDS — paper repeat count (pattern 4, cluster 2, mal 5)
# =============================================================================

#SBATCH --job-name=fair_repro
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

DATASET="${DATASET:?Set DATASET (pattern, cluster, mal)}"
NUM_SEEDS="${NUM_SEEDS:?Set NUM_SEEDS}"

task_id=${SLURM_ARRAY_TASK_ID:-1}
max_task=$((2 * NUM_SEEDS))
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$max_task" ]; then
    log_message "task_id=${task_id} out of range (max ${max_task})"
    exit 1
fi

if [ "$task_id" -le "$NUM_SEEDS" ]; then
    variant="baseline"
    seed=$((task_id - 1))
else
    variant="hybrid_attn1"
    seed=$((task_id - NUM_SEEDS - 1))
fi

log_message "Fair repro task ${task_id}/${max_task}: ${DATASET} ${variant} seed=${seed}"

export GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}"

exec bash bash_interface/sweeps/sweep_wrapper_gnnplus_repro.sh \
    --dataset="${DATASET}" \
    --repro_variant="${variant}" \
    --seed="${seed}"
