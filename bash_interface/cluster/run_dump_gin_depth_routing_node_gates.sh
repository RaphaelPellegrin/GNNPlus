#!/usr/bin/env bash
# Dump per-graph root gates for GIN depth-routing gated runs.
#
# Submit: bash bash_interface/cluster/submit_dump_gin_depth_routing_node_gates.sh

#SBATCH --job-name=gin_depth_gdump
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --output=logs_gnnplus/%x_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

results_root="${GIN_DEPTH_GDUMP_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
dataset_dir="${GIN_DEPTH_GDUMP_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"

python scripts/synthetic/dump_gin_depth_routing_node_gates.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --tracks toy \
  --device auto \
  --skip-existing

log_message "Gate dump complete under ${results_root}/toy/l2_a0g1_gated_*"
