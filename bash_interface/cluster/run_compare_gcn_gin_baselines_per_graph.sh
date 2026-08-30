#!/usr/bin/env bash
# Pairwise per-graph GCN-only vs GIN-only comparison (SLURM worker).
#
# Submit: bash bash_interface/cluster/submit_compare_gcn_gin_baselines_per_graph.sh

#SBATCH --job-name=gcn_gin_pairwise
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
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

results_root="${GCN_GIN_PAIRWISE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
out_dir="${GCN_GIN_PAIRWISE_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
lr_tag="${GCN_GIN_PAIRWISE_LR_TAG:-lr001}"

log_message "pairwise baseline compare results_root=${results_root} lr=${lr_tag}"

python scripts/synthetic/compare_gcn_gin_baselines_per_graph.py \
  --results-root "${results_root}" \
  --dataset-dir "${GNNPLUS_DATASET_DIR}" \
  --out-dir "${out_dir}" \
  --lr-tag "${lr_tag}" \
  --device auto

log_message "Done."
