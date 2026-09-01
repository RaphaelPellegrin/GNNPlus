#!/usr/bin/env bash
# Opposite-sign τ pair analysis (login node if pairwise CSV exists; else GPU).
#
# Fast path (after fig05 pairwise job):
#   export GCN_GIN_OPPOSITE_FROM_CSV=results/gcn_gin_routing/analysis/pairwise_baseline_per_graph.csv
#
# Full path (+ optional gated eval):
#   unset GCN_GIN_OPPOSITE_FROM_CSV
#   export GCN_GIN_OPPOSITE_INCLUDE_GATED=1
#
# Submit: bash bash_interface/cluster/submit_analyze_opposite_sign_pairs.sh

#SBATCH --job-name=gcn_gin_opp_pairs
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=logs_gnnplus/%x_%j.log
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

results_root="${GCN_GIN_OPPOSITE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
out_dir="${GCN_GIN_OPPOSITE_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
lr_tag="${GCN_GIN_OPPOSITE_LR_TAG:-lr001}"
from_csv="${GCN_GIN_OPPOSITE_FROM_CSV:-${out_dir}/pairwise_baseline_per_graph.csv}"
include_gated="${GCN_GIN_OPPOSITE_INCLUDE_GATED:-0}"

cmd=(
  python scripts/synthetic/analyze_opposite_sign_pairs.py
  --dataset-dir "${GNNPLUS_DATASET_DIR}"
  --out-dir "${out_dir}"
  --lr-tag "${lr_tag}"
  --device auto
)

if [[ -f "${from_csv}" ]]; then
  log_message "opposite-sign pairs from CSV ${from_csv}"
  cmd+=(--from-per-graph-csv "${from_csv}")
fi

if [[ "${include_gated}" == "1" ]]; then
  log_message "SiGMA eval (gated + ungated) results_root=${results_root}"
  cmd+=(--results-root "${results_root}" --include-gated)
elif [[ ! -f "${from_csv}" ]]; then
  log_message "opposite-sign pairs from checkpoints results_root=${results_root}"
  cmd+=(--results-root "${results_root}")
fi

log_message "Running: ${cmd[*]}"
"${cmd[@]}"

log_message "Done."
