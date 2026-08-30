#!/usr/bin/env bash
# Render paper table figures from existing analysis CSVs (login node, no GPU).
#
# Prereq: pairwise_baseline_summary.csv and/or mask_ablation_summary.csv under
#   results/gcn_gin_routing/analysis/
#
# Usage:
#   bash bash_interface/cluster/run_plot_gcn_gin_routing_table_figures.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Login-node safe: conda PATH only (skip torch import check in common_env.sh).
export GNNPLUS_LIGHTWEIGHT_ENV=1
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

analysis_dir="${GCN_GIN_TABLE_ANALYSIS_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
lr_tag="${GCN_GIN_TABLE_LR_TAG:-lr001}"
mask_track="${GCN_GIN_TABLE_MASK_TRACK:-toy}"

python scripts/synthetic/gcn_gin_routing_table_figures.py \
  --analysis-dir "${analysis_dir}" \
  --lr-tag "${lr_tag}" \
  --mask-track "${mask_track}"

echo "Table figures → ${analysis_dir}/paper_figures/fig05_pairwise_baseline_table.png"
echo "Table figures → ${analysis_dir}/paper_figures/fig06_mask_ablation_table.png"
