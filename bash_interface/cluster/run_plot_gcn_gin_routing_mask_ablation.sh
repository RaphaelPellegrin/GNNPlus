#!/usr/bin/env bash
# Regenerate mask ablation bar chart from existing CSV (login node, no GPU).
#
# Usage:
#   bash bash_interface/cluster/run_plot_gcn_gin_routing_mask_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export GNNPLUS_LIGHTWEIGHT_ENV=1
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

out_dir="${GCN_GIN_MASK_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
ymin="${GCN_GIN_MASK_YMIN:-0.3}"

python scripts/synthetic/eval_gcn_gin_routing_masks.py \
  --out-dir "${out_dir}" \
  --plot-only \
  --ymin "${ymin}"

echo "Mask ablation figure → ${out_dir}/paper_figures/fig06_mask_ablation.png"
