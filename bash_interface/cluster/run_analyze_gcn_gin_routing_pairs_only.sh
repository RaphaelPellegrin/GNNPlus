#!/usr/bin/env bash
# Evaluate GCN/GIN routing models on opposite-sign twin pairs only (test split).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/gcn_gin_routing}"
DATASET_DIR="${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}"
OUT_DIR="${OUT_DIR:-results/gcn_gin_routing/analysis/pairs_only}"
LR_TAG="${LR_TAG:-lr001}"

python scripts/synthetic/analyze_gcn_gin_routing_pairs_only.py \
  --results-root "${RESULTS_ROOT}" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --lr-tag "${LR_TAG}"

echo "Pairs-only figures -> ${OUT_DIR}/paper_figures/"
