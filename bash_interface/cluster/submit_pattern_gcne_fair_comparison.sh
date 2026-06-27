#!/usr/bin/env bash
# PATTERN GCNE baseline vs hybrid fair repro (1 and 2 attention heads).
#
# Baseline anchor: qcz7umtl (pattern_gcne_seed2_cluster, SBM ≈ 0.866)
#   configs/gcn/pattern.yaml — custom_gnn gcne MP stack (NOT plain GCN).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_pattern_gcne_fair_comparison.sh
#
# Skip baseline (already have qcz7umtl): PATTERN_FAIR_TASKS=2-5 bash .../submit_...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-2}"
TASKS="${PATTERN_FAIR_TASKS:-1-5}"
PARALLEL="${PATTERN_FAIR_PARALLEL:-2}"

echo "→ PATTERN GCNE fair comparison (seed=${SEED}): tasks ${TASKS}"
echo "   Baseline: configs/gcn/pattern.yaml (gcne MP — same as qcz7umtl)"
echo "   Hybrid:   pattern-gcne-repro-a{1,2}.yaml, lr ∈ {0.001, 0.002}"
sbatch \
    --job-name=pattern_fair \
    --array="${TASKS}%${PARALLEL}" \
    --mem=128GB \
    --time=120:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_pattern_gcne_fair_comparison.sh

echo ""
echo "Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/qcz7umtl"
echo "Metric: test/accuracy-SBM"
