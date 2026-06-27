#!/usr/bin/env bash
# COCO-SP GatedGCN+ baseline vs hybrid fair repro (1 and 2 attention heads).
#
# Baseline anchor: 5b4z9l3u (coco_gatedgcn_seed1_cluster)
#   configs/gatedgcn/coco.yaml — custom_gnn gatedgcn MP stack.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_coco_gatedgcn_fair_comparison.sh
#
# Skip baseline (already have 5b4z9l3u): COCO_FAIR_TASKS=2-5 bash .../submit_...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-1}"
TASKS="${COCO_FAIR_TASKS:-1-5}"
PARALLEL="${COCO_FAIR_PARALLEL:-2}"

echo "→ COCO GatedGCN+ fair comparison (seed=${SEED}): tasks ${TASKS}"
echo "   Baseline: configs/gatedgcn/coco.yaml (gatedgcn MP — same as 5b4z9l3u)"
echo "   Hybrid:   coco-hybrid-5b4z9l3u-a{1,2}g1.yaml, lr ∈ {0.001, 0.002}"
sbatch \
    --job-name=coco_fair \
    --array="${TASKS}%${PARALLEL}" \
    --mem=128GB \
    --time=192:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_coco_gatedgcn_fair_comparison.sh

echo ""
echo "Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5b4z9l3u"
echo "Metric: test/f1"
