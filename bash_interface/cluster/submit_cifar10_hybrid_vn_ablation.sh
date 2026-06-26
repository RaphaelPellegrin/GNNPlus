#!/usr/bin/env bash
# CIFAR10: anchor t8prvgqr vs +{1,2,4} virtual nodes (seed 0).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_cifar10_hybrid_vn_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-0}"
PARALLEL="${VN_ABLATION_PARALLEL:-4}"

echo "→ CIFAR10 hybrid VN ablation: tasks 1–4 (baseline + vn 1/2/4), seed=${SEED}, mem=64GB"
sbatch \
    --job-name=cifar10_hybrid_vn \
    --array="1-4%${PARALLEL}" \
    --mem=64GB \
    --time=96:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_cifar10_hybrid_vn_ablation.sh

echo ""
echo "W&B: cifar10_hybrid_a4g4_dh128_anchor_seed${SEED} | _vn1_ | _vn2_ | _vn4_"
echo "Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/t8prvgqr"
echo "Metric: best/test_accuracy (paper GatedGCN+ ~0.7006; anchor ~0.7998)"
