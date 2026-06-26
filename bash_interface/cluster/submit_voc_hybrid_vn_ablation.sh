#!/usr/bin/env bash
# VOC-SP: anchor mq8f0mx2 vs +{1,2,4} virtual nodes (seed 0).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_voc_hybrid_vn_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-0}"
PARALLEL="${VN_ABLATION_PARALLEL:-4}"

echo "→ VOC hybrid VN ablation: tasks 1–4 (baseline + vn 1/2/4), seed=${SEED}, mem=128GB"
sbatch \
    --job-name=voc_hybrid_vn \
    --array="1-4%${PARALLEL}" \
    --mem=128GB \
    --time=120:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_voc_hybrid_vn_ablation.sh

echo ""
echo "W&B: voc_hybrid_a2g4_dh48_anchor_seed${SEED} | _vn1_ | _vn2_ | _vn4_"
echo "Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/mq8f0mx2"
echo "Metric: best/test_f1 (anchor ~0.451)"
