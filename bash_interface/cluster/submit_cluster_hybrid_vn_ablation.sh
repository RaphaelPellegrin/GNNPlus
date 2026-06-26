#!/usr/bin/env bash
# CLUSTER: anchor o6owwoqp vs +{1,2,4} virtual nodes (seed 1).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_cluster_hybrid_vn_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-1}"
PARALLEL="${VN_ABLATION_PARALLEL:-4}"

echo "→ CLUSTER hybrid VN ablation: tasks 1–4 (baseline + vn 1/2/4), seed=${SEED}, mem=128GB"
sbatch \
    --job-name=cluster_hybrid_vn \
    --array="1-4%${PARALLEL}" \
    --mem=128GB \
    --time=120:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_cluster_hybrid_vn_ablation.sh

echo ""
echo "W&B: cluster_hybrid_a1g1_dh64_anchor_seed${SEED} | _vn1_ | _vn2_ | _vn4_"
echo "Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o6owwoqp"
echo "Metric: best/test_accuracy-SBM (paper GatedGCN+ ~0.791; anchor ~0.790)"
