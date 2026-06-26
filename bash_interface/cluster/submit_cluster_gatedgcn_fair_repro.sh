#!/usr/bin/env bash
# Fair repro: CLUSTER GatedGCN+ standard vs +1 attention (seed 1, anchor n4unldzn).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_cluster_gatedgcn_fair_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-1}"

echo "→ CLUSTER GatedGCN+ fair repro: tasks 1–2 (standard + hybrid a1), seed=${SEED}, mem=128GB"
sbatch \
    --job-name=cluster_gatedgcn_fair \
    --array=1-2%2 \
    --mem=128GB \
    --time=120:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_cluster_gatedgcn_fair_repro.sh

echo ""
echo "W&B: cluster_gatedgcn_seed${SEED}_cluster | cluster_gatedgcn_seed${SEED}_repro_hybrid_attn1"
echo "Metric: best/test_accuracy-SBM (target standard ~0.79 from paper)"
