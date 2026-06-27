#!/usr/bin/env bash
# CLUSTER hybrid refine around ht9bntg2 (best SBM ≈ 0.793).
#
# Grid: d_h {48,64,96,128} × lr {8e-4,1e-3,1.492e-3,2e-3} × attn_mask {full,graph_restricted}
# 32 tasks; task 7 = ht9bntg2 repro (d_h=64, lr=0.001492, full).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_cluster_hybrid_ht9bntg2_refine_sweep.sh
#
# W&B sweep alternative:
#   bash bash_interface/sweeps/create_sweep.sh \
#     bash_interface/sweeps/cluster_hybrid_ht9bntg2_refine_sweep.yaml
#   SWEEP_ARRAY_TASKS=32 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=1 \
#     SWEEP_SLURM_MEM=128GB SWEEP_SLURM_TIME=120:00:00 \
#     bash bash_interface/sweeps/relaunch_sweep_agents.sh \
#       cluster weber-geoml-harvard-university/GNNPlus/$(cat bash_interface/sweeps/.last_sweep_id)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-1}"
PARALLEL="${CLUSTER_REFINE_PARALLEL:-4}"

echo "→ CLUSTER hybrid ht9bntg2 refine: 32 tasks, seed=${SEED}, mem=128GB"
echo "   d_h ∈ {48,64,96,128}; lr ∈ {0.0008,0.001,0.001492,0.002}; attn_mask ∈ {full,graph_restricted}"
echo "   Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ht9bntg2"
sbatch \
    --job-name=cluster_refine \
    --array="1-32%${PARALLEL}" \
    --mem=128GB \
    --time=120:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_cluster_hybrid_ht9bntg2_refine_sweep.sh

echo ""
echo "Task 7 reproduces ht9bntg2 (d_h=64, lr=0.001492, full). Metric: test/accuracy-SBM"
