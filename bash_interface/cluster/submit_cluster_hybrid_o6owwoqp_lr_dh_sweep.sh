#!/usr/bin/env bash
# CLUSTER hybrid o6owwoqp — grid over hybrid_d_h {64,128} × base_lr (log-spaced 3e-4…3e-3).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_cluster_hybrid_o6owwoqp_lr_dh_sweep.sh
#
# W&B sweep alternative:
#   bash bash_interface/sweeps/create_sweep.sh \
#     bash_interface/sweeps/cluster_hybrid_o6owwoqp_lr_dh_sweep.yaml
#   SWEEP_ARRAY_TASKS=8 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=1 \
#     SWEEP_SLURM_MEM=128GB SWEEP_SLURM_TIME=120:00:00 \
#     bash bash_interface/sweeps/relaunch_sweep_agents.sh \
#       cluster weber-geoml-harvard-university/GNNPlus/$(cat bash_interface/sweeps/.last_sweep_id)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

SEED="${SEED:-1}"
PARALLEL="${LR_DH_SWEEP_PARALLEL:-4}"

echo "→ CLUSTER hybrid LR×d_h grid (o6owwoqp anchor): 8 tasks, seed=${SEED}, mem=128GB"
echo "   d_h ∈ {64, 128}; lr ∈ {0.0003, 0.000669, 0.001492, 0.003}"
sbatch \
    --job-name=cluster_lr_dh \
    --array="1-8%${PARALLEL}" \
    --mem=128GB \
    --time=120:00:00 \
    --export="ALL,SEED=${SEED},ENV_NAME=gnnplus" \
    bash_interface/cluster/run_cluster_hybrid_o6owwoqp_lr_dh_sweep.sh

echo ""
echo "Anchor repro (original o6owwoqp): d_h=64, lr≈0.000336 — closest grid point task 1 (lr=0.0003)"
echo "https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o6owwoqp"
echo "Metric: best/test_accuracy-SBM (anchor ~0.790)"
