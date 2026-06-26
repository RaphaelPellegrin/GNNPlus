#!/usr/bin/env bash
# COCO hybrid Bayes sweep kb2ye07d — queue N more trials on a single GPU.
#
# Submits SLURM array 1-N with %1 parallelism → at most ONE GPU busy at a time.
# Each task runs exactly one wandb trial (RUNS_PER_AGENT=1) with 192h walltime.
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_coco_hybrid_kb2ye07d_more_runs.sh
#
# Optional env:
#   COCO_SWEEP_RUNS=32          # default 32
#   SWEEP_ID=weber-geoml-harvard-university/GNNPlus/kb2ye07d

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_RUNS="${COCO_SWEEP_RUNS:-32}"
SWEEP_ID="${SWEEP_ID:-weber-geoml-harvard-university/GNNPlus/kb2ye07d}"

echo "→ COCO sweep ${SWEEP_ID##*/}: ${NUM_RUNS} trials, 1 GPU max (%1), 128GB, 192h/job"
echo "   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/kb2ye07d"

SWEEP_ARRAY_TASKS="${NUM_RUNS}" \
SWEEP_ARRAY_PARALLEL=1 \
RUNS_PER_AGENT=1 \
SWEEP_SLURM_MEM=128GB \
SWEEP_SLURM_TIME=192:00:00 \
bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    coco "${SWEEP_ID}"

echo ""
echo "Monitor: squeue -u \$USER | grep gnnplus_sweep_coco"
echo "Log:     logs_gnnplus/sweep_agent_<JOBID>_1.log  (tasks run one-at-a-time)"
