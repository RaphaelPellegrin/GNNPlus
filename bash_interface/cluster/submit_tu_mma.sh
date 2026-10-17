#!/usr/bin/env bash
# Submit TU Tables 17/18 MMA (ungated hetero a2g4): 2 tables × 6 ds × 2 LRs × 5 seeds = 120.
#
# Prerequisites (cluster):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus   # or a worktree of this branch
#   git pull
#   bash bash_interface/cluster/submit_tu_mma.sh
#
# Task blocks (10 tasks each = 2 LRs × 5 seeds), dataset order
# MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY:
#   Tab.17 (d_h=16):  1-10, 11-20, 21-30, 31-40, 41-50, 51-60
#   Tab.18 (d_h=4):  61-70, 71-80, 81-90, 91-100, 101-110, 111-120
#
# Subsets:
#   TU_MMA_ARRAY=1-60   bash bash_interface/cluster/submit_tu_mma.sh   # Tab.17 only
#   TU_MMA_ARRAY=61-120 bash bash_interface/cluster/submit_tu_mma.sh   # Tab.18 only
# Smoke (MUTAG Tab.17, seed 0, both LRs → tasks 1,6):
#   TU_MMA_ARRAY=1,6 TU_MMA_PARALLEL=2 bash bash_interface/cluster/submit_tu_mma.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS=120
ARRAY_SPEC="${TU_MMA_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_MMA_PARALLEL:-20}"
PARTITION="${TU_MMA_PARTITION:-mweber_gpu}"
NICE="${TU_MMA_NICE:-10000}"
MEM="${TU_MMA_MEM:-128GB}"
TIME="${TU_MMA_TIME:-48:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_mma] GNNPLUS_OUT_DIR unset -> ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_mma.sh

sbatch_args=(
    --parsable
    --job-name=tu_mma
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_mma_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)
if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(sbatch "${sbatch_args[@]}" bash_interface/cluster/run_tu_mma.sh)"

cat <<EOF

=== TU MMA (Tab.17 + Tab.18) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (2 tables x 6 ds x 2 LR x 5 seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Logs:          logs_gnnplus/tu_mma_${job_id}_<TASK>.log
  W&B:           GNNPlus / tu_hh_<ds>_MMA_{lr001,lr01} (Tab.17), tu_1x_<ds>_MMA_{lr001,lr01} (Tab.18)
  Outs:          \$GNNPLUS_OUT_DIR/tu_mma/<t17|t18>_<ds>_MMA_<lr>_seed<s>/

  Paste JOBID into CLUSTER_LAUNCHES.md / Paper_tu_mma_sigma_lite.md.
  Report the better LR mean +- std per (table, dataset), as for SiGMA.
EOF
