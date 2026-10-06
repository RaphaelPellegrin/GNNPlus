#!/usr/bin/env bash
# Submit TU Tables 17/18 SiGMA-lite: 2 tables x 6 datasets x 2 LRs x 5 seeds = 120 jobs.
#
# Run from a checkout of branch paper_sigma_lite (model.type sigma_lite), e.g. a
# worktree next to the main cluster checkout:
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git fetch origin
#   git worktree add -b paper_sigma_lite ../GNNPlus-paper-sigma-lite origin/paper_sigma_lite
# then:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus-paper-sigma-lite
#   git pull
#   bash bash_interface/cluster/submit_sigma_lite_tu.sh
#
# Task blocks (10 tasks each = 2 LRs x 5 seeds), dataset order
# MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY:
#   Tab.17 (d_h=16):  1-10, 11-20, 21-30, 31-40, 41-50, 51-60
#   Tab.18 (d_h=4):  61-70, 71-80, 81-90, 91-100, 101-110, 111-120
#
# Subsets:
#   SIGMA_LITE_ARRAY=1-60   bash bash_interface/cluster/submit_sigma_lite_tu.sh   # Tab.17 only
#   SIGMA_LITE_ARRAY=1,6    SIGMA_LITE_PARALLEL=2 bash ...                          # smoke: MUTAG Tab.17, seed 0, both LRs
# GELU gate (Gated-GPS) instead of sigmoid:
#   SIGMA_LITE_GATE_ACT=gelu bash bash_interface/cluster/submit_sigma_lite_tu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS=120
ARRAY_SPEC="${SIGMA_LITE_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_LITE_PARALLEL:-20}"
PARTITION="${SIGMA_LITE_PARTITION:-mweber_gpu}"
NICE="${SIGMA_LITE_NICE:-10000}"
MEM="${SIGMA_LITE_MEM:-128GB}"
TIME="${SIGMA_LITE_TIME:-48:00:00}"
GATE_ACT="${SIGMA_LITE_GATE_ACT:-sigmoid}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_sigma_lite_tu] GNNPLUS_OUT_DIR unset -> ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_sigma_lite_tu.sh

sbatch_args=(
    --parsable
    --job-name=tu_sigma_lite
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_sigma_lite_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_LITE_GATE_ACT="${GATE_ACT}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)
if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(sbatch "${sbatch_args[@]}" bash_interface/cluster/run_sigma_lite_tu.sh)"

cat <<EOF

=== TU SiGMA-lite (Tab.17 + Tab.18) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (2 tables x 6 ds x 2 LR x 5 seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Gate:          ${GATE_ACT}
  Logs:          logs_gnnplus/tu_sigma_lite_${job_id}_<TASK>.log
  W&B:           GNNPlus / tu_hh_<ds>_SiGMA_lite_{lr001,lr01} (Tab.17), tu_1x_<ds>_SiGMA_lite_{lr001,lr01} (Tab.18)
  Outs:          \$GNNPLUS_OUT_DIR/tu_sigma_lite/<t17|t18>_<ds>_SiGMA_lite_<lr>_seed<s>/

  Report the better LR mean +- std per (table, dataset), as for SiGMA.
EOF
