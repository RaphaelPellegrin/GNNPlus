#!/usr/bin/env bash
# Relaunch NCI1 SiGMA (homo/hetero) to beat GCN 80.51±0.71%.
#
# Changes vs 37434534:
#   LR ∈ {5e-4, 2e-3}   (was {1e-3, 1e-2}; prior best was 1e-3)
#   max_epoch=2000       (was 1000)
#   schedule_patience=100 (was 50)
#   parallel up to 10 GPUs
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_tu_nci1_sigma_retry.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_NCI1_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_NCI1_NUM_VARIANTS:-4}"
NUM_TASKS="${TU_NCI1_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_NCI1_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_NCI1_PARALLEL:-10}"
PARTITION="${TU_NCI1_PARTITION:-mweber_gpu}"
NICE="${TU_NCI1_NICE:-10000}"
MEM="${TU_NCI1_MEM:-64GB}"
TIME="${TU_NCI1_TIME:-96:00:00}"
MAX_EPOCH="${TU_NCI1_MAX_EPOCH:-2000}"
PATIENCE="${TU_NCI1_PATIENCE:-100}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_nci1_sigma_retry] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=tu_nci1_sigma
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_nci1_sigma_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_NCI1_NUM_SEEDS="${NUM_SEEDS}",TU_NCI1_NUM_TASKS="${NUM_TASKS}",TU_NCI1_MAX_EPOCH="${MAX_EPOCH}",TU_NCI1_PATIENCE="${PATIENCE}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_nci1_sigma_retry.sh
)"

cat <<EOF

=== NCI1 SiGMA retry submitted (beat GCN 80.51%) ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (4 variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  max_epoch:     ${MAX_EPOCH}  (patience=${PATIENCE})
  LRs:           5e-4, 2e-3
  Logs:          logs_gnnplus/tu_nci1_sigma_${job_id}_<TASK>.log

  Task map (blocks of ${NUM_SEEDS}):
    1–5   SiGMA_homo   lr=5e-4
    6–10  SiGMA_homo   lr=2e-3
    11–15 SiGMA_hetero lr=5e-4
    16–20 SiGMA_hetero lr=2e-3

  W&B: tu_hh_nci1_{SiGMA_homo,SiGMA_hetero}_{lr5e4,lr2e3}_ep${MAX_EPOCH}

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group tu_hh_nci1_SiGMA_hetero_lr5e4_ep${MAX_EPOCH} --metric best_test_perf --state finished

  Paste JOBID into Paper_tu_sigma_homo_hetero.md

EOF
