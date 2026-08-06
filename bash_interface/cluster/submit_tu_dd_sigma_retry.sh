#!/usr/bin/env bash
# Relaunch DD SiGMA (homo/hetero × 2 LR × 5 seeds = 20) after batch-64 OOMs.
#
# Changes vs original 37434534:
#   train.batch_size=16  (was 64)
#   --mem=128GB          (was 64GB)
#   W&B groups: tu_hh_dd_{SiGMA_homo,SiGMA_hetero}_{lr001,lr01}_bs16
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_tu_dd_sigma_retry.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_DD_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_DD_NUM_VARIANTS:-4}"
NUM_TASKS="${TU_DD_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_DD_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_DD_PARALLEL:-4}"
PARTITION="${TU_DD_PARTITION:-mweber_gpu}"
NICE="${TU_DD_NICE:-10000}"
MEM="${TU_DD_MEM:-128GB}"
TIME="${TU_DD_TIME:-96:00:00}"
BATCH_SIZE="${TU_DD_BATCH_SIZE:-16}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_dd_sigma_retry] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=tu_dd_sigma
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_dd_sigma_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_DD_NUM_SEEDS="${NUM_SEEDS}",TU_DD_NUM_TASKS="${NUM_TASKS}",TU_DD_BATCH_SIZE="${BATCH_SIZE}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_dd_sigma_retry.sh
)"

cat <<EOF

=== DD SiGMA retry submitted (batch=${BATCH_SIZE}, mem=${MEM}) ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (4 SiGMA variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL}
  Logs:          logs_gnnplus/tu_dd_sigma_${job_id}_<TASK>.log
  Outs:          \$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/dd_<variant>_<lr>_bs${BATCH_SIZE}_seed<s>/

  Task map (blocks of ${NUM_SEEDS}):
    1–5   SiGMA_homo   lr=0.001
    6–10  SiGMA_homo   lr=0.01
    11–15 SiGMA_hetero lr=0.001
    16–20 SiGMA_hetero lr=0.01

  W&B: tu_hh_dd_{SiGMA_homo,SiGMA_hetero}_{lr001,lr01}_bs${BATCH_SIZE}

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group tu_hh_dd_SiGMA_hetero_lr001_bs${BATCH_SIZE} --metric best_test_perf --state finished

  Paste JOBID into Paper_tu_sigma_homo_hetero.md

EOF
