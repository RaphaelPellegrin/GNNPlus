#!/usr/bin/env bash
# Submit Table 5 SiGMA-lite (7 datasets × 5 seeds = 35 jobs).
# Max 6 concurrent GPUs. Default: wait for SiGMA/MMA seed15 array 53598773.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_sigma_lite.sh
#
# Smoke (peptides_func seed0 → task 1):
#   PAPER_T5_LITE_ARRAY=1 PAPER_T5_LITE_PARALLEL=1 PAPER_T5_LITE_DEPENDENCY= \
#     bash bash_interface/cluster/submit_paper_table5_sigma_lite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_LITE_NUM_SEEDS:-5}"
NUM_DATASETS=7
NUM_TASKS="${PAPER_T5_LITE_NUM_TASKS:-$((NUM_DATASETS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_LITE_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_LITE_PARALLEL:-6}"
PARTITION="${PAPER_T5_LITE_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_LITE_NICE:-10000}"
MEM="${PAPER_T5_LITE_MEM:-128GB}"
TIME="${PAPER_T5_LITE_TIME:-120:00:00}"
GATE_ACT="${PAPER_T5_LITE_GATE_ACT:-sigmoid}"
# Wait for Table 5 SiGMA/MMA 15-seed fill (53598773) to leave the queue.
DEPENDENCY="${PAPER_T5_LITE_DEPENDENCY:-afterany:53598773}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_paper_table5_sigma_lite] GNNPLUS_OUT_DIR unset -> ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_paper_table5_sigma_lite.sh

sbatch_args=(
    --parsable
    --job-name=t5_sigma_lite
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/t5_sigma_lite_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_LITE_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_LITE_NUM_TASKS="${NUM_TASKS}",PAPER_T5_LITE_GATE_ACT="${GATE_ACT}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ -n "${DEPENDENCY}" ]; then
    sbatch_args+=(--dependency="${DEPENDENCY}")
fi
if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(sbatch "${sbatch_args[@]}" bash_interface/cluster/run_paper_table5_sigma_lite.sh)"

cat <<EOF

=== Table 5 SiGMA-lite submitted ===
  ARRAY JOBID:   ${job_id}
  Dependency:    ${DEPENDENCY:-<none>}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (7 ds × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Gate act:      ${GATE_ACT}
  Logs:          logs_gnnplus/t5_sigma_lite_${job_id}_<TASK>.log

  Recipe: same Table 5 SiGMA anchors/heads; model.type=sigma_lite
          (gate after concat). Seeds match Table 5 (PATTERN 5–9).

  W&B: paper_T5_<ds>_SiGMA_lite  (pattern: paper_T5_pattern_gritvn4_SiGMA_lite)

  Paste JOBID into CLUSTER_LAUNCHES.md / Paper_table5_sigma_lite.md
EOF
