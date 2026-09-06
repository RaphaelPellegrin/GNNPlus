#!/usr/bin/env bash
# Submit TU Tables 17/18 SiGMA (hetero) ungated ablation.
#
# 2 tables × 6 datasets × 2 LRs × 5 seeds = 120 jobs.
# Override: gnn.hybrid.gate=none on hetero a2g4 anchors (dh16 / dh4).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_tu_sigma_ungated.sh
#
# Tab.17 only (tasks 1–60) / Tab.18 only (61–120):
#   TU_UNGATED_ARRAY=1-60  bash bash_interface/cluster/submit_tu_sigma_ungated.sh
#   TU_UNGATED_ARRAY=61-120 bash bash_interface/cluster/submit_tu_sigma_ungated.sh
#
# Smoke (MUTAG Tab.17 both LRs, 1 seed → tasks 1,6):
#   TU_UNGATED_ARRAY=1,6 TU_UNGATED_PARALLEL=2 \
#     bash bash_interface/cluster/submit_tu_sigma_ungated.sh
#
# Paste printed JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_UNGATED_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_UNGATED_NUM_VARIANTS:-2}"
NUM_DATASETS="${TU_UNGATED_NUM_DATASETS:-6}"
NUM_TABLES="${TU_UNGATED_NUM_TABLES:-2}"
NUM_TASKS="${TU_UNGATED_NUM_TASKS:-$((NUM_TABLES * NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_UNGATED_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_UNGATED_PARALLEL:-20}"
PARTITION="${TU_UNGATED_PARTITION:-mweber_gpu}"
NICE="${TU_UNGATED_NICE:-10000}"
MEM="${TU_UNGATED_MEM:-128GB}"
TIME="${TU_UNGATED_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_ungated] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_sigma_ungated.sh

sbatch_args=(
    --parsable
    --job-name=tu_sigma_ung
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_sigma_ung_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_UNGATED_NUM_SEEDS="${NUM_SEEDS}",TU_UNGATED_NUM_VARIANTS="${NUM_VARIANTS}",TU_UNGATED_NUM_DATASETS="${NUM_DATASETS}",TU_UNGATED_NUM_TABLES="${NUM_TABLES}",TU_UNGATED_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_ungated.sh
)"

cat <<EOF

=== TU SiGMA ungated (Tab.17 + Tab.18) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_TABLES} tables × ${NUM_DATASETS} ds × ${NUM_VARIANTS} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_sigma_ung_${job_id}_<TASK>.log
  Override:      gnn.hybrid.gate=none  (hetero a2g4)
  Tab.17 (1–60): d_h=16 → W&B tu_hh_<ds>_SiGMA_ungated_{lr001,lr01}
  Tab.18 (61–120): d_h=4 → W&B tu_1x_<ds>_SiGMA_ungated_{lr001,lr01}
  Outs:          \$GNNPLUS_OUT_DIR/tu_sigma_{homo_hetero,1x_gcn}/<ds>_SiGMA_ungated_<lr>_seed<s>/
  Docs:          Paper_tu_sigma_homo_hetero.md

  Per-dataset block (10 tasks, seeds 0–4):
    +0–4   ungated lr=0.001
    +5–9   ungated lr=0.01
  Dataset order: MUTAG → ENZYMES → PROTEINS → COLLAB → IMDB-BINARY → REDDIT-BINARY
  Table order:   Tab.17 (dh16) then Tab.18 (dh4)

  Paper protocol: report better LR mean±std; compare to gated SiGMA (hetero).

  Paste JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

EOF
