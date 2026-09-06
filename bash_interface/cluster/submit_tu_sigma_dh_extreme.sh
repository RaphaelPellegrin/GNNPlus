#!/usr/bin/env bash
# Submit TU Tab.18-style extreme d_h ∈ {1, 2}: SiGMA hetero gated vs ungated.
#
# 2 dh × 6 datasets × 4 variants × 5 seeds = 240 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_tu_sigma_dh_extreme.sh
#
# d_h=1 only (tasks 1–120) / d_h=2 only (121–240):
#   TU_DH_EXT_ARRAY=1-120   bash bash_interface/cluster/submit_tu_sigma_dh_extreme.sh
#   TU_DH_EXT_ARRAY=121-240 bash bash_interface/cluster/submit_tu_sigma_dh_extreme.sh
#
# Smoke (MUTAG d_h=1 gated+ungated lr001, 1 seed → tasks 1,11):
#   TU_DH_EXT_ARRAY=1,11 TU_DH_EXT_PARALLEL=2 \
#     bash bash_interface/cluster/submit_tu_sigma_dh_extreme.sh
#
# Paste printed JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_DH_EXT_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_DH_EXT_NUM_VARIANTS:-4}"
NUM_DATASETS="${TU_DH_EXT_NUM_DATASETS:-6}"
NUM_TABLES="${TU_DH_EXT_NUM_TABLES:-2}"
NUM_TASKS="${TU_DH_EXT_NUM_TASKS:-$((NUM_TABLES * NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_DH_EXT_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_DH_EXT_PARALLEL:-20}"
PARTITION="${TU_DH_EXT_PARTITION:-mweber_gpu}"
NICE="${TU_DH_EXT_NICE:-10000}"
MEM="${TU_DH_EXT_MEM:-128GB}"
TIME="${TU_DH_EXT_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_dh_extreme] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_sigma_dh_extreme.sh

sbatch_args=(
    --parsable
    --job-name=tu_dh_ext
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_dh_ext_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_DH_EXT_NUM_SEEDS="${NUM_SEEDS}",TU_DH_EXT_NUM_VARIANTS="${NUM_VARIANTS}",TU_DH_EXT_NUM_DATASETS="${NUM_DATASETS}",TU_DH_EXT_NUM_TABLES="${NUM_TABLES}",TU_DH_EXT_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_dh_extreme.sh
)"

cat <<EOF

=== TU SiGMA extreme d_h (Tab.18-style) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_TABLES} dh × ${NUM_DATASETS} ds × ${NUM_VARIANTS} var × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_dh_ext_${job_id}_<TASK>.log
  Configs:       configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-dh{1,2}-anchor.yaml
  Outs:          \$GNNPLUS_OUT_DIR/tu_sigma_dh_extreme/<ds>_<variant>_dh{1,2}_<lr>_seed<s>/

  d_h=1  → tasks 1–120    W&B tu_dh1_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}
  d_h=2  → tasks 121–240  W&B tu_dh2_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}

  Per-dataset block (20 tasks, seeds 0–4):
    +0–4    gated   lr=0.001
    +5–9    gated   lr=0.01
    +10–14  ungated lr=0.001
    +15–19  ungated lr=0.01
  Dataset order: MUTAG → ENZYMES → PROTEINS → COLLAB → IMDB-BINARY → REDDIT-BINARY

  Paper protocol: better LR per family; compare gated hetero vs ungated at same d_h.
  Hypothesis (App. H): gated > ungated more clearly at d_h=1 than at Tab.18 d_h=4.

  Paste JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

EOF
