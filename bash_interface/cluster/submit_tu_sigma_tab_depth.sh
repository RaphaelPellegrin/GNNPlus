#!/usr/bin/env bash
# Submit TU Tab.17/18 SiGMA hetero gated vs ungated at L ∈ {4, 2, 1}.
#
# 3 depths × 2 tables × 6 datasets × 4 variants × 5 seeds = 720 jobs.
# Override: gnn.layers_mp ∈ {4,2,1}; ungated uses gnn.hybrid.gate=none.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_tu_sigma_tab_depth.sh
#
# Per depth (240 tasks each):
#   TU_TAB_L_ARRAY=1-240    bash ...   # L=4
#   TU_TAB_L_ARRAY=241-480  bash ...   # L=2
#   TU_TAB_L_ARRAY=481-720  bash ...   # L=1
#
# Tab.17 only within L=4 (dh16, tasks 1–120 of the L=4 block):
#   TU_TAB_L_ARRAY=1-120 bash bash_interface/cluster/submit_tu_sigma_tab_depth.sh
#
# Smoke (MUTAG L4 Tab.17 gated+ungated lr001 seed0 → tasks 1,11):
#   TU_TAB_L_ARRAY=1,11 TU_TAB_L_PARALLEL=2 TU_TAB_L_NICE=0 \
#     bash bash_interface/cluster/submit_tu_sigma_tab_depth.sh
#
# Paste printed JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_TAB_L_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_TAB_L_NUM_VARIANTS:-4}"
NUM_DATASETS="${TU_TAB_L_NUM_DATASETS:-6}"
NUM_TABLES="${TU_TAB_L_NUM_TABLES:-2}"
NUM_DEPTHS="${TU_TAB_L_NUM_DEPTHS:-3}"
NUM_TASKS="${TU_TAB_L_NUM_TASKS:-$((NUM_DEPTHS * NUM_TABLES * NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_TAB_L_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_TAB_L_PARALLEL:-20}"
PARTITION="${TU_TAB_L_PARTITION:-mweber_gpu}"
NICE="${TU_TAB_L_NICE:-10000}"
MEM="${TU_TAB_L_MEM:-128GB}"
TIME="${TU_TAB_L_TIME:-96:00:00}"
LAYERS="${TU_TAB_L_LS:-4 2 1}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_tab_depth] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_sigma_tab_depth.sh

sbatch_args=(
    --parsable
    --job-name=tu_tab_L
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_tab_L_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_TAB_L_NUM_SEEDS="${NUM_SEEDS}",TU_TAB_L_NUM_VARIANTS="${NUM_VARIANTS}",TU_TAB_L_NUM_DATASETS="${NUM_DATASETS}",TU_TAB_L_NUM_TABLES="${NUM_TABLES}",TU_TAB_L_NUM_DEPTHS="${NUM_DEPTHS}",TU_TAB_L_NUM_TASKS="${NUM_TASKS}",TU_TAB_L_LS="${LAYERS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_tab_depth.sh
)"

cat <<EOF

=== TU SiGMA Tab.17/18 depth ablation (L∈{${LAYERS}}) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}
                 (${NUM_DEPTHS} L × ${NUM_TABLES} tables × ${NUM_DATASETS} ds × ${NUM_VARIANTS} var × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_tab_L_${job_id}_<TASK>.log
  Override:      gnn.layers_mp ∈ {${LAYERS}}; ungated → gnn.hybrid.gate=none
  Out:           \$GNNPLUS_OUT_DIR/tu_sigma_tab_depth/

  Depth blocks (240 tasks each):
    L=4 → 1–240 · L=2 → 241–480 · L=1 → 481–720
  Within each L: Tab.17 dh16 (1–120) then Tab.18 dh4 (121–240)

  Per-dataset block (20 tasks, seeds 0–4):
    +0–4    gated   lr=0.001
    +5–9    gated   lr=0.01
    +10–14  ungated lr=0.001
    +15–19  ungated lr=0.01
  Dataset order: MUTAG → ENZYMES → PROTEINS → COLLAB → IMDB-BINARY → REDDIT-BINARY

  W&B Tab.17: tu_L<k>_hh_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}
  W&B Tab.18: tu_L<k>_1x_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}

  Smoke: tasks 1,11 = MUTAG L4 Tab.17 gated/ungated lr001 seed0
  Docs:  Paper_tu_sigma_homo_hetero.md

  Paste JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

EOF
