#!/usr/bin/env bash
# Submit TU Tab.17/18 protocol: SiGMA hetero gated vs ungated × L × d_h × H.
#
# Default H ∈ {64, 8}:
#   6 ds × 5 L × 4 d_h × 2 H × 2 gate × 2 LR × 5 seeds = 4800 jobs.
#
# Narrow fill H ∈ {4, 2} (same task map; does not remape H=64/8 jobs):
#   TU_LDHH_HS="4 2" bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch (phased):
#   # smoke default H=64: MUTAG L1 dh1 gated+ungated lr001 seed0
#   TU_LDHH_ARRAY=1,11 TU_LDHH_PARALLEL=2 \
#     bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh
#   # smoke narrow H=4:
#   TU_LDHH_HS="4 2" TU_LDHH_ARRAY=1,11 TU_LDHH_PARALLEL=2 \
#     bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh
#   # MUTAG only (800 tasks)
#   TU_LDHH_ARRAY=1-800 bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh
#   TU_LDHH_HS="4 2" TU_LDHH_ARRAY=1-800 bash ...
#
# Paste JOBID into Paper_tu_sigma_depth_dh_h.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_LDHH_NUM_SEEDS:-5}"
NUM_LRS="${TU_LDHH_NUM_LRS:-2}"
NUM_GATES="${TU_LDHH_NUM_GATES:-2}"
NUM_DH="${TU_LDHH_NUM_DH:-4}"
NUM_L="${TU_LDHH_NUM_L:-5}"
NUM_DATASETS="${TU_LDHH_NUM_DATASETS:-6}"
TU_LDHH_HS="${TU_LDHH_HS:-64 8}"
# shellcheck disable=SC2206
HS_ARR=(${TU_LDHH_HS})
NUM_H="${TU_LDHH_NUM_H:-${#HS_ARR[@]}}"
if [ "${NUM_H}" -ne "${#HS_ARR[@]}" ]; then
    echo "TU_LDHH_NUM_H=${NUM_H} != |Hs|=${#HS_ARR[@]} (Hs=${TU_LDHH_HS})"
    exit 1
fi
NUM_TASKS="${TU_LDHH_NUM_TASKS:-$((NUM_DATASETS * NUM_L * NUM_DH * NUM_H * NUM_GATES * NUM_LRS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_LDHH_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_LDHH_PARALLEL:-20}"
PARTITION="${TU_LDHH_PARTITION:-mweber_gpu}"
NICE="${TU_LDHH_NICE:-10000}"
MEM="${TU_LDHH_MEM:-128GB}"
TIME="${TU_LDHH_TIME:-96:00:00}"
PER_DS=$((NUM_L * NUM_DH * NUM_H * NUM_GATES * NUM_LRS * NUM_SEEDS))
H0="${HS_ARR[0]}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_depth_dh_h] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

CFG="configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml"
if [ ! -f "${CFG}" ]; then
    echo "MISSING ${CFG}"
    exit 1
fi

chmod +x bash_interface/cluster/run_tu_sigma_depth_dh_h.sh

sbatch_args=(
    --parsable
    --job-name=tu_LdhH
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_LdhH_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_LDHH_HS="${TU_LDHH_HS}",TU_LDHH_NUM_SEEDS="${NUM_SEEDS}",TU_LDHH_NUM_LRS="${NUM_LRS}",TU_LDHH_NUM_GATES="${NUM_GATES}",TU_LDHH_NUM_H="${NUM_H}",TU_LDHH_NUM_DH="${NUM_DH}",TU_LDHH_NUM_L="${NUM_L}",TU_LDHH_NUM_DATASETS="${NUM_DATASETS}",TU_LDHH_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_depth_dh_h.sh
)"

cat <<EOF

=== TU SiGMA L × d_h × H gated/ungated submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}
                 (${NUM_DATASETS} ds × ${NUM_L} L × ${NUM_DH} d_h × ${NUM_H} H × ${NUM_GATES} gate × ${NUM_LRS} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_LdhH_${job_id}_<TASK>.log
  Base cfg:      ${CFG}  (overrides: layers_mp, dim_inner, d_h, gate)
  Out:           \$GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/

  Grid:
    L   ∈ {1, 2, 4, 8, 16}
    d_h ∈ {1, 2, 4, 16}
    H   ∈ {${TU_LDHH_HS}}
    gate: headwise vs none (ungated)

  Dataset blocks (${PER_DS} tasks each):
    MUTAG 1–${PER_DS} · ENZYMES $((PER_DS+1))–$((2*PER_DS)) · …
    (full 1–${NUM_TASKS})

  Per (ds,L,dh,H) block of 20:
    +0–4   gated   lr=0.001
    +5–9   gated   lr=0.01
    +10–14 ungated lr=0.001
    +15–19 ungated lr=0.01

  Smoke: tasks 1,11 = MUTAG L1 dh1 H${H0} gated/ungated lr001 seed0

  W&B: tu_L<k>_dh<m>_H<h>_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}
  Docs: Paper_tu_sigma_depth_dh_h.md

  Paste JOBID into Paper_tu_sigma_depth_dh_h.md + CLUSTER_LAUNCHES.md

EOF
