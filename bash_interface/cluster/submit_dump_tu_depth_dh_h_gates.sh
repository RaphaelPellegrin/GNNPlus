#!/usr/bin/env bash
# Submit gate dumps for TU L×d_h×H gated ckpts (default: MUTAG L=1 H=8 best LRs).
#
# 4 (d_h,lr) × 5 seeds = 20 jobs. Needs existing ckpts on $GNNPLUS_OUT_DIR.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh
#
# Smoke (dh1 lr01 seed0 = task 1):
#   TU_LDHH_GDUMP_ARRAY=1 TU_LDHH_GDUMP_PARALLEL=1 \
#     bash bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh
#
# Skip already-dumped:
#   GATE_DUMP_SKIP_EXISTING=1 bash bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_LDHH_GDUMP_NUM_SEEDS:-5}"
# Comma-separated (spaces break sbatch --export).
CELLS="${TU_LDHH_GDUMP_CELLS:-1:lr01,2:lr01,4:lr001,16:lr01}"
IFS=',' read -r -a CELL_ARR <<< "${CELLS}"
NUM_CELLS=${#CELL_ARR[@]}
NUM_TASKS="${TU_LDHH_GDUMP_NUM_TASKS:-$((NUM_CELLS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_LDHH_GDUMP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_LDHH_GDUMP_PARALLEL:-10}"
PARTITION="${TU_LDHH_GDUMP_PARTITION:-mweber_gpu}"
NICE="${TU_LDHH_GDUMP_NICE:-10000}"
MEM="${TU_LDHH_GDUMP_MEM:-32GB}"
TIME="${TU_LDHH_GDUMP_TIME:-02:00:00}"
L="${TU_LDHH_GDUMP_L:-1}"
H="${TU_LDHH_GDUMP_H:-8}"
DS="${TU_LDHH_GDUMP_DS:-mutag}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
  echo "[submit_dump_tu_depth_dh_h_gates] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_dump_tu_depth_dh_h_gates.sh

sbatch_args=(
  --parsable
  --job-name=tu_LdhH_gdmp
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --gpus=1
  --output="logs_gnnplus/tu_LdhH_gdmp_%A_%a.log"
  --export=ALL,ENV_NAME=gnnplus,PYTHONNOUSERSITE=1,TU_LDHH_GDUMP_NUM_SEEDS="${NUM_SEEDS}",TU_LDHH_GDUMP_CELLS="${CELLS}",TU_LDHH_GDUMP_NUM_TASKS="${NUM_TASKS}",TU_LDHH_GDUMP_L="${L}",TU_LDHH_GDUMP_H="${H}",TU_LDHH_GDUMP_DS="${DS}",TU_LDHH_GDUMP_DS_NAME="${TU_LDHH_GDUMP_DS_NAME:-MUTAG}",GATE_DUMP_EPOCH="${GATE_DUMP_EPOCH:--1}",GATE_DUMP_LEVEL="${GATE_DUMP_LEVEL:-graph}",GATE_DUMP_SKIP_EXISTING="${GATE_DUMP_SKIP_EXISTING:-0}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
  sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
  sbatch "${sbatch_args[@]}" \
    bash_interface/cluster/run_dump_tu_depth_dh_h_gates.sh
)"

cat <<EOF

=== TU L×d_h×H gate dump submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_CELLS} cells × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Slice:         ${DS} L=${L} H=${H} gated cells: ${CELLS}
  Out:           \$GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/<run>/gate_values_per_graph.pt
  Logs:          logs_gnnplus/tu_LdhH_gdmp_${job_id}_<TASK>.log

  Smoke: TU_LDHH_GDUMP_ARRAY=1  (MUTAG L1 dh1 H8 gated lr01 seed0)

  Paste JOBID into Paper_tu_sigma_depth_dh_h.md

EOF
