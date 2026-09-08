#!/usr/bin/env bash
# Submit joint L12+L4 aggregate→eval, dependent on the L4 select array.
#
# Does NOT touch L12-only outputs from 45265929
# (sigma_native_fair_v2_per_fold.json / sigma_grid_eval_native_fair_v2).
#
# Usage (cluster, after L4 select is submitted):
#   TU_ERRICA_DEPENDENCY_JOBID=45268464 \
#     bash bash_interface/cluster/submit_tu_errica_native_fair_v2_joint_agg_eval.sh
#
# Env knobs:
#   TU_ERRICA_DEPENDENCY_JOBID   required — L4 select JOBID (e.g. 45268464)
#   TU_ERRICA_DEPENDENCY_TYPE    afterok (default) | afterany
#   TU_ERRICA_PARTITION / MEM / TIME / NICE / DRY_RUN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEPENDENCY_JOBID="${TU_ERRICA_DEPENDENCY_JOBID:-}"
if [ -z "${DEPENDENCY_JOBID}" ]; then
  echo "[native_fair_v2_joint agg_eval] set TU_ERRICA_DEPENDENCY_JOBID=<L4 select JOBID>"
  echo "  e.g. TU_ERRICA_DEPENDENCY_JOBID=45268464 bash bash_interface/cluster/submit_tu_errica_native_fair_v2_joint_agg_eval.sh"
  exit 1
fi

DEPENDENCY_TYPE="${TU_ERRICA_DEPENDENCY_TYPE:-afterok}"
PARTITION="${TU_ERRICA_PARTITION:-mweber_gpu}"
MEM="${TU_ERRICA_MEM:-32GB}"
TIME="${TU_ERRICA_TIME:-4:00:00}"
NICE="${TU_ERRICA_NICE:-0}"
DRY_RUN="${TU_ERRICA_DRY_RUN:-0}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_native_fair_v2_joint"
mkdir -p "${LOGDIR}"

echo "[native_fair_v2_joint agg_eval] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
echo "[native_fair_v2_joint agg_eval] partition=${PARTITION} time=${TIME} mem=${MEM}"
echo "[native_fair_v2_joint agg_eval] selection → configs/tu_errica/selections/sigma_native_fair_v2_joint_per_fold.json"
echo "[native_fair_v2_joint agg_eval] eval campaign → sigma_grid_eval_native_fair_v2_joint"
echo "[native_fair_v2_joint agg_eval] (L12-only paths untouched)"
echo "[native_fair_v2_joint agg_eval] logs → ${LOGDIR}/agg_eval_<JOBID>.log"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[native_fair_v2_joint agg_eval] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_nfv2_joint_agg \
  --dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --cpus-per-task=4 \
  --export=ALL,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/agg_eval_%j.log" \
  "${SCRIPT_DIR}/run_tu_errica_native_fair_v2_joint_agg_eval.sh")"

echo "Submitted native_fair_v2_joint agg→eval JOBID=${JOBID}  (waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID})"
echo "Monitor: squeue -j ${JOBID}; tail -f ${LOGDIR}/agg_eval_${JOBID}.log"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
