#!/usr/bin/env bash
# Submit UNION L12 + reddit_reg aggregate→eval, dependent on reddit_reg select.
#
# Does NOT touch L12-only outputs
# (sigma_native_fair_v2_per_fold.json / sigma_grid_eval_native_fair_v2).
# Eval is REDDIT-only (30 tasks) on gpu_h200.
#
# Usage (cluster, after reddit_reg select is submitted):
#   TU_ERRICA_DEPENDENCY_JOBID=<reddit_reg_select_JOBID> \
#     bash bash_interface/cluster/submit_tu_errica_native_fair_v2_reddit_reg_agg_eval.sh
#
# Env knobs:
#   TU_ERRICA_DEPENDENCY_JOBID   required — reddit_reg select JOBID
#   TU_ERRICA_DEPENDENCY_TYPE    afterok (default) | afterany
#   TU_ERRICA_PARTITION / MEM / TIME / NICE / DRY_RUN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEPENDENCY_JOBID="${TU_ERRICA_DEPENDENCY_JOBID:-}"
if [ -z "${DEPENDENCY_JOBID}" ]; then
  echo "[native_fair_v2_reddit_reg agg_eval] set TU_ERRICA_DEPENDENCY_JOBID=<reddit_reg select JOBID>"
  echo "  e.g. TU_ERRICA_DEPENDENCY_JOBID=455XXXXX bash bash_interface/cluster/submit_tu_errica_native_fair_v2_reddit_reg_agg_eval.sh"
  exit 1
fi

DEPENDENCY_TYPE="${TU_ERRICA_DEPENDENCY_TYPE:-afterok}"
PARTITION="${TU_ERRICA_PARTITION:-gpu_h200}"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_native_fair_v2_reddit_reg"
mkdir -p "${LOGDIR}"

echo "[native_fair_v2_reddit_reg agg_eval] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
echo "[native_fair_v2_reddit_reg agg_eval] partition=${PARTITION} time=${TIME} mem=${MEM}"
echo "[native_fair_v2_reddit_reg agg_eval] selection → configs/tu_errica/selections/sigma_native_fair_v2_reddit_reg_joint_per_fold.json"
echo "[native_fair_v2_reddit_reg agg_eval] eval campaign → sigma_grid_eval_native_fair_v2_reddit_reg_joint (REDDIT 30)"
echo "[native_fair_v2_reddit_reg agg_eval] (L12-only paths untouched)"
echo "[native_fair_v2_reddit_reg agg_eval] logs → ${LOGDIR}/agg_eval_<JOBID>.log"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[native_fair_v2_reddit_reg agg_eval] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_nfv2_reddit_reg_agg \
  --dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --cpus-per-task=4 \
  --export=ALL,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR,GNNPLUS_LIGHTWEIGHT_ENV=1 \
  --output="${LOGDIR}/agg_eval_%j.log" \
  "${SCRIPT_DIR}/run_tu_errica_native_fair_v2_reddit_reg_agg_eval.sh")"

echo "Submitted native_fair_v2_reddit_reg_joint agg→eval JOBID=${JOBID}  (waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID})"
echo "Monitor: squeue -j ${JOBID}; tail -f ${LOGDIR}/agg_eval_${JOBID}.log"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
