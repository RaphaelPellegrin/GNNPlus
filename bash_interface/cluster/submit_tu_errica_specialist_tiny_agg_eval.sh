#!/usr/bin/env bash
# Submit aggregate→eval chain for specialist_tiny, dependent on the select array.
#
# Sources conda/W&B inside run_tu_errica_specialist_tiny_agg_eval.sh via common_env.sh.
#
# Usage (cluster, after select is submitted):
#   TU_ERRICA_DEPENDENCY_JOBID=45303391 \
#     bash bash_interface/cluster/submit_tu_errica_specialist_tiny_agg_eval.sh
#
# Env knobs:
#   TU_ERRICA_DEPENDENCY_JOBID   required — select array JOBID (e.g. 45303391)
#   TU_ERRICA_DEPENDENCY_TYPE    afterok (default) | afterany
#   TU_ERRICA_PARTITION          default mweber_gpu
#   TU_ERRICA_MEM / TIME / NICE / DRY_RUN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEPENDENCY_JOBID="${TU_ERRICA_DEPENDENCY_JOBID:-}"
if [ -z "${DEPENDENCY_JOBID}" ]; then
  echo "[specialist_tiny agg_eval] set TU_ERRICA_DEPENDENCY_JOBID=<select JOBID>"
  echo "  e.g. TU_ERRICA_DEPENDENCY_JOBID=45303391 bash bash_interface/cluster/submit_tu_errica_specialist_tiny_agg_eval.sh"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_specialist_tiny"
mkdir -p "${LOGDIR}"

echo "[specialist_tiny agg_eval] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
echo "[specialist_tiny agg_eval] partition=${PARTITION} time=${TIME} mem=${MEM}"
echo "[specialist_tiny agg_eval] logs → ${LOGDIR}/agg_eval_<JOBID>.log"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[specialist_tiny agg_eval] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_spec_tiny_agg \
  --dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --cpus-per-task=4 \
  --export=ALL,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/agg_eval_%j.log" \
  "${SCRIPT_DIR}/run_tu_errica_specialist_tiny_agg_eval.sh")"

echo "Submitted specialist_tiny agg→eval JOBID=${JOBID}  (waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID})"
echo "Monitor: squeue -j ${JOBID}; tail -f ${LOGDIR}/agg_eval_${JOBID}.log"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
