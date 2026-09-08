#!/usr/bin/env bash
# Submit aggregate→eval chain for native_fair (H200 select 45235956).
#
# Agg job itself is lightweight (W&B → JSON → sbatch eval). Default partition
# is mweber_gpu so it starts ASAP; eval array still lands on gpu_h200 via
# sigma_grid_eval_native_fair.
#
# Usage (cluster, after select is running/finished):
#   TU_ERRICA_DEPENDENCY_JOBID=45235956 \
#     bash bash_interface/cluster/submit_tu_errica_native_fair_agg_eval.sh
#
# Env knobs:
#   TU_ERRICA_DEPENDENCY_JOBID   optional — omit to run immediately (re-agg)
#   TU_ERRICA_DEPENDENCY_TYPE    afterok (default) | afterany
#   TU_ERRICA_PARTITION / MEM / TIME / NICE / DRY_RUN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEPENDENCY_JOBID="${TU_ERRICA_DEPENDENCY_JOBID:-}"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_native_fair"
mkdir -p "${LOGDIR}"

if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "[native_fair agg_eval] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
else
  echo "[native_fair agg_eval] no dependency (run immediately)"
fi
echo "[native_fair agg_eval] partition=${PARTITION} time=${TIME} mem=${MEM}"
echo "[native_fair agg_eval] selection → configs/tu_errica/selections/sigma_native_fair_per_fold.json"
echo "[native_fair agg_eval] eval campaign → sigma_grid_eval_native_fair (gpu_h200, 90)"
echo "[native_fair agg_eval] logs → ${LOGDIR}/agg_eval_<JOBID>.log"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[native_fair agg_eval] DRY_RUN=1 — not submitting"
  exit 0
fi

SBATCH_ARGS=(
  --parsable
  --job-name=tu_errica_nf_agg_eval
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --nice="${NICE}"
  --gpus=1
  --cpus-per-task=4
  --export=ALL,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR,GNNPLUS_LIGHTWEIGHT_ENV=1
  --output="${LOGDIR}/agg_eval_%j.log"
)
if [ -n "${DEPENDENCY_JOBID}" ]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}")
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_tu_errica_native_fair_agg_eval.sh")"

echo "Submitted native_fair agg→eval JOBID=${JOBID}"
if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "  waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
fi
echo "Monitor: squeue -j ${JOBID}; tail -f ${LOGDIR}/agg_eval_${JOBID}.log"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
