#!/usr/bin/env bash
# Submit re-agg + REDDIT-only eval after anchor_boost REDDIT fill.
#
# Usage (cluster):
#   TU_ERRICA_DEPENDENCY_JOBID=<reddit_fill_jobid> \
#     bash bash_interface/cluster/submit_tu_errica_anchor_boost_reddit_agg_eval.sh

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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_anchor_boost"
mkdir -p "${LOGDIR}"

if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "[anchor_boost reddit agg_eval] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
else
  echo "[anchor_boost reddit agg_eval] no dependency (run immediately)"
fi
echo "[anchor_boost reddit agg_eval] re-agg → sigma_anchor_boost_per_fold.json"
echo "[anchor_boost reddit agg_eval] eval array 31-60 (REDDIT only; PROTEINS already done)"
echo "[anchor_boost reddit agg_eval] logs → ${LOGDIR}/reddit_agg_eval_<JOBID>.log"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[anchor_boost reddit agg_eval] DRY_RUN=1 — not submitting"
  exit 0
fi

SBATCH_ARGS=(
  --parsable
  --job-name=tu_errica_ab_reddit_agg
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --nice="${NICE}"
  --gpus=1
  --cpus-per-task=4
  --export=ALL,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR,GNNPLUS_LIGHTWEIGHT_ENV=1
  --output="${LOGDIR}/reddit_agg_eval_%j.log"
)
if [ -n "${DEPENDENCY_JOBID}" ]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}")
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_tu_errica_anchor_boost_reddit_agg_eval.sh")"

echo "Submitted anchor_boost REDDIT agg→eval JOBID=${JOBID}"
if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "  waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
fi
echo "Monitor: squeue -j ${JOBID}; tail -f ${LOGDIR}/reddit_agg_eval_${JOBID}.log"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
