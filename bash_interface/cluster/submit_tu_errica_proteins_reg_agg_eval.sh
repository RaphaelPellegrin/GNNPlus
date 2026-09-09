#!/usr/bin/env bash
# Submit UNION anchor_boost + proteins_reg aggregate→eval, dependent on select.
#
# Does NOT touch anchor_boost-only outputs
# (sigma_anchor_boost_per_fold.json / sigma_grid_eval_anchor_boost).
# Eval is PROTEINS-only (30 tasks) on mweber_gpu.
#
# Usage (cluster, after proteins_reg select is submitted):
#   TU_ERRICA_DEPENDENCY_JOBID=<proteins_reg_select_JOBID> \
#     bash bash_interface/cluster/submit_tu_errica_proteins_reg_agg_eval.sh
#
# Env knobs:
#   TU_ERRICA_DEPENDENCY_JOBID   required — proteins_reg select JOBID
#   TU_ERRICA_DEPENDENCY_TYPE    afterok (default) | afterany
#   TU_ERRICA_PARTITION / MEM / TIME / NICE / DRY_RUN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEPENDENCY_JOBID="${TU_ERRICA_DEPENDENCY_JOBID:-}"
if [ -z "${DEPENDENCY_JOBID}" ]; then
  echo "[proteins_reg agg_eval] set TU_ERRICA_DEPENDENCY_JOBID=<proteins_reg select JOBID>"
  echo "  e.g. TU_ERRICA_DEPENDENCY_JOBID=455XXXXX bash bash_interface/cluster/submit_tu_errica_proteins_reg_agg_eval.sh"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_proteins_reg"
mkdir -p "${LOGDIR}"

echo "[proteins_reg agg_eval] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
echo "[proteins_reg agg_eval] partition=${PARTITION} time=${TIME} mem=${MEM}"
echo "[proteins_reg agg_eval] selection → configs/tu_errica/selections/sigma_proteins_reg_joint_per_fold.json"
echo "[proteins_reg agg_eval] eval campaign → sigma_grid_eval_proteins_reg_joint (PROTEINS 30)"
echo "[proteins_reg agg_eval] (anchor_boost-only paths untouched)"
echo "[proteins_reg agg_eval] logs → ${LOGDIR}/agg_eval_<JOBID>.log"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[proteins_reg agg_eval] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_proteins_reg_agg \
  --dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --cpus-per-task=4 \
  --export=ALL,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR,GNNPLUS_LIGHTWEIGHT_ENV=1 \
  --output="${LOGDIR}/agg_eval_%j.log" \
  "${SCRIPT_DIR}/run_tu_errica_proteins_reg_agg_eval.sh")"

echo "Submitted proteins_reg_joint agg→eval JOBID=${JOBID}  (waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID})"
echo "Monitor: squeue -j ${JOBID}; tail -f ${LOGDIR}/agg_eval_${JOBID}.log"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
