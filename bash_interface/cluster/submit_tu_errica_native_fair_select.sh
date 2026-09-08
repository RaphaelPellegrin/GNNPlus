#!/usr/bin/env bash
# Submit Errica SiGMA native_fair HP-select (PROTEINS / NCI1 / REDDIT on gpu_h200).
#
# Compact fair (NOT GIN-isomorphic full64):
#   a1g2_{gin_sage,gcn_gin} + a0g2_{gin_sage,gcn_gin}
#   × lr∈{1e-3,1e-2} × L∈{4,12}
#   bs=32, d_h=16, H=64 fixed → 16 configs × 3 × 10 = 480 select.
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_native_fair_select.sh
#   TU_ERRICA_PARALLEL=10 bash bash_interface/cluster/submit_tu_errica_native_fair_select.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_native_fair/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[native_fair] missing ${MANIFEST}"
  echo "[native_fair] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode native_fair"
  exit 1
fi

NUM_TASKS="$(python3 -c "import json; print(json.load(open('${MANIFEST}'))['num_tasks'])")"
ARRAY_SPEC="${TU_ERRICA_ARRAY:-1-${NUM_TASKS}}"
# gpu_h200 MaxTime is typically 3 days.
PARTITION="${TU_ERRICA_PARTITION:-gpu_h200}"
PARALLEL="${TU_ERRICA_PARALLEL:-40}"
MEM="${TU_ERRICA_MEM:-128GB}"
if [ "${PARTITION}" = "gpu_h200" ]; then
  TIME="${TU_ERRICA_TIME:-72:00:00}"
else
  TIME="${TU_ERRICA_TIME:-96:00:00}"
fi
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

echo "[native_fair] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[native_fair] partition=${PARTITION} nice=${NICE} time=${TIME}"
echo "[native_fair] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[native_fair] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_native_fair \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_native_fair,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_native_fair JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
