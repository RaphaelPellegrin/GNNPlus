#!/usr/bin/env bash
# Submit Errica SiGMA native_fair_v2 HP-select (PROTEINS / NCI1 / REDDIT on mweber_gpu).
#
# Companion to native_fair (H200): UniGCN mixes + mid LR.
#   a1g2_{gin_sage,gin_unigcn,gcn_gin} + a0g2_{gin_sage,gcn_gin,gcn_unigcn}
#   × lr∈{1e-3,5e-3} × L∈{4,12}
#   bs=32, d_h=16, H=64 fixed → 24 configs × 3 × 10 = 720 select.
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_native_fair_v2_select.sh
#   TU_ERRICA_PARALLEL=10 bash bash_interface/cluster/submit_tu_errica_native_fair_v2_select.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_native_fair_v2/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[native_fair_v2] missing ${MANIFEST}"
  echo "[native_fair_v2] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode native_fair_v2"
  exit 1
fi

NUM_TASKS="$(python3 -c "import json; print(json.load(open('${MANIFEST}'))['num_tasks'])")"
ARRAY_SPEC="${TU_ERRICA_ARRAY:-1-${NUM_TASKS}}"
PARTITION="${TU_ERRICA_PARTITION:-mweber_gpu}"
PARALLEL="${TU_ERRICA_PARALLEL:-20}"
MEM="${TU_ERRICA_MEM:-128GB}"
TIME="${TU_ERRICA_TIME:-96:00:00}"
NICE="${TU_ERRICA_NICE:-0}"
DRY_RUN="${TU_ERRICA_DRY_RUN:-0}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_native_fair_v2"
mkdir -p "${LOGDIR}"

echo "[native_fair_v2] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[native_fair_v2] partition=${PARTITION} nice=${NICE} time=${TIME}"
echo "[native_fair_v2] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[native_fair_v2] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_native_fair_v2 \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_native_fair_v2,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_native_fair_v2 JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
