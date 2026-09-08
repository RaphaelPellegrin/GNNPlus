#!/usr/bin/env bash
# Submit Errica SiGMA tiny_pnr HP-select (PROTEINS / NCI1 / REDDIT only).
#
# Ultra-tiny *sensible* SiGMA grid (contrast full64 lr=0.01 + L=4):
#   a2g4 gated · lr=1e-3 · L=12 · H=64 · drop=0.5 · pool=add
#   search only bs∈{16,32} × d_h∈{8,16} → 4 configs × 3 × 10 = 120 select.
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_tiny_pnr_select.sh
#   TU_ERRICA_PARALLEL=10 bash bash_interface/cluster/submit_tu_errica_tiny_pnr_select.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_tiny_pnr/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[tiny_pnr] missing ${MANIFEST}"
  echo "[tiny_pnr] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode tiny_pnr"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_tiny_pnr"
mkdir -p "${LOGDIR}"

echo "[tiny_pnr] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[tiny_pnr] partition=${PARTITION} nice=${NICE}"
echo "[tiny_pnr] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[tiny_pnr] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_tiny_pnr \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_tiny_pnr,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_tiny_pnr JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
