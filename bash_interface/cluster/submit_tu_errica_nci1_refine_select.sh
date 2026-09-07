#!/usr/bin/env bash
# Submit Errica SiGMA nci1_refine HP-select (NCI1 only).
#
# Arch: a2g4 (sigma-hetero-errica-base.yaml) — best SiGMA family on NCI1 so far
#       (fixed8 eval 80.7 vs a1g2 80.4).
# Grid: deep fixed8 center (bs=32, lr=1e-3, L=12, d_h=16) ×
#       dropout=0.5 × pool∈{add,mean} → 2 configs × 10 folds = 20 select.
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_nci1_refine_select.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_nci1_refine/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[nci1_refine] missing ${MANIFEST}"
  echo "[nci1_refine] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode nci1_refine"
  exit 1
fi

NUM_TASKS="$(python3 -c "import json; print(json.load(open('${MANIFEST}'))['num_tasks'])")"
ARRAY_SPEC="${TU_ERRICA_ARRAY:-1-${NUM_TASKS}}"
PARTITION="${TU_ERRICA_PARTITION:-mweber_gpu}"
PARALLEL="${TU_ERRICA_PARALLEL:-40}"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_nci1_refine"
mkdir -p "${LOGDIR}"

echo "[nci1_refine] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[nci1_refine] partition=${PARTITION} nice=${NICE}"
echo "[nci1_refine] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[nci1_refine] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_nci1_refine \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_nci1_refine,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_nci1_refine JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
