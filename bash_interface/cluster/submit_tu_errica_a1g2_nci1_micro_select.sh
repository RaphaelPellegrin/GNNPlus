#!/usr/bin/env bash
# Submit Errica SiGMA a1g2 NCI1 micro HP-select.
#
# Arch: a1g2 (1 attn + GIN,SAGE).
# Grid: L12/H64/d_h16 fixed; bs∈{32,128} × lr∈{1e-3,1e-2} → 4 configs
#       × 1 dataset × 10 folds = 40 select tasks.
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_a1g2_nci1_micro_select.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_a1g2_nci1_micro/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[a1g2_nci1_micro] missing ${MANIFEST}"
  echo "[a1g2_nci1_micro] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode a1g2_nci1_micro"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_a1g2_nci1_micro"
mkdir -p "${LOGDIR}"

echo "[a1g2_nci1_micro] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[a1g2_nci1_micro] partition=${PARTITION} nice=${NICE}"
echo "[a1g2_nci1_micro] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[a1g2_nci1_micro] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_a1g2_nci1 \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_a1g2_nci1_micro,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_a1g2_nci1_micro JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
