#!/usr/bin/env bash
# Submit Errica SiGMA a1g2 micro HP-select (PROTEINS + REDDIT-BINARY).
#
# Arch: a1g2 (1 attn + GCN,GIN; no SAGE/GAT).
# Grid: L12/H64/d_h16 fixed; bs∈{16,64} × lr∈{1e-3,1e-2} → 4 configs
#       × 2 datasets × 10 folds = 80 select tasks.
# Does NOT cancel other jobs. Respects grid batch_size (no forced bs=16).
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_a1g2_micro_select.sh
#
# Optional env:
#   TU_ERRICA_PARTITION   default: mweber_gpu
#   TU_ERRICA_PARALLEL    default: 20
#   TU_ERRICA_MEM         default: 128GB
#   TU_ERRICA_TIME        default: 96:00:00
#   TU_ERRICA_NICE        default: 0
#   TU_ERRICA_DRY_RUN     if 1, print plan only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_a1g2_micro/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[a1g2_micro] missing ${MANIFEST}"
  echo "[a1g2_micro] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode a1g2_micro"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_a1g2_micro"
mkdir -p "${LOGDIR}"

echo "[a1g2_micro] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[a1g2_micro] partition=${PARTITION} nice=${NICE}"
echo "[a1g2_micro] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[a1g2_micro] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_a1g2_micro \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_a1g2_micro,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_a1g2_micro JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
