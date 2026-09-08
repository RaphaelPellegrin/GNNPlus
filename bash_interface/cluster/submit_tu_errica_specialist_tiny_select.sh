#!/usr/bin/env bash
# Submit Errica SiGMA specialist_tiny HP-select (PROTEINS / NCI1 / REDDIT).
#
# Single-MP specialists matching classical winners:
#   PROTEINS / REDDIT → GCN (a0g1_gcn, a1g1_gcn)
#   NCI1             → SAGE (a0g1_sage, a1g1_sage)
# Train: lr∈{1e-3, 1e-4}; L=12, d_h=16, bs=32 fixed → 4 × 3 × 10 = 120 select.
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_specialist_tiny_select.sh
#
# Optional env: TU_ERRICA_PARTITION / PARALLEL / MEM / TIME / NICE / DRY_RUN / ARRAY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_specialist_tiny/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[specialist_tiny] missing ${MANIFEST}"
  echo "[specialist_tiny] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode specialist_tiny"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_specialist_tiny"
mkdir -p "${LOGDIR}"

echo "[specialist_tiny] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[specialist_tiny] partition=${PARTITION} nice=${NICE} time=${TIME}"
echo "[specialist_tiny] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[specialist_tiny] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_specialist_tiny \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_specialist_tiny,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_specialist_tiny JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
