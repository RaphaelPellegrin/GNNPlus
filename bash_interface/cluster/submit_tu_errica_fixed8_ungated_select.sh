#!/usr/bin/env bash
# Submit Errica SiGMA UNGATED fixed8 HP-select (all 7 datasets).
#
# Same 8-config SIGMA_GRID as gated fixed8, but gnn.hybrid.gate=none
# (configs/tu_errica/sigma-hetero-ungated-errica-base.yaml).
# → 560 select tasks. Does NOT cancel other jobs (full64 / anchor_boost).
#
# Usage (cluster, after source ~/.gnnplus_env + git pull):
#   bash bash_interface/cluster/submit_tu_errica_fixed8_ungated_select.sh
#
# Optional env: TU_ERRICA_PARTITION / PARALLEL / MEM / TIME / NICE / DRY_RUN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[fixed8_ungated] missing ${MANIFEST}"
  exit 1
fi
if [ ! -f "${REPO_ROOT}/configs/tu_errica/sigma-hetero-ungated-errica-base.yaml" ]; then
  echo "[fixed8_ungated] missing ungated base yaml"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_fixed8_ungated"
mkdir -p "${LOGDIR}"

echo "[fixed8_ungated] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[fixed8_ungated] partition=${PARTITION} nice=${NICE} gate=none"
echo "[fixed8_ungated] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[fixed8_ungated] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_sigma_fixed8_ung \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_fixed8_ungated,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_fixed8_ungated JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
