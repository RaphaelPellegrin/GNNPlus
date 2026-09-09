#!/usr/bin/env bash
# Submit Errica SiGMA proteins_reg HP-select (PROTEINS only).
#
# Arch: a2g4 (sigma-hetero-errica-base.yaml). Modal locks: bs=16, lr=1e-3,
# dropout=0.5, pool=add. Search: L∈{8,12} × d_h∈{8,16} → 4 × 10 = 40 select.
# After select: UNION with anchor_boost via joint agg→eval (PROTEINS-only).
#
# Typical:
#   bash bash_interface/cluster/submit_tu_errica_proteins_reg_select.sh
#
# Env knobs:
#   TU_ERRICA_PARALLEL / PARTITION / MEM / TIME / NICE / DRY_RUN / ARRAY
#   TU_ERRICA_DEPENDENCY_JOBID / TU_ERRICA_DEPENDENCY_TYPE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_proteins_reg/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[proteins_reg] missing ${MANIFEST}"
  echo "[proteins_reg] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode proteins_reg"
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
DEPENDENCY_JOBID="${TU_ERRICA_DEPENDENCY_JOBID:-}"
DEPENDENCY_TYPE="${TU_ERRICA_DEPENDENCY_TYPE:-afterok}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_proteins_reg"
mkdir -p "${LOGDIR}"

echo "[proteins_reg] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[proteins_reg] partition=${PARTITION} nice=${NICE} time=${TIME}"
if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "[proteins_reg] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
fi
echo "[proteins_reg] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[proteins_reg] DRY_RUN=1 — not submitting"
  exit 0
fi

SBATCH_ARGS=(
  --parsable
  --job-name=tu_errica_sigma_proteins_reg
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --nice="${NICE}"
  --gpus=1
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_proteins_reg,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR
  --output="${LOGDIR}/select_%A_%a.log"
)

if [ -n "${DEPENDENCY_JOBID}" ]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}")
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_proteins_reg JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "  waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
fi
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
echo "After select: TU_ERRICA_DEPENDENCY_JOBID=${JOBID} bash bash_interface/cluster/submit_tu_errica_proteins_reg_agg_eval.sh"
