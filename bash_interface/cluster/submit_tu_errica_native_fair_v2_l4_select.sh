#!/usr/bin/env bash
# Submit Errica SiGMA native_fair_v2_l4 HP-select (L=4 add-on for v2).
#
# Same UniGCN arches / lr=1e-3 / d_h=32 as native_fair_v2, but layers_mp=4 only.
# Does not redo the L=12 select. Count: 6 × 3 × 10 = 180 select.
#
# Typical: queue after native_fair_v2 agg→eval (JOBID 45265929) so L=12 winners
# land first and GPU slots free before the shallow fill:
#   TU_ERRICA_DEPENDENCY_JOBID=45265929 \
#     bash bash_interface/cluster/submit_tu_errica_native_fair_v2_l4_select.sh
#
# Env knobs:
#   TU_ERRICA_DEPENDENCY_JOBID   optional — wait on this job (e.g. 45265929)
#   TU_ERRICA_DEPENDENCY_TYPE    afterok (default) | afterany
#   TU_ERRICA_PARALLEL / PARTITION / MEM / TIME / NICE / DRY_RUN / ARRAY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_native_fair_v2_l4/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[native_fair_v2_l4] missing ${MANIFEST}"
  echo "[native_fair_v2_l4] run: python scripts/tu_errica/generate_sigma_errica_grids.py --mode native_fair_v2_l4"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_native_fair_v2_l4"
mkdir -p "${LOGDIR}"

echo "[native_fair_v2_l4] tasks=${NUM_TASKS} array=${ARRAY_SPEC}%${PARALLEL}"
echo "[native_fair_v2_l4] partition=${PARTITION} nice=${NICE} time=${TIME}"
if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "[native_fair_v2_l4] dependency=${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
fi
echo "[native_fair_v2_l4] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[native_fair_v2_l4] DRY_RUN=1 — not submitting"
  exit 0
fi

SBATCH_ARGS=(
  --parsable
  --job-name=tu_errica_sigma_native_fair_v2_l4
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --nice="${NICE}"
  --gpus=1
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_native_fair_v2_l4,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR
  --output="${LOGDIR}/select_%A_%a.log"
)

if [ -n "${DEPENDENCY_JOBID}" ]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}")
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted sigma_grid_select_native_fair_v2_l4 JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
if [ -n "${DEPENDENCY_JOBID}" ]; then
  echo "  waits on ${DEPENDENCY_TYPE}:${DEPENDENCY_JOBID}"
fi
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
echo "After L4 finishes: merge L12+L4 W&B campaigns for fold winners, then re-eval."
