#!/usr/bin/env bash
# Aggregate nci1_refine select winners, then submit Errica eval (30 tasks).
#
# Select (45149015) is already finished — submit with no dependency, or
# afterok if you re-ran select.
#
# NOTE: Do not resolve SCRIPT_DIR via BASH_SOURCE — Slurm copies this file.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"

# shellcheck source=common_env.sh
export GNNPLUS_LIGHTWEIGHT_ENV="${GNNPLUS_LIGHTWEIGHT_ENV:-1}"
source "${SCRIPT_DIR}/common_env.sh"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

SELECTION_OUT="${REPO_ROOT}/configs/tu_errica/selections/sigma_nci1_refine_per_fold.json"

log_message "nci1_refine: aggregating select → ${SELECTION_OUT}"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_nci1_refine

if [ ! -f "${SELECTION_OUT}" ]; then
  log_message "ERROR: selection file missing after aggregate: ${SELECTION_OUT}"
  exit 1
fi
log_message "selection written: ${SELECTION_OUT}"

log_message "nci1_refine: submitting sigma_grid_eval_nci1_refine"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_nci1_refine

log_message "nci1_refine agg→eval chain done (eval array submitted)"
