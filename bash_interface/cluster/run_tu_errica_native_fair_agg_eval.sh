#!/usr/bin/env bash
# Aggregate native_fair (H200) select winners, then submit Errica eval.
#
# Intended as the body of a SLURM job that depends on select array 45235956
# (see submit_tu_errica_native_fair_agg_eval.sh).
#
# NOTE: Do not resolve SCRIPT_DIR via BASH_SOURCE — Slurm copies this file to
# /var/slurmd/.../slurm_script. Use SLURM_SUBMIT_DIR / GNNPLUS_PROJECT_ROOT.

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

SELECTION_OUT="${REPO_ROOT}/configs/tu_errica/selections/sigma_native_fair_per_fold.json"

log_message "native_fair: aggregating select → ${SELECTION_OUT}"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_native_fair

if [ ! -f "${SELECTION_OUT}" ]; then
  log_message "ERROR: selection file missing after aggregate: ${SELECTION_OUT}"
  exit 1
fi
log_message "selection written: ${SELECTION_OUT}"

log_message "native_fair: submitting sigma_grid_eval_native_fair (gpu_h200)"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_native_fair

log_message "native_fair agg→eval chain done (eval array submitted)"
