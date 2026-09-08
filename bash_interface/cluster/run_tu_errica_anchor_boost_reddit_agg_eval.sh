#!/usr/bin/env bash
# Re-aggregate anchor_boost (incl. REDDIT fill), then eval REDDIT only (tasks 31–60).
#
# PROTEINS eval (1–30) already finished (44938699). This only submits the
# REDDIT half of sigma_grid_eval_anchor_boost.
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

SELECTION_OUT="${REPO_ROOT}/configs/tu_errica/selections/sigma_anchor_boost_per_fold.json"

log_message "anchor_boost: re-aggregating select → ${SELECTION_OUT}"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_anchor_boost

if [ ! -f "${SELECTION_OUT}" ]; then
  log_message "ERROR: selection file missing after aggregate: ${SELECTION_OUT}"
  exit 1
fi
log_message "selection written: ${SELECTION_OUT}"

# Compact layout: proteins 1–30, reddit-b 31–60.
log_message "anchor_boost: submitting REDDIT eval only (array 31-60)"
TU_ERRICA_CAMPAIGN=sigma_grid_eval_anchor_boost \
  TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
  TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
  TU_ERRICA_ARRAY=31-60 \
  TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_anchor_boost_per_fold.json \
  bash bash_interface/cluster/submit_tu_errica_fair.sh

log_message "anchor_boost REDDIT agg→eval chain done (eval array 31-60 submitted)"
