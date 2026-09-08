#!/usr/bin/env bash
# Aggregate native_fair_v2 select winners, then submit Errica eval.
#
# Intended as the body of a SLURM job that depends on the select array
# (see submit_tu_errica_native_fair_v2_agg_eval.sh). Sources conda / W&B via
# common_env.sh (same as other cluster workers).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=common_env.sh
# Agg/eval orchestration only needs wandb + python (no torch train import).
export GNNPLUS_LIGHTWEIGHT_ENV="${GNNPLUS_LIGHTWEIGHT_ENV:-1}"
source "${SCRIPT_DIR}/common_env.sh"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

SELECTION_OUT="${REPO_ROOT}/configs/tu_errica/selections/sigma_native_fair_v2_per_fold.json"

log_message "native_fair_v2: aggregating select → ${SELECTION_OUT}"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_native_fair_v2

if [ ! -f "${SELECTION_OUT}" ]; then
  log_message "ERROR: selection file missing after aggregate: ${SELECTION_OUT}"
  exit 1
fi
log_message "selection written: ${SELECTION_OUT}"

log_message "native_fair_v2: submitting sigma_grid_eval_native_fair_v2"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_native_fair_v2

log_message "native_fair_v2 agg→eval chain done (eval array submitted)"
