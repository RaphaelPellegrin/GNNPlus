#!/usr/bin/env bash
# Aggregate specialist_tiny select winners, then submit Errica eval.
#
# Intended as the body of a SLURM job that depends on the select array
# (see submit_tu_errica_specialist_tiny_agg_eval.sh). Sources conda / W&B via
# common_env.sh (same as other cluster workers).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

SELECTION_OUT="${REPO_ROOT}/configs/tu_errica/selections/sigma_specialist_tiny_per_fold.json"

log_message "specialist_tiny: aggregating select → ${SELECTION_OUT}"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_specialist_tiny

if [ ! -f "${SELECTION_OUT}" ]; then
  log_message "ERROR: selection file missing after aggregate: ${SELECTION_OUT}"
  exit 1
fi
log_message "selection written: ${SELECTION_OUT}"

log_message "specialist_tiny: submitting sigma_grid_eval_specialist_tiny"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_specialist_tiny

log_message "specialist_tiny agg→eval chain done (eval array submitted)"
