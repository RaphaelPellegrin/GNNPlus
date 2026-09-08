#!/usr/bin/env bash
# Merge native_fair_v2 (L=12) + native_fair_v2_l4 select winners, then eval.
#
# Writes a SEPARATE selection JSON and uses a SEPARATE eval campaign so the
# L12-only chain (45265929 → sigma_native_fair_v2_per_fold.json /
# sigma_grid_eval_native_fair_v2) is never overwritten.
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

# Distinct from L12-only: configs/tu_errica/selections/sigma_native_fair_v2_per_fold.json
SELECTION_OUT="${REPO_ROOT}/configs/tu_errica/selections/sigma_native_fair_v2_joint_per_fold.json"

log_message "native_fair_v2_joint: merging L12+L4 selects → ${SELECTION_OUT}"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_native_fair_v2_joint

if [ ! -f "${SELECTION_OUT}" ]; then
  log_message "ERROR: selection file missing after aggregate: ${SELECTION_OUT}"
  exit 1
fi
log_message "selection written: ${SELECTION_OUT}"

log_message "native_fair_v2_joint: submitting sigma_grid_eval_native_fair_v2_joint"
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_native_fair_v2_joint

log_message "native_fair_v2_joint agg→eval chain done (eval array submitted)"
