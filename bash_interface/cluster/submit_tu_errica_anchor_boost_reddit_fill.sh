#!/usr/bin/env bash
# Rerun missing REDDIT-BINARY tasks from anchor_boost select (44840486).
#
# W&B coverage (2026-09-08): 120/240 reddit finished, 120 failed. Failures are
# systematically hp_id 12–23 (second half of the 24-config grid = bs=64 block)
# for every fold. PROTEINS (tasks 1–240) are complete.
#
# SLURM 1-based task IDs for missing reddit HPs:
#   253-264,277-288,301-312,325-336,349-360,373-384,397-408,421-432,445-456,469-480
#
# Usage (cluster):
#   bash bash_interface/cluster/submit_tu_errica_anchor_boost_reddit_fill.sh
#
# After fill completes:
#   TU_ERRICA_DEPENDENCY_JOBID=<fill_jobid> \
#     bash bash_interface/cluster/submit_tu_errica_anchor_boost_reddit_agg_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${REPO_ROOT}/configs/tu_errica/sigma_grids_anchor_boost/manifest.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "[anchor_boost reddit fill] missing ${MANIFEST}"
  exit 1
fi

# Fixed missing set from W&B (bs=64 half of each fold). Override with TU_ERRICA_ARRAY if needed.
ARRAY_SPEC="${TU_ERRICA_ARRAY:-253-264,277-288,301-312,325-336,349-360,373-384,397-408,421-432,445-456,469-480}"
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

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_anchor_boost"
mkdir -p "${LOGDIR}"

echo "[anchor_boost reddit fill] array=${ARRAY_SPEC}%${PARALLEL}"
echo "[anchor_boost reddit fill] partition=${PARTITION} nice=${NICE} time=${TIME}"
echo "[anchor_boost reddit fill] logs → ${LOGDIR}"
echo "[anchor_boost reddit fill] fills missing bs=64 HPs (hp 12–23) on REDDIT only"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[anchor_boost reddit fill] DRY_RUN=1 — not submitting"
  exit 0
fi

JOBID="$(sbatch --parsable \
  --job-name=tu_errica_ab_reddit_fill \
  --array="${ARRAY_SPEC}%${PARALLEL}" \
  --partition="${PARTITION}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --nice="${NICE}" \
  --gpus=1 \
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_anchor_boost,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR \
  --output="${LOGDIR}/select_%A_%a.log" \
  "${SCRIPT_DIR}/run_tu_errica_fair.sh")"

echo "Submitted anchor_boost REDDIT fill JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "After OK, queue re-agg + REDDIT eval 31–60:"
echo "  TU_ERRICA_DEPENDENCY_JOBID=${JOBID} \\"
echo "    bash bash_interface/cluster/submit_tu_errica_anchor_boost_reddit_agg_eval.sh"
echo "Paste into CLUSTER_LAUNCHES.md / Paper_tu_errica_fair_comparison.md"
