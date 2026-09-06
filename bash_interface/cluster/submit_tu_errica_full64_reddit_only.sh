#!/usr/bin/env bash
# Resubmit full64 SiGMA HP-select for REDDIT-BINARY only (manifest tasks 3201–3840).
#
# Does NOT cancel any existing jobs (including full64 FAILED rerun 44509970).
# Same W&B campaign / task indices as the parent, so aggregate_sigma_full64 works.
# Overlap with 44509970 is OK: duplicate runs are fine; aggregate keeps best val.
#
# Usage (cluster, after source ~/.gnnplus_env):
#   bash bash_interface/cluster/submit_tu_errica_full64_reddit_only.sh
#
# Optional env:
#   TU_ERRICA_PARTITION   default: gpu_h200
#   TU_ERRICA_PARALLEL    default: 20
#   TU_ERRICA_MEM         default: 128GB
#   TU_ERRICA_TIME        default: 72:00:00
#   TU_ERRICA_NICE        default: 0
#   TU_ERRICA_EXCLUDE     default: holygpu8a12204 (empty string to disable)
#   TU_ERRICA_DRY_RUN     if 1, print plan only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARRAY_SPEC="${TU_ERRICA_ARRAY:-3201-3840}"
PARTITION="${TU_ERRICA_PARTITION:-gpu_h200}"
PARALLEL="${TU_ERRICA_PARALLEL:-20}"
MEM="${TU_ERRICA_MEM:-128GB}"
TIME="${TU_ERRICA_TIME:-72:00:00}"
NICE="${TU_ERRICA_NICE:-0}"
EXCLUDE="${TU_ERRICA_EXCLUDE:-holygpu8a12204}"
DRY_RUN="${TU_ERRICA_DRY_RUN:-0}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_full64_reddit"
mkdir -p "${LOGDIR}"

# Sanity: confirm REDDIT band in manifest (1-based SLURM indices).
python3 - <<'PY'
"""Verify full64 manifest REDDIT task range matches 3201–3840."""
from __future__ import annotations

import json
from pathlib import Path

manifest = json.loads(
    Path("configs/tu_errica/sigma_grids_full64/manifest.json").read_text()
)
tasks = manifest["tasks"]
reddit = [
    (i + 1, t)
    for i, t in enumerate(tasks)
    if str(t.get("dataset", "")).upper() == "REDDIT-BINARY"
]
assert reddit, "no REDDIT-BINARY tasks in full64 manifest"
lo, hi = reddit[0][0], reddit[-1][0]
print(f"[full64_reddit] manifest reddit tasks={len(reddit)} range={lo}-{hi}")
if (lo, hi) != (3201, 3840):
    raise SystemExit(
        f"unexpected REDDIT range {lo}-{hi}; update TU_ERRICA_ARRAY / script"
    )
PY

echo "[full64_reddit] array=${ARRAY_SPEC}%${PARALLEL}"
echo "[full64_reddit] partition=${PARTITION} nice=${NICE} exclude=${EXCLUDE:-none}"
echo "[full64_reddit] logs → ${LOGDIR}"
echo "[full64_reddit] NO scancel — leaves 44509970 and other full64 jobs alone"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[full64_reddit] DRY_RUN=1 — not submitting"
  exit 0
fi

SBATCH_ARGS=(
  --parsable
  --job-name=tu_errica_sigma_full64_reddit
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --nice="${NICE}"
  --gpus=1
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_full64,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR
  --output="${LOGDIR}/reddit_%A_%a.log"
)

if [ -n "${EXCLUDE}" ]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE}")
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_tu_errica_fair.sh")"
echo "Submitted sigma_grid_select_full64 REDDIT-only JOBID=${JOBID}  tasks=${ARRAY_SPEC}"
echo "Paste into Paper_tu_errica_fair_comparison.md / CLUSTER_LAUNCHES.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
echo "Log: ${LOGDIR}/reddit_${JOBID}_<task>.log"
