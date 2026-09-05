#!/usr/bin/env bash
# Collect FAILED array tasks from prior full64 select jobs and resubmit them.
#
# Usage (on cluster, after sourcing ~/.gnnplus_env):
#   bash bash_interface/cluster/submit_tu_errica_full64_rerun_failed.sh
#
# Optional env:
#   TU_ERRICA_FULL64_PARENT_JOBS   default: 44262912,44266493
#   TU_ERRICA_PARTITION           default: gpu_h200
#   TU_ERRICA_PARALLEL            default: 20
#   TU_ERRICA_MEM                 default: 128GB
#   TU_ERRICA_TIME                default: 72:00:00
#   TU_ERRICA_NICE                default: 0
#   TU_ERRICA_EXCLUDE             default: holygpu8a12204
#   TU_ERRICA_SACCT_START         optional sacct -S (FASRC max window ~7d;
#                                 default: omit -S and query by JobID only)
#   TU_ERRICA_DRY_RUN             if 1, only print array spec (no sbatch)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PARENT_JOBS="${TU_ERRICA_FULL64_PARENT_JOBS:-44262912,44266493}"
PARTITION="${TU_ERRICA_PARTITION:-gpu_h200}"
PARALLEL="${TU_ERRICA_PARALLEL:-20}"
MEM="${TU_ERRICA_MEM:-128GB}"
TIME="${TU_ERRICA_TIME:-72:00:00}"
NICE="${TU_ERRICA_NICE:-0}"
EXCLUDE="${TU_ERRICA_EXCLUDE:-holygpu8a12204}"
DRY_RUN="${TU_ERRICA_DRY_RUN:-0}"
SACCT_START="${TU_ERRICA_SACCT_START:-}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_full64_rerun"
mkdir -p "${LOGDIR}"

TMP_FAILED="$(mktemp)"
TMP_RAW="$(mktemp)"
trap 'rm -f "${TMP_FAILED}" "${TMP_RAW}"' EXIT

IFS=',' read -r -a JOB_ARR <<< "${PARENT_JOBS}"
: > "${TMP_RAW}"
for jid in "${JOB_ARR[@]}"; do
  jid="$(echo "${jid}" | tr -d '[:space:]')"
  [ -n "${jid}" ] || continue
  # Prefer -j without -S: FASRC rejects wide -S windows (~7d max) and can
  # return 0 rows. Optional TU_ERRICA_SACCT_START if you need a narrow window.
  # parsable2 avoids column truncation of compact JobIDs.
  SACCT_CMD=(sacct -j "${jid}" -X
    --state=FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED
    --parsable2 --format=JobID,State,ExitCode -n)
  if [ -n "${SACCT_START}" ]; then
    SACCT_CMD=(sacct -j "${jid}" -X -S "${SACCT_START}"
      --state=FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED
      --parsable2 --format=JobID,State,ExitCode -n)
  fi
  "${SACCT_CMD[@]}" >> "${TMP_RAW}" || true
done

N_RAW="$(wc -l < "${TMP_RAW}" | tr -d ' ')"
echo "[full64_rerun] sacct raw lines=${N_RAW} (start=${SACCT_START:-none})"

python3 - "${TMP_RAW}" "${TMP_FAILED}" <<'PY'
"""Parse sacct JobIDs (including compact array ranges) into task ID list."""
from __future__ import annotations

import re
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

task_re = re.compile(r"_(\d+)(?:\.\S+)?$")
range_re = re.compile(r"_\[(\d+)-(\d+)(?:%\d+)?\]")
single_bracket_re = re.compile(r"_\[(\d+)(?:%\d+)?\]")

ids: set[int] = set()
for line in raw_path.read_text().splitlines():
    if not line.strip():
        continue
    jobid = line.split("|", 1)[0].strip()
    m_range = range_re.search(jobid)
    if m_range:
        lo, hi = int(m_range.group(1)), int(m_range.group(2))
        ids.update(range(lo, hi + 1))
        continue
    m_one = single_bracket_re.search(jobid)
    if m_one:
        ids.add(int(m_one.group(1)))
        continue
    m_task = task_re.search(jobid)
    if m_task:
        ids.add(int(m_task.group(1)))

sorted_ids = sorted(ids)
out_path.write_text("\n".join(str(i) for i in sorted_ids) + ("\n" if sorted_ids else ""))
print(f"[full64_rerun] parsed unique task ids={len(sorted_ids)}")
if sorted_ids:
    print(f"[full64_rerun] task id range={sorted_ids[0]}..{sorted_ids[-1]}")
PY

N_FAILED="$(wc -l < "${TMP_FAILED}" | tr -d ' ')"
if [ "${N_FAILED}" -eq 0 ]; then
  echo "[full64_rerun] No FAILED task IDs found for jobs: ${PARENT_JOBS}"
  echo "[full64_rerun] Debug: head of sacct raw:"
  head -20 "${TMP_RAW}" || true
  echo "[full64_rerun] Tip: sacct -j 44262912 -X --state=FAILED --format=ExitCode -n | wc -l"
  exit 1
fi

ARRAY_SPEC="$(
  python3 - "${TMP_FAILED}" <<'PY'
from pathlib import Path
import sys

ids = [int(x) for x in Path(sys.argv[1]).read_text().split() if x.strip()]
ids = sorted(set(ids))
parts: list[str] = []
start = prev = ids[0]
for x in ids[1:]:
    if x == prev + 1:
        prev = x
        continue
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    start = prev = x
parts.append(str(start) if start == prev else f"{start}-{prev}")
print(",".join(parts))
PY
)"

# Guard: earlier bug only caught tasks 1,3-8 (~7). Real cliff is ~2k+.
if [ "${N_FAILED}" -lt 100 ]; then
  echo "[full64_rerun] WARNING: n_failed=${N_FAILED} looks too small (expected ~2000+)."
  echo "[full64_rerun] ARRAY_SPEC=${ARRAY_SPEC}"
  echo "[full64_rerun] Refusing to submit. Check sacct -S / parsable output, then rerun."
  exit 2
fi

echo "[full64_rerun] parents=${PARENT_JOBS}"
echo "[full64_rerun] n_failed=${N_FAILED}"
echo "[full64_rerun] array_spec(head)=$(echo "${ARRAY_SPEC}" | cut -c1-120)..."
echo "[full64_rerun] partition=${PARTITION} parallel=%${PARALLEL} exclude=${EXCLUDE}"
echo "[full64_rerun] logs → ${LOGDIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[full64_rerun] DRY_RUN=1 — not submitting"
  echo "ARRAY_SPEC=${ARRAY_SPEC}"
  exit 0
fi

SBATCH_ARGS=(
  --parsable
  --job-name=tu_errica_sigma_grid_select_full64
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --nice="${NICE}"
  --gpus=1
  --export=ALL,TU_ERRICA_CAMPAIGN=sigma_grid_select_full64,GNNPLUS_DATASET_DIR,GNNPLUS_OUT_DIR
  --output="${LOGDIR}/full64_%A_%a.log"
)

if [ -n "${EXCLUDE}" ]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE}")
fi

JOBID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_tu_errica_fair.sh")"
echo "Submitted sigma_grid_select_full64 RERUN JOBID=${JOBID}  tasks=${N_FAILED} failed cells"
echo "Paste into Paper_tu_errica_fair_comparison.md"
echo "Monitor: squeue -j ${JOBID}; sacct -j ${JOBID} -X --format=State -n | sort | uniq -c"
