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

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_errica_full64_rerun"
mkdir -p "${LOGDIR}"

TMP_FAILED="$(mktemp)"
trap 'rm -f "${TMP_FAILED}"' EXIT

# Compact array jobs often show one PENDING line; expand via sacct task rows.
# JobID forms: 44262912_1927 or 44262912_[1927-1930]
IFS=',' read -r -a JOB_ARR <<< "${PARENT_JOBS}"
for jid in "${JOB_ARR[@]}"; do
  jid="$(echo "${jid}" | tr -d '[:space:]')"
  [ -n "${jid}" ] || continue
  sacct -j "${jid}" -X --state=FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED \
    --format=JobID -n 2>/dev/null \
    | awk '{print $1}' \
    | while read -r tok; do
        case "${tok}" in
          *_\[*\])
            # e.g. 44262912_[1927-1930%20] — skip compact; rely on expanded rows
            ;;
          *_*)
            tid="${tok##*_}"
            tid="${tid%%.*}"
            if [[ "${tid}" =~ ^[0-9]+$ ]]; then
              echo "${tid}"
            fi
            ;;
        esac
      done
done | sort -n | uniq > "${TMP_FAILED}"

N_FAILED="$(wc -l < "${TMP_FAILED}" | tr -d ' ')"
if [ "${N_FAILED}" -eq 0 ]; then
  echo "[full64_rerun] No FAILED task IDs found for jobs: ${PARENT_JOBS}"
  echo "[full64_rerun] Tip: sacct -j 44262912 -X --state=FAILED --format=JobID%20,State,ExitCode,NodeList | head"
  exit 1
fi

# Build SLURM array compact ranges: 1,3-5,10
ARRAY_SPEC="$(
  python3 - "${TMP_FAILED}" <<'PY'
from pathlib import Path
import sys

ids = [int(x) for x in Path(sys.argv[1]).read_text().split() if x.strip()]
ids = sorted(set(ids))
if not ids:
    raise SystemExit("empty failed list")

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
