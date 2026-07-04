#!/usr/bin/env bash
# Launch peptides-func + peptides-struct gated MP sweeps with a shared GPU cap.
#
# Default: 8 GPUs total → 4 parallel agents per sweep (4 + 4).
#
# Usage (cluster login, after git pull):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#
#   # Create sweeps + launch (first time):
#   bash bash_interface/sweeps/launch_peptides_gated_mp_sweeps.sh --create
#
#   # Launch only (sweep ids from env or bash_interface/sweeps/sweeps.log):
#   FUNC_SWEEP_ID=weber-geoml-harvard-university/GNNPlus/abc123 \
#   STRUCT_SWEEP_ID=weber-geoml-harvard-university/GNNPlus/def456 \
#     bash bash_interface/sweeps/launch_peptides_gated_mp_sweeps.sh
#
# Env overrides:
#   TOTAL_GPU_PARALLEL=8   max GPUs across both sweeps (default 8)
#   SWEEP_ARRAY_TASKS=16    agents per sweep array
#   RUNS_PER_AGENT=3
#   SWEEP_SLURM_TIME=240:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
SWEEPS_DIR="${REPO_ROOT}/bash_interface/sweeps"
LOG_FILE="${SWEEPS_DIR}/sweeps.log"

TOTAL_GPU_PARALLEL="${TOTAL_GPU_PARALLEL:-8}"
SWEEP_ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-16}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-3}"
SWEEP_SLURM_TIME="${SWEEP_SLURM_TIME:-240:00:00}"

FUNC_YAML="${SWEEPS_DIR}/peptides_func_hybrid_gated_mp_sweep.yaml"
STRUCT_YAML="${SWEEPS_DIR}/peptides_struct_hybrid_gated_mp_sweep.yaml"

if [ $((TOTAL_GPU_PARALLEL % 2)) -ne 0 ]; then
    echo "TOTAL_GPU_PARALLEL must be even (split evenly across 2 sweeps); got ${TOTAL_GPU_PARALLEL}" >&2
    exit 2
fi
PARALLEL_EACH=$((TOTAL_GPU_PARALLEL / 2))

lookup_sweep_id() {
    local yaml_path="$1"
    if [ ! -f "${LOG_FILE}" ]; then
        return 1
    fi
    grep -F "${yaml_path}" "${LOG_FILE}" | tail -n 1 | awk '{print $3}'
}

CREATE=false
for arg in "$@"; do
    if [ "${arg}" = "--create" ]; then
        CREATE=true
    fi
done

if [ "${CREATE}" = true ]; then
    echo "=== Creating peptides-func sweep ==="
    bash "${SWEEPS_DIR}/create_sweep.sh" "${FUNC_YAML}"
    echo ""
    echo "=== Creating peptides-struct sweep ==="
    bash "${SWEEPS_DIR}/create_sweep.sh" "${STRUCT_YAML}"
    echo ""
fi

FUNC_SWEEP_ID="${FUNC_SWEEP_ID:-$(lookup_sweep_id "${FUNC_YAML}" || true)}"
STRUCT_SWEEP_ID="${STRUCT_SWEEP_ID:-$(lookup_sweep_id "${STRUCT_YAML}" || true)}"

if [ -z "${FUNC_SWEEP_ID}" ] || [ -z "${STRUCT_SWEEP_ID}" ]; then
    echo "Set FUNC_SWEEP_ID and STRUCT_SWEEP_ID, or run with --create first." >&2
    echo "  FUNC_SWEEP_ID=${FUNC_SWEEP_ID:-<missing>}" >&2
    echo "  STRUCT_SWEEP_ID=${STRUCT_SWEEP_ID:-<missing>}" >&2
    exit 2
fi

echo "=== Launching sweep agents (max ${TOTAL_GPU_PARALLEL} GPUs = ${PARALLEL_EACH} per sweep) ==="
echo "  func:   ${FUNC_SWEEP_ID}"
echo "  struct: ${STRUCT_SWEEP_ID}"
echo "  tasks=${SWEEP_ARRAY_TASKS}  parallel_each=${PARALLEL_EACH}  runs/agent=${RUNS_PER_AGENT}  time=${SWEEP_SLURM_TIME}"
echo ""

SWEEP_SLURM_TIME="${SWEEP_SLURM_TIME}" \
SWEEP_ARRAY_TASKS="${SWEEP_ARRAY_TASKS}" \
SWEEP_ARRAY_PARALLEL="${PARALLEL_EACH}" \
RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
    bash "${SWEEPS_DIR}/relaunch_sweep_agents.sh" \
        peptides_func "${FUNC_SWEEP_ID}"

SWEEP_SLURM_TIME="${SWEEP_SLURM_TIME}" \
SWEEP_ARRAY_TASKS="${SWEEP_ARRAY_TASKS}" \
SWEEP_ARRAY_PARALLEL="${PARALLEL_EACH}" \
RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
    bash "${SWEEPS_DIR}/relaunch_sweep_agents.sh" \
        peptides_struct "${STRUCT_SWEEP_ID}"

echo ""
echo "Monitor: squeue -u \${USER} | grep gnnplus_sweep"
echo "Logs:    logs_gnnplus/sweep_agent_<JOBID>_<TASK>.log"
