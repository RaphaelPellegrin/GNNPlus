#!/usr/bin/env bash
# Launch peptides-struct tfeksgbl phase-2 sweeps A/B/C with shared GPU cap.
#
# Default: 12 GPUs total → 4 parallel agents per sweep (4 + 4 + 4).
#
# Usage (cluster login, after git pull):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#
#   # Create all three sweeps + launch:
#   bash bash_interface/sweeps/launch_peptides_struct_tfeksgbl_phase2_sweeps.sh --create
#
#   # Relaunch agents only (IDs from sweeps.log or env):
#   bash bash_interface/sweeps/launch_peptides_struct_tfeksgbl_phase2_sweeps.sh
#
# Env:
#   TOTAL_GPU_PARALLEL=12   max GPUs across all 3 sweeps (default 12, must be divisible by 3)
#   SWEEP_ARRAY_TASKS=16
#   RUNS_PER_AGENT=3
#   SWEEP_SLURM_TIME=240:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
SWEEPS_DIR="${REPO_ROOT}/bash_interface/sweeps"
LOG_FILE="${SWEEPS_DIR}/sweeps.log"

TOTAL_GPU_PARALLEL="${TOTAL_GPU_PARALLEL:-12}"
SWEEP_ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-16}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-3}"
SWEEP_SLURM_TIME="${SWEEP_SLURM_TIME:-240:00:00}"

SWEEP_A_YAML="${SWEEPS_DIR}/peptides_struct_hybrid_tfeksgbl_sweep_a_reg.yaml"
SWEEP_B_YAML="${SWEEPS_DIR}/peptides_struct_hybrid_tfeksgbl_sweep_b_scale.yaml"
SWEEP_C_YAML="${SWEEPS_DIR}/peptides_struct_hybrid_tfeksgbl_sweep_c_vn.yaml"

if [ $((TOTAL_GPU_PARALLEL % 3)) -ne 0 ]; then
    echo "TOTAL_GPU_PARALLEL must be divisible by 3 (split evenly across A/B/C); got ${TOTAL_GPU_PARALLEL}" >&2
    exit 2
fi
PARALLEL_EACH=$((TOTAL_GPU_PARALLEL / 3))

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
    echo "=== Creating sweep A (regularization) ==="
    bash "${SWEEPS_DIR}/create_sweep.sh" "${SWEEP_A_YAML}"
    echo ""
    echo "=== Creating sweep B (architecture scale) ==="
    bash "${SWEEPS_DIR}/create_sweep.sh" "${SWEEP_B_YAML}"
    echo ""
    echo "=== Creating sweep C (VN + readout) ==="
    bash "${SWEEPS_DIR}/create_sweep.sh" "${SWEEP_C_YAML}"
    echo ""
fi

SWEEP_A_ID="${SWEEP_A_ID:-$(lookup_sweep_id "${SWEEP_A_YAML}" || true)}"
SWEEP_B_ID="${SWEEP_B_ID:-$(lookup_sweep_id "${SWEEP_B_YAML}" || true)}"
SWEEP_C_ID="${SWEEP_C_ID:-$(lookup_sweep_id "${SWEEP_C_YAML}" || true)}"

if [ -z "${SWEEP_A_ID}" ] || [ -z "${SWEEP_B_ID}" ] || [ -z "${SWEEP_C_ID}" ]; then
    echo "Set SWEEP_A_ID, SWEEP_B_ID, SWEEP_C_ID, or run with --create first." >&2
    echo "  A=${SWEEP_A_ID:-<missing>}" >&2
    echo "  B=${SWEEP_B_ID:-<missing>}" >&2
    echo "  C=${SWEEP_C_ID:-<missing>}" >&2
    exit 2
fi

echo "=== Launching phase-2 agents (max ${TOTAL_GPU_PARALLEL} GPUs = ${PARALLEL_EACH} per sweep) ==="
echo "  A (reg):   ${SWEEP_A_ID}"
echo "  B (scale): ${SWEEP_B_ID}"
echo "  C (vn):    ${SWEEP_C_ID}"
echo ""

relaunch_one() {
    local sweep_id="$1"
    SWEEP_SLURM_TIME="${SWEEP_SLURM_TIME}" \
    SWEEP_ARRAY_TASKS="${SWEEP_ARRAY_TASKS}" \
    SWEEP_ARRAY_PARALLEL="${PARALLEL_EACH}" \
    RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
        bash "${SWEEPS_DIR}/relaunch_sweep_agents.sh" peptides_struct "${sweep_id}"
}

relaunch_one "${SWEEP_A_ID}"
relaunch_one "${SWEEP_B_ID}"
relaunch_one "${SWEEP_C_ID}"

echo ""
echo "Monitor: squeue -u \${USER} | grep gnnplus_sweep_peptides_struct"
