#!/usr/bin/env bash
# =============================================================================
# Launch more W&B sweep agents on an **existing** sweep (no new sweep id).
#
# Usage:
#   source ~/.gnnplus_env
#   export WANDB_PROJECT=GNNPlus
#   export GNNPLUS_DATASET_DIR=...
#   cd /path/to/GNNPlus
#
#   # By slug + sweep id:
#   SWEEP_ARRAY_TASKS=24 RUNS_PER_AGENT=4 \
#     bash bash_interface/sweeps/relaunch_sweep_agents.sh \
#       mnist weber-geoml-harvard-university/GNNPlus/mhc71f9c
#
#   # Auto-pick latest sweep id for slug from sweeps.log:
#   bash bash_interface/sweeps/relaunch_sweep_agents.sh mnist
#
#   # Tier 1 (MNIST + CIFAR10) with more agents:
#   bash bash_interface/sweeps/relaunch_sweep_agents.sh tier1
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
SWEEPS_DIR="${REPO_ROOT}/bash_interface/sweeps"
LOG_FILE="${SWEEPS_DIR}/sweeps.log"

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export ENV_NAME="${ENV_NAME:-gnnplus}"

ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-24}"
ARRAY_PARALLEL="${SWEEP_ARRAY_PARALLEL:-8}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-4}"
SLURM_TIME="${SWEEP_SLURM_TIME:-96:00:00}"

sweep_mem() {
    case "$1" in
        coco|voc|cluster|pattern|pcba) echo "128GB" ;;
        ppa) echo "96GB" ;;
        zinc) echo "64GB" ;;
        *) echo "64GB" ;;
    esac
}

sweep_time() {
    case "$1" in
        zinc) echo "${SLURM_TIME:-192:00:00}" ;;
        *) echo "${SLURM_TIME}" ;;
    esac
}

lookup_sweep_id() {
    local slug="$1"
    local yaml="${SWEEPS_DIR}/${slug}_hybrid_gnnplus_sweep.yaml"
    if [ ! -f "${LOG_FILE}" ]; then
        echo "Missing ${LOG_FILE}" >&2
        return 1
    fi
    grep "${yaml}" "${LOG_FILE}" | tail -n 1 | awk '{print $3}'
}

relaunch_one() {
    local slug="$1"
    local sweep_id="$2"
    local mem time_budget

    if [ -z "${sweep_id}" ]; then
        sweep_id="$(lookup_sweep_id "${slug}")" || true
    fi
    if [ -z "${sweep_id}" ]; then
        echo "No sweep id for ${slug}; pass explicitly or create a sweep first." >&2
        return 1
    fi

    mem="$(sweep_mem "${slug}")"
    time_budget="$(sweep_time "${slug}")"

    echo "=== Relaunch agents: ${slug} sweep=${sweep_id} tasks=${ARRAY_TASKS} runs/agent=${RUNS_PER_AGENT} mem=${mem} ==="
    SWEEP_ID="${sweep_id}" \
    SWEEP_DATASET="${slug}" \
    RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
    sbatch \
        --job-name="gnnplus_sweep_${slug}" \
        --array="1-${ARRAY_TASKS}%${ARRAY_PARALLEL}" \
        --mem="${mem}" \
        --time="${time_budget}" \
        --export=ALL,SWEEP_ID="${sweep_id}",SWEEP_DATASET="${slug}",RUNS_PER_AGENT="${RUNS_PER_AGENT}",WANDB_PROJECT="${WANDB_PROJECT}",ENV_NAME="${ENV_NAME}" \
        bash_interface/sweeps/run_wandb_sweep_agent.sh
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <slug|tier1> [SWEEP_ID]" >&2
    echo "  tier1 = mnist + cifar10" >&2
    exit 2
fi

case "$1" in
    tier1)
        for slug in mnist cifar10; do
            relaunch_one "${slug}" "${2:-}"
        done
        ;;
    *)
        relaunch_one "$1" "${2:-}"
        ;;
esac

echo ""
echo "Logs: logs_gnnplus/sweep_agent_<JOBID>_<TASK>.log"
echo "Gate check: grep 'Hybrid gate stats: logging' logs_gnnplus/sweep_agent_<JOBID>_1.log"
