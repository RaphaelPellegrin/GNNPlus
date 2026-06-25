#!/usr/bin/env bash
# Create + launch W&B best-hybrid Bayes sweeps for PATTERN, CLUSTER, MalNet-Tiny.
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export ENV_NAME=gnnplus
#   conda deactivate 2>/dev/null || true
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_best_hybrid_sweep_suite.sh
#   bash bash_interface/cluster/submit_best_hybrid_sweep_suite.sh pattern
#   bash bash_interface/cluster/submit_best_hybrid_sweep_suite.sh --create-only cluster mal
#
# Optional env:
#   SWEEP_ARRAY_TASKS=8 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=4 SWEEP_SLURM_TIME=120:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export ENV_NAME="${ENV_NAME:-gnnplus}"

ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-8}"
ARRAY_PARALLEL="${SWEEP_ARRAY_PARALLEL:-4}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-4}"
SLURM_TIME="${SWEEP_SLURM_TIME:-120:00:00}"

CREATE_ONLY=0
if [ "${1:-}" = "--create-only" ]; then
    CREATE_ONLY=1
    shift
fi

DEFAULT_ORDER=(pattern cluster mal)

sweep_mem() {
    case "$1" in
        pattern|cluster) echo "128GB" ;;
        mal) echo "64GB" ;;
        *) echo "64GB" ;;
    esac
}

launch_dataset() {
    local slug="$1"
    local yaml="bash_interface/sweeps/${slug}_best_hybrid_sweep.yaml"
    local mem sweep_id

    if [ ! -f "${yaml}" ]; then
        echo "ERROR: missing ${yaml}" >&2
        return 1
    fi

    echo "=== Create sweep: ${slug} (${yaml}) ==="
    bash bash_interface/sweeps/create_sweep.sh "${yaml}"
    sweep_id="$(cat bash_interface/sweeps/.last_sweep_id)"
    mem="$(sweep_mem "${slug}")"

    echo "=== Launch agents: ${slug} sweep=${sweep_id} mem=${mem} time=${SLURM_TIME} ==="
    if [ "${CREATE_ONLY}" -eq 1 ]; then
        echo "  [create-only] skip sbatch"
        return 0
    fi

    SWEEP_ARRAY_TASKS="${ARRAY_TASKS}" \
    SWEEP_ARRAY_PARALLEL="${ARRAY_PARALLEL}" \
    RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
    SWEEP_SLURM_TIME="${SLURM_TIME}" \
    SWEEP_SLURM_MEM="${mem}" \
    bash bash_interface/sweeps/relaunch_sweep_agents.sh \
        "${slug}" "${sweep_id}"
}

if [ "$#" -gt 0 ]; then
    REQUESTED=("$@")
else
    REQUESTED=("${DEFAULT_ORDER[@]}")
fi

for name in "${REQUESTED[@]}"; do
  case "${name}" in
    pattern|cluster|mal) ;;
    *)
      echo "ERROR: unknown dataset '${name}'. Use: ${DEFAULT_ORDER[*]}" >&2
      exit 1
      ;;
  esac
  launch_dataset "${name}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs_gnnplus/sweep_agent_<JOBID>_<TASK>.log"
echo "W&B:     https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
