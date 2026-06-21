#!/usr/bin/env bash
# =============================================================================
# SLURM array → W&B sweep agent for GNNPlus hybrid_gnn sweeps.
#
#   SWEEP_ID=weber-geoml-harvard-university/GNNPlus/<id> \
#   SWEEP_DATASET=mnist \
#     sbatch --array=1-16%8 bash_interface/sweeps/run_wandb_sweep_agent.sh
#
# Env:
#   SWEEP_ID          — required (entity/project/sweep_id)
#   SWEEP_DATASET     — optional; sets mem (coco/voc/cluster/pattern/pcba → 128GB)
#   RUNS_PER_AGENT    — wandb agent --count (default 2; 0 = until sweep ends)
#   WANDB_PROJECT     — default GNNPlus
#   ENV_NAME          — default gnnplus
# =============================================================================

#SBATCH --job-name=gnnplus_sweep
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/sweep_agent_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

if [ -z "${SWEEP_ID:-}" ]; then
    echo "SWEEP_ID is not set" >&2
    exit 2
fi

RUNS_PER_AGENT="${RUNS_PER_AGENT:-2}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=../cluster/common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"

case "${SWEEP_DATASET:-}" in
    coco|voc|cluster|pattern|pcba)
        if [ "${SLURM_MEM_PER_NODE:-}" = "" ] || [ "${SLURM_MEM_PER_NODE:-0}" -lt 128000 ]; then
            echo "[sweep_agent] note: ${SWEEP_DATASET} may need 128GB; submit with --mem=128GB if OOM"
        fi
        ;;
esac

if ! python -c "import wandb" >/dev/null 2>&1; then
    python -m pip install wandb --quiet
fi

log_message "wandb agent SWEEP_ID=${SWEEP_ID} runs_per_agent=${RUNS_PER_AGENT}"

_wandb_agent_extra=()
if python -m wandb agent --help 2>&1 | grep -Fq -- '--forward-signals'; then
    _wandb_agent_extra+=(--forward-signals)
fi

if [ "${RUNS_PER_AGENT}" = "0" ]; then
    exec python -m wandb agent "${_wandb_agent_extra[@]}" "${SWEEP_ID}"
fi
exec python -m wandb agent "${_wandb_agent_extra[@]}" --count "${RUNS_PER_AGENT}" "${SWEEP_ID}"
