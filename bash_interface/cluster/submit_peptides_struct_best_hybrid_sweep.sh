#!/usr/bin/env bash
# Create + launch peptides-struct best-hybrid Bayes sweep (MOE rholn782 repro).
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export ENV_NAME=gnnplus
#   conda deactivate 2>/dev/null || true
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_peptides_struct_best_hybrid_sweep.sh
#   bash bash_interface/cluster/submit_peptides_struct_best_hybrid_sweep.sh --create-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export ENV_NAME="${ENV_NAME:-gnnplus}"

ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-16}"
ARRAY_PARALLEL="${SWEEP_ARRAY_PARALLEL:-8}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-4}"
SLURM_TIME="${SWEEP_SLURM_TIME:-120:00:00}"
YAML="bash_interface/sweeps/peptides_struct_best_hybrid_sweep.yaml"

CREATE_ONLY=0
if [ "${1:-}" = "--create-only" ]; then
    CREATE_ONLY=1
fi

echo "=== Create sweep: peptides_struct (${YAML}) ==="
bash bash_interface/sweeps/create_sweep.sh "${YAML}"
SWEEP_ID="$(cat bash_interface/sweeps/.last_sweep_id)"

if [ "${CREATE_ONLY}" -eq 1 ]; then
    echo "Sweep id: ${SWEEP_ID}"
    exit 0
fi

echo "=== Launch agents: peptides_struct sweep=${SWEEP_ID} ==="
SWEEP_ARRAY_TASKS="${ARRAY_TASKS}" \
SWEEP_ARRAY_PARALLEL="${ARRAY_PARALLEL}" \
RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
SWEEP_SLURM_TIME="${SLURM_TIME}" \
bash bash_interface/sweeps/relaunch_sweep_agents.sh peptides_struct "${SWEEP_ID}"

echo ""
echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/sweeps/$(basename "${SWEEP_ID}")"
echo "Filter: config.add_virtual_nodes, config.hybrid_readout_mlp, summary.preprocess/*"
