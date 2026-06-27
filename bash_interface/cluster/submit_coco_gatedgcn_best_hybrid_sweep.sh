#!/usr/bin/env bash
# Create + launch COCO-SP GatedGCN+ best-hybrid Bayes sweep (anchor 5b4z9l3u, seed 1).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_coco_gatedgcn_best_hybrid_sweep.sh
#   bash bash_interface/cluster/submit_coco_gatedgcn_best_hybrid_sweep.sh --create-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export ENV_NAME="${ENV_NAME:-gnnplus}"

ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-16}"
ARRAY_PARALLEL="${SWEEP_ARRAY_PARALLEL:-2}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-4}"
SLURM_TIME="${SWEEP_SLURM_TIME:-192:00:00}"
SLURM_MEM="${SWEEP_SLURM_MEM:-128GB}"
YAML="bash_interface/sweeps/coco_gatedgcn_best_hybrid_sweep.yaml"

CREATE_ONLY=0
if [ "${1:-}" = "--create-only" ]; then
    CREATE_ONLY=1
fi

echo "=== Create sweep: coco gatedgcn (${YAML}) ==="
bash bash_interface/sweeps/create_sweep.sh "${YAML}"
SWEEP_ID="$(cat bash_interface/sweeps/.last_sweep_id)"

if [ "${CREATE_ONLY}" -eq 1 ]; then
    echo "Sweep id: ${SWEEP_ID}"
    exit 0
fi

echo "=== Launch agents: coco sweep=${SWEEP_ID} mem=${SLURM_MEM} ==="
SWEEP_ARRAY_TASKS="${ARRAY_TASKS}" \
SWEEP_ARRAY_PARALLEL="${ARRAY_PARALLEL}" \
RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
SWEEP_SLURM_TIME="${SLURM_TIME}" \
SWEEP_SLURM_MEM="${SLURM_MEM}" \
bash bash_interface/sweeps/relaunch_sweep_agents.sh coco "${SWEEP_ID}"

echo ""
echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/sweeps/$(basename "${SWEEP_ID}")"
echo "Baseline anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5b4z9l3u"
echo "Sweeps: attn {1,2}, d_h, LR, batch {8,16} | fixed 1×GATEDGCN MP, seed 1"
echo "Logs: logs_gnnplus/sweep_agent_<JOBID>_<TASK>.log"
