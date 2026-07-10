#!/usr/bin/env bash
# Submit peptides-struct hybrid v2: best hybrid (y3ygn39y) + UniGCN upgrades.
#
# Architecture: a2g2 (2×attn + GINE + UNIGCN), L8, ep=300, lr=7e-4.
#
# Defaults: 3 seeds, max 5 concurrent GPUs.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_struct_unigcn_hybrid_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PS_UNIGCN_V2_NUM_SEEDS:-3}"
NUM_TASKS="${PS_UNIGCN_V2_NUM_TASKS:-${NUM_SEEDS}}"
ARRAY_SPEC="${PS_UNIGCN_V2_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PS_UNIGCN_V2_PARALLEL:-5}"
NICE="${PS_UNIGCN_V2_NICE:-10000}"
MEM="${PS_UNIGCN_V2_MEM:-64GB}"
TIME="${PS_UNIGCN_V2_TIME:-120:00:00}"
WANDB_GROUP="${PS_UNIGCN_V2_WANDB_GROUP:-peptides_struct_unigcn_hybrid_v2}"

sbatch_args=(
    --parsable
    --job-name=ps_unigcn_v2
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/ps_unigcn_v2_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PS_UNIGCN_V2_NUM_TASKS="${NUM_TASKS}",PS_UNIGCN_V2_NUM_SEEDS="${NUM_SEEDS}",PS_UNIGCN_V2_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_struct_unigcn_hybrid_v2.sh
)"

echo ""
echo "=== Peptides-struct UniGCN hybrid v2 submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds 0..$((NUM_SEEDS - 1))), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a2g2-gine-unigcn-v2.yaml"
echo "  Architecture: a2g2 GINE+UNIGCN, L8, d_h=200, ep=300, lr=7e-4"
echo "  Anchor:       y3ygn39y / 63avcc5m + UniGCN upgrades"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/ps_unigcn_v2_${job_id}_<TASK>.log"
echo ""
