#!/usr/bin/env bash
# Submit peptides-struct UniGCN:
#   (A) custom_gnn unitarygcn baseline
#   (B) hybrid a1g2 = y3ygn39y (1 attn + GINE) + 1×UNIGCN
#
# Defaults: 3 seeds × 2 variants = 6 jobs, max 5 concurrent GPUs.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_struct_unigcn_baseline_vs_hybrid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PS_UNIGCN_NUM_SEEDS:-3}"
NUM_TASKS=$((2 * NUM_SEEDS))
ARRAY_SPEC="${PS_UNIGCN_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PS_UNIGCN_PARALLEL:-5}"
NICE="${PS_UNIGCN_NICE:-10000}"
MEM="${PS_UNIGCN_MEM:-64GB}"
TIME="${PS_UNIGCN_TIME:-120:00:00}"
WANDB_GROUP="${PS_UNIGCN_WANDB_GROUP:-peptides_struct_unigcn_baseline_vs_hybrid}"

sbatch_args=(
    --parsable
    --job-name=ps_unigcn
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/ps_unigcn_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PS_UNIGCN_NUM_SEEDS="${NUM_SEEDS}",PS_UNIGCN_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_struct_unigcn_baseline_vs_hybrid.sh
)"

echo ""
echo "=== Peptides-struct UniGCN baseline vs hybrid submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (2 variants × ${NUM_SEEDS} seeds), parallel=${PARALLEL}"
echo "  (A) custom:   configs/gcn/peptides-struct-unigcn.yaml"
echo "  (B) hybrid:   configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a1g2-gine-unigcn.yaml"
echo "                (y3ygn39y a1g1 GINE + 1×UNIGCN → a1g2)"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/ps_unigcn_${job_id}_<TASK>.log"
echo "  Source run:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/y3ygn39y"
echo ""
