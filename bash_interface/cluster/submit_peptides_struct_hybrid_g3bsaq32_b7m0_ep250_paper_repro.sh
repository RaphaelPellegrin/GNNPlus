#!/usr/bin/env bash
# Submit peptides-struct paper repro: g3bsaq32 b7_m0 × 10 seeds, ep=250 (exact repro).
#
# Source: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/g3bsaq32
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_g3bsaq32_b7m0_ep250_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-10}"
ARRAY_SPEC="${PAPER_REPRO_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PAPER_REPRO_PARALLEL:-10}"
MEM="${PEPTIDES_STRUCT_PAPER_MEM:-64GB}"
TIME="${PEPTIDES_STRUCT_PAPER_TIME:-240:00:00}"
WANDB_GROUP="paper_bestmodel_v2_peptides_struct_g3bsaq32_b7m0_ep250"

job_id="$(
    sbatch --parsable \
        --job-name=peptides_struct_g3bsaq32_ep250 \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/peptides_struct_g3bsaq32_ep250_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",PAPER_MAX_EPOCH=250,PAPER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_peptides_struct_hybrid_g3bsaq32_b7m0_paper_repro.sh
)"

echo ""
echo "=== peptides-struct g3bsaq32 b7_m0 paper repro (ep250) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds 0–9), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
echo "  Source run:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/g3bsaq32"
echo "  W&B group:    ${WANDB_GROUP}"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf"
echo ""
