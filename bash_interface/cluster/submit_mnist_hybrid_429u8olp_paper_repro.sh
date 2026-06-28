#!/usr/bin/env bash
# Submit MNIST paper repro v2: 429u8olp anchor × 5 seeds.
#
# Anchor run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/429u8olp
# Compare vs v1 cohort (lcvbyyss a2g2).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_mnist_hybrid_429u8olp_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
PARALLEL="${PAPER_REPRO_PARALLEL:-5}"
MEM="${MNIST_PAPER_MEM:-64GB}"
TIME="${MNIST_PAPER_TIME:-96:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=mnist_paper_v2 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/mnist_paper_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}" \
        bash_interface/cluster/run_mnist_hybrid_429u8olp_paper_repro.sh
)"

echo ""
echo "=== MNIST paper repro v2 (bestmodel_v2) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1)))"
echo "  Config:       configs/gated_hybrid/mnist-hybrid-429u8olp-anchor.yaml"
echo "  Logs:         logs_gnnplus/mnist_paper_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_mnist_429u8olp"
echo "  W&B tags:     paper_repro, bestmodel_v2, mnist, anchor_429u8olp"
echo "  Anchor:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/429u8olp"
echo "  Metric:       best_test_perf (val-best epoch test/accuracy)"
echo "  Note:         v2 MNIST (a8g2 GATEDGCN+GAT) vs v1 / lcvbyyss a2g2"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_mnist_429u8olp"
echo ""
