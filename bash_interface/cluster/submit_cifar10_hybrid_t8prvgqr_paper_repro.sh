#!/usr/bin/env bash
# Submit CIFAR10 paper repro v2: t8prvgqr anchor (a4g4) × 5 seeds on gpu_h200.
#
# Parent run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/t8prvgqr
# vs v1 ulij45a2 (a8g4, d_h=256 on mweber_gpu).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_cifar10_hybrid_t8prvgqr_paper_repro.sh
#
# Override partition: CIFAR_PAPER_PARTITION=mweber_gpu bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
PARALLEL="${PAPER_REPRO_PARALLEL:-5}"
MEM="${CIFAR_PAPER_MEM:-128GB}"
TIME="${CIFAR_PAPER_TIME:-72:00:00}"
PARTITION="${CIFAR_PAPER_PARTITION:-gpu_h200}"

job_id="$(
    sbatch --parsable \
        --job-name=cifar10_paper_v2 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/cifar10_paper_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}" \
        bash_interface/cluster/run_cifar10_hybrid_t8prvgqr_paper_repro.sh
)"

echo ""
echo "=== CIFAR10 paper repro v2 (bestmodel_v2) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Partition:    ${PARTITION}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1)))"
echo "  Config:       configs/gated_hybrid/cifar10-hybrid-t8prvgqr-anchor.yaml"
echo "  Logs:         logs_gnnplus/cifar10_paper_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_cifar10_t8prvgqr"
echo "  W&B tags:     paper_repro, bestmodel_v2, cifar10, anchor_t8prvgqr, hybrid_a4g4"
echo "  Parent:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/t8prvgqr"
echo "  Metric:       best_test_perf (val-best epoch test/accuracy)"
echo "  Note:         a4g4 d_h=128 vs v1 ulij45a2 a8g4 d_h=256"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_cifar10_t8prvgqr"
echo ""
