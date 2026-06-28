#!/usr/bin/env bash
# Submit COCO-SP paper repro v2: 5b4z9l3u / q57ng7d2 anchor (a1g1) × 5 seeds.
#
# Parent run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q57ng7d2
# Baseline:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5b4z9l3u
# vs v1 o5hr3tma (a2g8 sweep best).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
PARALLEL="${PAPER_REPRO_PARALLEL:-3}"
MEM="${COCO_PAPER_MEM:-128GB}"
TIME="${COCO_PAPER_TIME:-192:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=coco_paper_v2 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/coco_paper_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}" \
        bash_interface/cluster/run_coco_hybrid_5b4z9l3u_paper_repro.sh
)"

echo ""
echo "=== COCO-SP paper repro v2 (bestmodel_v2) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1))), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
echo "  Logs:         logs_gnnplus/coco_paper_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_coco_5b4z9l3u"
echo "  W&B tags:     paper_repro, bestmodel_v2, coco, anchor_5b4z9l3u, hybrid_a1g1"
echo "  Parent:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q57ng7d2"
echo "  Metric:       best_test_perf (val-best epoch test/f1)"
echo "  Note:         a1g1 GatedGCN+ anchor vs v1 o5hr3tma a2g8"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_coco_5b4z9l3u"
echo ""
