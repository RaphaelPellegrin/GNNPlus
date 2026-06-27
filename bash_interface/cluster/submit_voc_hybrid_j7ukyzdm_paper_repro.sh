#!/usr/bin/env bash
# Submit VOC-SP paper repro: j7ukyzdm anchor × 5 seeds (bestmodel_v1).
#
# Anchor run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/j7ukyzdm
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_voc_hybrid_j7ukyzdm_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
PARALLEL="${PAPER_REPRO_PARALLEL:-5}"
MEM="${VOC_PAPER_MEM:-128GB}"
TIME="${VOC_PAPER_TIME:-120:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=voc_paper_v1 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/voc_paper_v1_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}" \
        bash_interface/cluster/run_voc_hybrid_j7ukyzdm_paper_repro.sh
)"

echo ""
echo "=== VOC-SP paper repro (bestmodel_v1) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1)))"
echo "  Config:       configs/gated_hybrid/voc-hybrid-j7ukyzdm-anchor.yaml"
echo "  Logs:         logs_gnnplus/voc_paper_v1_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v1_voc_j7ukyzdm"
echo "  W&B tags:     paper_repro, bestmodel_v1, voc, anchor_j7ukyzdm"
echo "  Anchor:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/j7ukyzdm"
echo "  Metric:       best_test_perf (val-best epoch test/f1)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v1_voc_j7ukyzdm"
echo ""
