#!/usr/bin/env bash
# Submit MalNet-Tiny paper repro v3: vcb1cuql anchor × 5 seeds.
#
# Anchor run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vcb1cuql
# Compare vs v1 (9h3jqzkm a0g2) and v2 (4j21kp8d a1g2).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_malnet_hybrid_vcb1cuql_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
PARALLEL="${PAPER_REPRO_PARALLEL:-5}"
MEM="${MALNET_PAPER_MEM:-64GB}"
TIME="${MALNET_PAPER_TIME:-48:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=malnet_paper_v3 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/malnet_paper_v3_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}" \
        bash_interface/cluster/run_malnet_hybrid_vcb1cuql_paper_repro.sh
)"

echo ""
echo "=== MalNet-Tiny paper repro v3 (bestmodel_v3) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1)))"
echo "  Config:       configs/gated_hybrid/malnet-hybrid-vcb1cuql-anchor.yaml"
echo "  Logs:         logs_gnnplus/malnet_paper_v3_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v3_malnet_vcb1cuql"
echo "  W&B tags:     paper_repro, bestmodel_v3, malnet, anchor_vcb1cuql"
echo "  Anchor:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vcb1cuql"
echo "  Metric:       best_test_perf (val-best epoch test/accuracy)"
echo "  Note:         v3 MalNet (a1g1 GCNE, d_h=64) vs v1/v2"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v3_malnet_vcb1cuql"
echo ""
