#!/usr/bin/env bash
# Submit peptides-func paper repro: 3g180qle lineage a1g8 (1 attn + 8× GCN) × 5 seeds.
#
# MP baseline: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3g180qle
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_hybrid_3g180qle_a1g8_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
ARRAY_SPEC="${PAPER_REPRO_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PAPER_REPRO_PARALLEL:-5}"
MEM="${PEPTIDES_FUNC_PAPER_MEM:-64GB}"
TIME="${PEPTIDES_FUNC_PAPER_TIME:-240:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=peptides_func_3g180qle_a1g8 \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/peptides_func_3g180qle_a1g8_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_peptides_func_hybrid_3g180qle_a1g8_paper_repro.sh
)"

echo ""
echo "=== peptides-func paper repro (3g180qle a1g8 GCN×8) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds map task_id-1), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME} (300 epochs)"
echo "  Config:       configs/gated_hybrid/peptides-func-hybrid-3g180qle-a1g8-anchor.yaml"
echo "  MP baseline:  https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3g180qle"
echo "  Logs:         logs_gnnplus/peptides_func_3g180qle_a1g8_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_peptides_func_3g180qle_a1g8_ep300"
echo "  Metric:       best_test_perf (val-best epoch test/ap, higher is better)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_peptides_func_3g180qle_a1g8_ep300 \\"
echo "    --metric best_test_perf"
echo ""
