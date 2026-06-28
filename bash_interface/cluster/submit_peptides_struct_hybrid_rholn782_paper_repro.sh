#!/usr/bin/env bash
# Submit peptides-struct paper repro v1: MOE_6 rholn782 hybrid anchor × 5 seeds.
#
# MOE anchor: https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/rholn782
# GNNPlus GINE baseline (no VN): https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xfb9wdir
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_rholn782_paper_repro.sh
#
# If OOM with virtual nodes, retry with smaller batch:
#   PEPTIDES_PAPER_BATCH_SIZE=32 bash bash_interface/cluster/submit_peptides_struct_hybrid_rholn782_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
PARALLEL="${PAPER_REPRO_PARALLEL:-3}"
MEM="${PEPTIDES_PAPER_MEM:-128GB}"
TIME="${PEPTIDES_PAPER_TIME:-192:00:00}"
BATCH_SIZE="${PEPTIDES_PAPER_BATCH_SIZE:-64}"

job_id="$(
    sbatch --parsable \
        --job-name=peptides_struct_paper_v1 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/peptides_struct_paper_v1_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",PEPTIDES_PAPER_BATCH_SIZE="${BATCH_SIZE}" \
        bash_interface/cluster/run_peptides_struct_hybrid_rholn782_paper_repro.sh
)"

echo ""
echo "=== peptides-struct paper repro v1 (bestmodel_v1) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1))), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-struct-hybrid-rholn782-anchor.yaml"
echo "  Batch size:   ${BATCH_SIZE} (override: PEPTIDES_PAPER_BATCH_SIZE=32)"
echo "  Logs:         logs_gnnplus/peptides_struct_paper_v1_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v1_peptides_struct_rholn782"
echo "  W&B tags:     paper_repro, bestmodel_v1, peptides_struct, anchor_rholn782, vn4"
echo "  MOE anchor:   https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/rholn782"
echo "  Metric:       best_test_perf (val-best epoch test/mae)"
echo "  Note:         hybrid a2g2 GINE+GGNN L12/H96 + 4 virtual nodes (MOE target < 0.23)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v1_peptides_struct_rholn782"
echo ""
