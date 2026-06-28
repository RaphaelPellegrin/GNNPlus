#!/usr/bin/env bash
# Submit peptides-struct paper repro v2: rholn782 hybrid, base_lr=6e-4 × 5 seeds.
#
# Parent run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tfeksgbl
# vs v1 (lr=4e-4): paper_bestmodel_v1_peptides_struct_rholn782
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_rholn782_lr6e-4_paper_repro.sh

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
        --job-name=peptides_struct_paper_v2 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/peptides_struct_paper_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",PEPTIDES_PAPER_BATCH_SIZE="${BATCH_SIZE}" \
        bash_interface/cluster/run_peptides_struct_hybrid_rholn782_lr6e-4_paper_repro.sh
)"

echo ""
echo "=== peptides-struct paper repro v2 (lr=6e-4) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1))), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-struct-hybrid-rholn782-lr6e-4-anchor.yaml"
echo "  base_lr:      0.0006 (v1 was 0.0004)"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Logs:         logs_gnnplus/peptides_struct_paper_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_peptides_struct_rholn782_lr6e-4"
echo "  Parent:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tfeksgbl"
echo "  Metric:       best_test_perf (val-best epoch test/mae)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_peptides_struct_rholn782_lr6e-4"
echo ""
