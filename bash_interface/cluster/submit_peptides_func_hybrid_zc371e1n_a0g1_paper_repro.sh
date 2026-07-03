#!/usr/bin/env bash
# Submit peptides-func paper repro v2: zc371e1n a0g1 (gated GCN only, no attention) × 5 seeds.
#
# Parent hybrid: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/zc371e1n
# MP baseline:    https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3g180qle
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull   # needs wandb.group support in config/train
#   bash bash_interface/cluster/submit_peptides_func_hybrid_zc371e1n_a0g1_paper_repro.sh
#
# Seed 2 only (task 3):
#   PAPER_REPRO_ARRAY=3-3 bash bash_interface/cluster/submit_peptides_func_hybrid_zc371e1n_a0g1_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
ARRAY_SPEC="${PAPER_REPRO_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PAPER_REPRO_PARALLEL:-3}"
MEM="${PEPTIDES_FUNC_PAPER_MEM:-64GB}"
TIME="${PEPTIDES_FUNC_PAPER_TIME:-120:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=peptides_func_a0g1_v2 \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/peptides_func_a0g1_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_peptides_func_hybrid_zc371e1n_a0g1_paper_repro.sh
)"

echo ""
echo "=== peptides-func paper repro v2 (zc371e1n a0g1 gated GCN) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds map task_id-1), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME} (600 epochs)"
echo "  Config:       configs/gated_hybrid/peptides-func-hybrid-zc371e1n-a0g1-anchor.yaml"
echo "  Logs:         logs_gnnplus/peptides_func_a0g1_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_peptides_func_zc371e1n_a0g1"
echo "  Parent a1g1:  https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/zc371e1n"
echo "  MP baseline:  https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3g180qle"
echo "  Metric:       best_test_perf (val-best epoch test/ap)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_peptides_func_zc371e1n_a0g1"
echo ""
