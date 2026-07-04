#!/usr/bin/env bash
# Submit peptides-struct paper repro: xfb9wdir a0g2 (2× gated GINE, no attention) × 5 seeds.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_struct_hybrid_xfb9wdir_a0g2_paper_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
ARRAY_SPEC="${PAPER_REPRO_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PAPER_REPRO_PARALLEL:-3}"
MEM="${PEPTIDES_STRUCT_PAPER_MEM:-64GB}"
TIME="${PEPTIDES_STRUCT_PAPER_TIME:-240:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=peptides_struct_a0g2_v2 \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/peptides_struct_a0g2_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_peptides_struct_hybrid_xfb9wdir_a0g2_paper_repro.sh
)"

echo ""
echo "=== peptides-struct paper repro (xfb9wdir a0g2 gated GINE×2) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds map task_id-1), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME} (500 epochs)"
echo "  Config:       configs/gated_hybrid/peptides-struct-hybrid-xfb9wdir-a0g2-anchor.yaml"
echo "  Logs:         logs_gnnplus/peptides_struct_a0g2_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_peptides_struct_xfb9wdir_a0g2_ep500"
echo "  Metric:       best_test_perf (val-best epoch test/mae, lower is better)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_peptides_struct_xfb9wdir_a0g2_ep500 \\"
echo "    --metric best_test_perf"
echo ""
