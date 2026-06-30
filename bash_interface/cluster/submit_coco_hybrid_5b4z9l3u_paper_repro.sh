#!/usr/bin/env bash
# Submit COCO-SP paper repro v2: 5b4z9l3u / q57ng7d2 anchor (a1g1) × 5 seeds on gpu_h200.
#
# Parent run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q57ng7d2
# Baseline:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5b4z9l3u
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u_paper_repro.sh
#
# Only seeds 3–4 (tasks 4–5) without touching an existing array:
#   PAPER_REPRO_ARRAY=4-5 bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u_paper_repro.sh
#
# Override partition: COCO_PAPER_PARTITION=mweber_gpu bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_NUM_SEEDS:-5}"
ARRAY_SPEC="${PAPER_REPRO_ARRAY:-1-${NUM_SEEDS}}"
MEM="${COCO_PAPER_MEM:-128GB}"
TIME="${COCO_PAPER_TIME:-72:00:00}"
PARTITION="${COCO_PAPER_PARTITION:-gpu_h200}"

job_id="$(
    sbatch --parsable \
        --job-name=coco_paper_v2 \
        --array="${ARRAY_SPEC}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/coco_paper_v2_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_coco_hybrid_5b4z9l3u_paper_repro.sh
)"

echo ""
echo "=== COCO-SP paper repro v2 (bestmodel_v2) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Partition:    ${PARTITION}"
echo "  Array spec:   ${ARRAY_SPEC} (no JobArrayTaskLimit)"
echo "  Time limit:   ${TIME} (300 epochs ≈ 65h on H200; gpu_h200 cap 72h)"
echo "  Config:       configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
echo "  Logs:         logs_gnnplus/coco_paper_v2_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v2_coco_5b4z9l3u"
echo "  Parent:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q57ng7d2"
echo "  Metric:       best_test_perf (val-best epoch test/f1)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v2_coco_5b4z9l3u"
echo ""
