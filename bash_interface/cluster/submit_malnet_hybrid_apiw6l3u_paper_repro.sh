#!/usr/bin/env bash
# Submit MalNet-Tiny paper repro v4: apiw6l3u lineage (a0g3) × 5 seeds.
#
# Parent run (a0g2, seed 2): https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u
# vs v1 9h3jqzkm (a0g2), v2 4j21kp8d (a1g2), v3 vcb1cuql (a1g1).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#   bash bash_interface/cluster/submit_malnet_hybrid_apiw6l3u_paper_repro.sh

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
        --job-name=malnet_paper_v4 \
        --array="1-${NUM_SEEDS}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/malnet_paper_v4_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,PAPER_NUM_SEEDS="${NUM_SEEDS}" \
        bash_interface/cluster/run_malnet_hybrid_apiw6l3u_paper_repro.sh
)"

echo ""
echo "=== MalNet-Tiny paper repro v4 (bestmodel_v4) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        1-${NUM_SEEDS} (seeds 0-$((NUM_SEEDS - 1)))"
echo "  Config:       configs/gated_hybrid/malnet-hybrid-apiw6l3u-a0g3-anchor.yaml"
echo "  Logs:         logs_gnnplus/malnet_paper_v4_${job_id}_<TASK>.log"
echo "  W&B group:    paper_bestmodel_v4_malnet_apiw6l3u"
echo "  W&B tags:     paper_repro, bestmodel_v4, malnet, anchor_apiw6l3u, hybrid_a0g3"
echo "  Parent:       https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u (a0g2)"
echo "  Architecture: 0×attn + 3×GCNE MP, d_h=110, graph_restricted, elementwise+rmsnorm"
echo "  Metric:       best_test_perf (val-best epoch test/accuracy)"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group paper_bestmodel_v4_malnet_apiw6l3u"
echo ""
