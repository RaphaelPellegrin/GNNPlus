#!/usr/bin/env bash
# Submit MalNet-Tiny GatedGCN+ staged ladder: 5 levels × 5 seeds = 25 jobs.
#
# Paper baseline: configs/gatedgcn/mal.yaml (ICML 2025 GNN+ default, arXiv:2502.09263)
#
# Defaults: max 5 concurrent GPUs, --nice=10000 so other lab users get priority.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_mal_gatedgcnplus_ladder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${MAL_GATEDGCNPLUS_LADDER_NUM_TASKS:-25}"
ARRAY_SPEC="${MAL_GATEDGCNPLUS_LADDER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${MAL_GATEDGCNPLUS_LADDER_PARALLEL:-5}"
NICE="${MAL_GATEDGCNPLUS_LADDER_NICE:-10000}"
PARTITION="${MAL_GATEDGCNPLUS_LADDER_PARTITION:-gpu_h200}"
MEM="${MAL_GATEDGCNPLUS_LADDER_MEM:-64GB}"

if [[ "${PARTITION}" == *"h200"* ]]; then
    DEFAULT_TIME="72:00:00"
else
    DEFAULT_TIME="240:00:00"
fi
TIME="${MAL_GATEDGCNPLUS_LADDER_TIME:-${DEFAULT_TIME}}"
WANDB_GROUP="${MAL_GATEDGCNPLUS_LADDER_WANDB_GROUP:-mal_gatedgcnplus_ladder}"

sbatch_args=(
    --parsable
    --job-name=mal_gated_ladder
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/mal_gated_ladder_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,MAL_GATEDGCNPLUS_LADDER_NUM_TASKS="${NUM_TASKS}",MAL_GATEDGCNPLUS_LADDER_NUM_SEEDS="5",MAL_GATEDGCNPLUS_LADDER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_mal_gatedgcnplus_ladder.sh
)"

echo ""
echo "=== MalNet-Tiny GatedGCN+ staged ladder submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (5 levels × 5 seeds = ${NUM_TASKS} total), parallel=${PARALLEL}"
echo "  Partition:    ${PARTITION}"
echo "  SLURM nice:   ${NICE} (0 = normal priority; higher = lower priority)"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/mal_gated_ladder_${job_id}_<TASK>.log"
echo ""
echo "  Level 0 (tasks 1–5):   paper GatedGCN+ baseline @ 100 (configs/gatedgcn/mal.yaml)"
echo "  Level 1 (tasks 6–10):  GatedGCN+ + headwise gate @ 100 (a0g1)"
echo "  Level 2 (tasks 11–15): hybrid a1g1 @ 100 (1×attn + 1×GatedGCN MP, gated, graph_restricted)"
echo "  Level 3 (tasks 16–20): hybrid a0g2 @ 100 (GatedGCN + GINE, gated, no attn)"
echo "  Level 4 (tasks 21–25): full hybrid a1g2 @ 100 (1×attn + GatedGCN + GINE, gated, graph_restricted)"
echo ""
echo "  W&B tags: level_0 … level_4 (+ gatedgcnplus_ladder, malnet)"
echo ""
echo "Aggregate when done (per level; higher accuracy is better):"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag level_4"
echo ""
