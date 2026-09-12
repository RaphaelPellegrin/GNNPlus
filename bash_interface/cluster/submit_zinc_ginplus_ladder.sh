#!/usr/bin/env bash
# Submit ZINC GIN+ staged ladder: 5 levels × 5 seeds = 25 jobs.
#
# Paper baseline: configs/gine/zinc.yaml (ICML 2025 GNN+ default, arXiv:2502.09263)
#
# Defaults: max 5 concurrent GPUs, --nice=10000 so other lab users get priority.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_zinc_ginplus_ladder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${ZINC_GINPLUS_LADDER_NUM_TASKS:-25}"
ARRAY_SPEC="${ZINC_GINPLUS_LADDER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${ZINC_GINPLUS_LADDER_PARALLEL:-5}"
NICE="${ZINC_GINPLUS_LADDER_NICE:-10000}"
PARTITION="${ZINC_GINPLUS_LADDER_PARTITION:-gpu_h200}"
MEM="${ZINC_GINPLUS_LADDER_MEM:-64GB}"

if [[ "${PARTITION}" == *"h200"* ]]; then
    DEFAULT_TIME="72:00:00"
else
    DEFAULT_TIME="240:00:00"
fi
TIME="${ZINC_GINPLUS_LADDER_TIME:-${DEFAULT_TIME}}"
WANDB_GROUP="${ZINC_GINPLUS_LADDER_WANDB_GROUP:-zinc_ginplus_ladder}"

sbatch_args=(
    --parsable
    --job-name=zinc_gin_ladder
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/zinc_gin_ladder_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,ZINC_GINPLUS_LADDER_NUM_TASKS="${NUM_TASKS}",ZINC_GINPLUS_LADDER_NUM_SEEDS="5",ZINC_GINPLUS_LADDER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_zinc_ginplus_ladder.sh
)"

echo ""
echo "=== ZINC GIN+ staged ladder submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (5 levels × 5 seeds = ${NUM_TASKS} total), parallel=${PARALLEL}"
echo "  Partition:    ${PARTITION}"
echo "  SLURM nice:   ${NICE} (0 = normal priority; higher = lower priority)"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/zinc_gin_ladder_${job_id}_<TASK>.log"
echo ""
echo "  Level 0 (tasks 1–5):   paper GIN+ baseline @ 80 (configs/gine/zinc.yaml)"
echo "  Level 1 (tasks 6–10):  GIN+ + headwise gate @ 80 (a0g1)"
echo "  Level 2 (tasks 11–15): hybrid a1g1 @ 80 (1×attn + 1×GINE MP, gated)"
echo "  Level 3 (tasks 16–20): hybrid a0g2 @ 80 (GINE + GatedGCN, gated, no attn)"
echo "  Level 4 (tasks 21–25): full hybrid a1g2 @ 80 (1×attn + GINE + GatedGCN, gated)"
echo ""
echo "  W&B tags: level_0 … level_4 (+ ginplus_ladder, zinc)"
echo ""
echo "Aggregate when done (per level; lower MAE is better):"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag level_4"
echo ""
