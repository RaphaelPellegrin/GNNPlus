#!/usr/bin/env bash
# Submit ZINC fairness ladder: 5 levels × 3 seeds = 15 jobs.
#
# Paper baseline: configs/gcn/zinc.yaml (ICML 2025 GNN+ default, arXiv:2502.09263)
#
# Defaults: max 5 concurrent GPUs, --nice=10000 so other lab users get priority.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_zinc_fairness_ladder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${ZINC_FAIRNESS_LADDER_NUM_TASKS:-15}"
ARRAY_SPEC="${ZINC_FAIRNESS_LADDER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${ZINC_FAIRNESS_LADDER_PARALLEL:-5}"
NICE="${ZINC_FAIRNESS_LADDER_NICE:-10000}"
MEM="${ZINC_LADDER_MEM:-64GB}"
TIME="${ZINC_LADDER_TIME:-240:00:00}"
WANDB_GROUP="${ZINC_FAIRNESS_LADDER_WANDB_GROUP:-zinc_fairness_ladder}"

sbatch_args=(
    --parsable
    --job-name=zinc_ladder
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/zinc_ladder_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,ZINC_FAIRNESS_LADDER_NUM_TASKS="${NUM_TASKS}",ZINC_FAIRNESS_LADDER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_zinc_fairness_ladder.sh
)"

echo ""
echo "=== ZINC fairness ladder submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (5 levels × 3 seeds), parallel=${PARALLEL}"
echo "  SLURM nice:   ${NICE} (0 = normal priority; higher = lower priority)"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/zinc_ladder_${job_id}_<TASK>.log"
echo ""
echo "  Level 0 (tasks 1–3):   paper GCNE baseline @ 64 (configs/gcn/zinc.yaml)"
echo "  Level 1 (tasks 4–6):   GCNE + headwise gate @ 64"
echo "  Level 2 (tasks 7–9):   hybrid a0g1 @ d_h=64"
echo "  Level 3 (tasks 10–12): hybrid a0g2 @ d_h=64 (no attn)"
echo "  Level 4 (tasks 13–15): hybrid a1g1 @ d_h=64 (attn + MP)"
echo ""
echo "  W&B tags: level_0 … level_4 (+ fairness_ladder, zinc)"
echo ""
echo "Aggregate when done (per level; lower MAE is better):"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag level_4"
echo ""
