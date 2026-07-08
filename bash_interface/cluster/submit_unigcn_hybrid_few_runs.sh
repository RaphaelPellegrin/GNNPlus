#!/usr/bin/env bash
# Submit UniGCN hybrid few-run sweep:
#   4 datasets × 3 variants × 3 seeds = 36 jobs (default)
#
# Variants:
#   a0g1 — gated UniGCN MP only
#   a1g1 — 1×attn + gated UniGCN
#   a1g2 — 1×attn + gated UniGCN + GINE
#
# Datasets: peptides-func, peptides-struct, CLUSTER, PATTERN
#
# Defaults: max 3 concurrent GPUs, --nice=10000.
#
# Usage (on Harvard cluster):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_unigcn_hybrid_few_runs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${UNIGCN_NUM_SEEDS:-3}"
NUM_VARIANTS="${UNIGCN_NUM_VARIANTS:-3}"
NUM_DATASETS="${UNIGCN_NUM_DATASETS:-4}"
NUM_TASKS=$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))
ARRAY_SPEC="${UNIGCN_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${UNIGCN_PARALLEL:-3}"
NICE="${UNIGCN_NICE:-10000}"
MEM="${UNIGCN_MEM:-64GB}"
TIME="${UNIGCN_TIME:-120:00:00}"
WANDB_GROUP="${UNIGCN_WANDB_GROUP:-unigcn_hybrid_few_runs}"

sbatch_args=(
    --parsable
    --job-name=unigcn_hybrid
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/unigcn_hybrid_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,UNIGCN_NUM_SEEDS="${NUM_SEEDS}",UNIGCN_NUM_VARIANTS="${NUM_VARIANTS}",UNIGCN_NUM_DATASETS="${NUM_DATASETS}",UNIGCN_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_unigcn_hybrid_few_runs.sh
)"

echo ""
echo "=== UniGCN hybrid few-run sweep submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_DATASETS} datasets × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})"
echo "  Parallel:     ${PARALLEL} GPUs max concurrent"
echo "  Seeds:        0..$((NUM_SEEDS - 1)) per (dataset, variant) — 3 jobs share same cfg"
echo "  SLURM nice:   ${NICE}"
echo "  Config dir:   configs/gated_hybrid/unigcn/"
echo "  Variants:     a0g1 (gated UniGCN), a1g1 (+1 attn), a1g2 (+1 attn + GINE)"
echo "  Datasets:     peptides-func, peptides-struct, CLUSTER, PATTERN"
echo "  W&B group:    ${WANDB_GROUP}_<dataset>_<variant> (3 seeds each, 12 subgroups)"
echo "  Logs:         logs_gnnplus/unigcn_hybrid_${job_id}_<TASK>.log"
echo ""
echo "Aggregate (peptides-func AP example):"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag peptides_func"
echo ""
