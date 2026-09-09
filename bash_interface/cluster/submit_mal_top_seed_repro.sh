#!/usr/bin/env bash
# Submit MalNet-Tiny top-run seed repro: 4 variants × 5 seeds = 20 jobs.
# Partition: mweber_gpu, max 8 concurrent GPUs.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_mal_top_seed_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${MAL_TOP_NUM_SEEDS:-5}"
NUM_VARIANTS=4
NUM_TASKS=$((NUM_VARIANTS * NUM_SEEDS))
ARRAY_SPEC="${MAL_TOP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${MAL_TOP_PARALLEL:-8}"
MEM="${MAL_TOP_MEM:-64GB}"
TIME="${MAL_TOP_TIME:-96:00:00}"
WANDB_PREFIX="${MAL_TOP_WANDB_PREFIX:-seed_repro_mal}"
PARTITION="${MAL_TOP_PARTITION:-mweber_gpu}"

job_id="$(
    sbatch --parsable \
        --job-name=mal_top_seed_repro \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/mal_top_seed_repro_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,MAL_TOP_NUM_SEEDS="${NUM_SEEDS}",MAL_TOP_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_mal_top_seed_repro.sh
)"

echo ""
echo "=== MalNet top seed repro submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Partition:    ${PARTITION} (max ${PARALLEL} GPUs)"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME}"
echo "  Variants:     v4cytwe0 | zk6ihqi8 | apiw6l3u | 5sx7r420"
echo "  Logs:         logs_gnnplus/mal_top_seed_repro_${job_id}_<TASK>.log"
echo "  W&B groups:   ${WANDB_PREFIX}_<variant>"
echo "  Metric:       best/test_accuracy (test at best/val_accuracy epoch)"
echo ""
echo "Log JOBID in CLUSTER_LAUNCHES.md; check: squeue -j ${job_id}"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_seed_repro_groups.py \\"
echo "    --preset zinc_mal_top"
echo ""
