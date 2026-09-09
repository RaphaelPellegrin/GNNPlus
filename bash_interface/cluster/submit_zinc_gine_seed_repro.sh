#!/usr/bin/env bash
# Submit ZINC GINE baseline × 5 seeds (configs/gine/zinc.yaml).
# Partition: mweber_gpu, max 8 concurrent GPUs.
#
# Source lineage (same recipe):
#   gbdpt2gc / x6a7vgim / jb8domtt
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_zinc_gine_seed_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${ZINC_GINE_NUM_SEEDS:-5}"
ARRAY_SPEC="${ZINC_GINE_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${ZINC_GINE_PARALLEL:-8}"
MEM="${ZINC_GINE_MEM:-64GB}"
TIME="${ZINC_GINE_TIME:-240:00:00}"
WANDB_GROUP="${ZINC_GINE_WANDB_GROUP:-seed_repro_zinc_gine}"
PARTITION="${ZINC_GINE_PARTITION:-mweber_gpu}"

job_id="$(
    sbatch --parsable \
        --job-name=zinc_gine_seed_repro \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/zinc_gine_seed_repro_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,ZINC_GINE_NUM_SEEDS="${NUM_SEEDS}",ZINC_GINE_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_zinc_gine_seed_repro.sh
)"

echo ""
echo "=== ZINC GINE seed repro submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Partition:    ${PARTITION} (max ${PARALLEL} GPUs)"
echo "  Tasks:        ${ARRAY_SPEC} (seeds map task_id-1), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME}"
echo "  Config:       configs/gine/zinc.yaml"
echo "  Logs:         logs_gnnplus/zinc_gine_seed_repro_${job_id}_<TASK>.log"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Metric:       best/test_mae (test at best/val_mae epoch)"
echo ""
echo "Log JOBID in CLUSTER_LAUNCHES.md; check: squeue -j ${job_id}"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_seed_repro_groups.py \\"
echo "    --preset zinc_mal_top"
echo ""
