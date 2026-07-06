#!/usr/bin/env bash
# Submit MalNet-Tiny hybrid architecture ablation (apiw6l3u lineage).
#
# 4 variants × 2 seeds = 8 jobs (default), all base_lr=4e-3:
#   a5g5 | a0g10 | a0g50 | a50g50
#
# Baseline run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_arch_ablation.sh
#
# a50g50 may OOM at 64GB — retry with:
#   MALNET_ARCH_ABLATION_MEM=128GB ARCH_ABLATION_ARRAY=7-8 \
#     bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_arch_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_VARIANTS="${ARCH_ABLATION_NUM_VARIANTS:-4}"
NUM_SEEDS="${ARCH_ABLATION_NUM_SEEDS:-2}"
NUM_TASKS="${ARCH_ABLATION_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${ARCH_ABLATION_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${ARCH_ABLATION_PARALLEL:-4}"
MEM="${MALNET_ARCH_ABLATION_MEM:-64GB}"
TIME="${MALNET_ARCH_ABLATION_TIME:-96:00:00}"
MAX_EPOCH="${ARCH_ABLATION_MAX_EPOCH:-250}"
MIN_LR="${ARCH_ABLATION_MIN_LR:-1e-6}"
BASE_LR="${ARCH_ABLATION_BASE_LR:-0.004}"
WANDB_GROUP_PREFIX="${ARCH_ABLATION_WANDB_GROUP:-malnet_arch_ablation_9h3jqzkm}"

job_id="$(
    sbatch --parsable \
        --job-name=malnet_9h3jqzkm_arch \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/malnet_9h3jqzkm_arch_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,ARCH_ABLATION_NUM_VARIANTS="${NUM_VARIANTS}",ARCH_ABLATION_NUM_SEEDS="${NUM_SEEDS}",ARCH_ABLATION_NUM_TASKS="${NUM_TASKS}",ARCH_ABLATION_MAX_EPOCH="${MAX_EPOCH}",ARCH_ABLATION_MIN_LR="${MIN_LR}",ARCH_ABLATION_BASE_LR="${BASE_LR}",ARCH_ABLATION_WANDB_GROUP="${WANDB_GROUP_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_malnet_hybrid_9h3jqzkm_arch_ablation.sh
)"

echo ""
echo "=== MalNet 9h3jqzkm architecture ablation submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds = ${NUM_TASKS}), parallel=${PARALLEL}"
echo "  base_lr:      ${BASE_LR}  min_lr=${MIN_LR}  max_epoch=${MAX_EPOCH}"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  Config:       configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml"
echo "  Baseline:     https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u"
echo "  Logs:         logs_gnnplus/malnet_9h3jqzkm_arch_${job_id}_<TASK>.log"
echo ""
echo "  Variant 0 (tasks 1-2): a5g5    attn=5   gnn=5   → ${WANDB_GROUP_PREFIX}_a5g5_b40"
echo "  Variant 1 (tasks 3-4): a0g10   attn=0   gnn=10  → ${WANDB_GROUP_PREFIX}_a0g10_b40"
echo "  Variant 2 (tasks 5-6): a0g50   attn=0   gnn=50  → ${WANDB_GROUP_PREFIX}_a0g50_b40"
echo "  Variant 3 (tasks 7-8): a50g50  attn=50  gnn=50  → ${WANDB_GROUP_PREFIX}_a50g50_b40"
echo ""
echo "Aggregate per variant when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP_PREFIX}_a5g5_b40 --metric best_test_perf"
echo ""
