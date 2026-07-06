#!/usr/bin/env bash
# Submit MalNet-Tiny hybrid LR ablation: 9h3jqzkm a0g2 × 5 LR × 10 seeds (50 jobs).
#
# Baseline run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   LR_SWEEP_PARALLEL=10 bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_lr_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_LR="${LR_SWEEP_NUM_LR:-5}"
NUM_SEEDS="${LR_SWEEP_NUM_SEEDS:-10}"
NUM_TASKS="${LR_SWEEP_NUM_TASKS:-$((NUM_LR * NUM_SEEDS))}"
ARRAY_SPEC="${LR_SWEEP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${LR_SWEEP_PARALLEL:-10}"
MEM="${MALNET_LR_SWEEP_MEM:-64GB}"
TIME="${MALNET_LR_SWEEP_TIME:-96:00:00}"
MAX_EPOCH="${LR_SWEEP_MAX_EPOCH:-250}"
MIN_LR="${LR_SWEEP_MIN_LR:-1e-6}"
WANDB_GROUP_PREFIX="${LR_SWEEP_WANDB_GROUP:-lr_ablation_malnet_hybrid_9h3jqzkm_a0g2}"

job_id="$(
    sbatch --parsable \
        --job-name=malnet_9h3jqzkm_lr \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/malnet_9h3jqzkm_lr_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,LR_SWEEP_NUM_LR="${NUM_LR}",LR_SWEEP_NUM_SEEDS="${NUM_SEEDS}",LR_SWEEP_NUM_TASKS="${NUM_TASKS}",LR_SWEEP_MAX_EPOCH="${MAX_EPOCH}",LR_SWEEP_MIN_LR="${MIN_LR}",LR_SWEEP_WANDB_GROUP="${WANDB_GROUP_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_malnet_hybrid_9h3jqzkm_lr_sweep.sh
)"

echo ""
echo "=== MalNet 9h3jqzkm LR ablation submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_LR} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS}), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME}  mem=${MEM}  max_epoch=${MAX_EPOCH}  min_lr=${MIN_LR}"
echo "  Config:       configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml"
echo "  Baseline:     https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u"
echo "  Logs:         logs_gnnplus/malnet_9h3jqzkm_lr_${job_id}_<TASK>.log"
echo ""
echo "  LR 0 (tasks  1-10): base_lr=1.914236e-3 (exact)  → ${WANDB_GROUP_PREFIX}_b1914_m1e6"
echo "  LR 1 (tasks 11-20): base_lr=1.7e-3                → ${WANDB_GROUP_PREFIX}_b17_m1e6"
echo "  LR 2 (tasks 21-30): base_lr=2.1e-3                → ${WANDB_GROUP_PREFIX}_b21_m1e6"
echo "  LR 3 (tasks 31-40): base_lr=2.3e-3                → ${WANDB_GROUP_PREFIX}_b23_m1e6"
echo "  LR 4 (tasks 41-50): base_lr=2.5e-3                → ${WANDB_GROUP_PREFIX}_b25_m1e6"
echo ""
echo "Aggregate per LR when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP_PREFIX}_b1914_m1e6 \\"
echo "    --metric best_test_perf"
echo ""
