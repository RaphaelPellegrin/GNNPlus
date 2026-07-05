#!/usr/bin/env bash
# Submit PATTERN hybrid LR ablation: ta9qtxb9 a2g2 × 5 LR × 10 seeds (50 jobs).
#
# Baseline run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   LR_SWEEP_PARALLEL=10 bash bash_interface/cluster/submit_pattern_hybrid_ta9qtxb9_lr_sweep.sh

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
MEM="${PATTERN_LR_SWEEP_MEM:-128GB}"
TIME="${PATTERN_LR_SWEEP_TIME:-96:00:00}"
WANDB_GROUP_PREFIX="${LR_SWEEP_WANDB_GROUP:-lr_ablation_pattern_hybrid_ta9qtxb9_a2g2}"

job_id="$(
    sbatch --parsable \
        --job-name=pattern_ta9qtxb9_lr \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition=mweber_gpu \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/pattern_ta9qtxb9_lr_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,LR_SWEEP_NUM_LR="${NUM_LR}",LR_SWEEP_NUM_SEEDS="${NUM_SEEDS}",LR_SWEEP_NUM_TASKS="${NUM_TASKS}",LR_SWEEP_WANDB_GROUP="${WANDB_GROUP_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_pattern_hybrid_ta9qtxb9_lr_sweep.sh
)"

echo ""
echo "=== PATTERN ta9qtxb9 LR ablation submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_LR} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS}), parallel=${PARALLEL}"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  Config:       configs/gated_hybrid/pattern-gcne-best-hybrid.yaml"
echo "  Baseline:     https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9"
echo "  Logs:         logs_gnnplus/pattern_ta9qtxb9_lr_${job_id}_<TASK>.log"
echo ""
echo "  LR 0 (tasks  1-10): base_lr=2.217392e-3 (exact)  → ${WANDB_GROUP_PREFIX}_b2217_m0"
echo "  LR 1 (tasks 11-20): base_lr=2.4e-3               → ${WANDB_GROUP_PREFIX}_b24_m0"
echo "  LR 2 (tasks 21-30): base_lr=2.6e-3               → ${WANDB_GROUP_PREFIX}_b26_m0"
echo "  LR 3 (tasks 31-40): base_lr=2.8e-3               → ${WANDB_GROUP_PREFIX}_b28_m0"
echo "  LR 4 (tasks 41-50): base_lr=3.0e-3               → ${WANDB_GROUP_PREFIX}_b30_m0"
echo ""
echo "Aggregate per LR when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP_PREFIX}_b2217_m0 \\"
echo "    --metric best_test_perf"
echo ""
