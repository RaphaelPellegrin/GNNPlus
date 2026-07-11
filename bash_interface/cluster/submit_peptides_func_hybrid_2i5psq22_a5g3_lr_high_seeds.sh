#!/usr/bin/env bash
# Submit peptides-func UniGCN hybrid: 2i5psq22 a5g3 × high LR × 10 seeds (30 jobs, %5).
#
# Anchor: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fpfl6ve9
# Config: configs/gated_hybrid/peptides-func-hybrid-2i5psq22-a5g3-unigcn-anchor.yaml
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_hybrid_2i5psq22_a5g3_lr_high_seeds.sh
#
# Optional: PF_2I5_HIGH_PARALLEL=5 PF_2I5_HIGH_TIME=120:00:00 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_LR="${PF_2I5_HIGH_NUM_LR:-3}"
NUM_SEEDS="${PF_2I5_HIGH_NUM_SEEDS:-10}"
NUM_TASKS="${PF_2I5_HIGH_NUM_TASKS:-$((NUM_LR * NUM_SEEDS))}"
ARRAY_SPEC="${PF_2I5_HIGH_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PF_2I5_HIGH_PARALLEL:-5}"
NICE="${PF_2I5_HIGH_NICE:-10000}"
MEM="${PF_2I5_HIGH_MEM:-64GB}"
TIME="${PF_2I5_HIGH_TIME:-120:00:00}"
WANDB_GROUP_PREFIX="${PF_2I5_HIGH_WANDB_GROUP:-peptides_func_2i5psq22_a5g3_lr_high}"

sbatch_args=(
    --parsable
    --job-name=pf_2i5_lrhigh
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/pf_2i5_lrhigh_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PF_2I5_HIGH_NUM_LR="${NUM_LR}",PF_2I5_HIGH_NUM_SEEDS="${NUM_SEEDS}",PF_2I5_HIGH_NUM_TASKS="${NUM_TASKS}",PF_2I5_HIGH_WANDB_GROUP="${WANDB_GROUP_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_hybrid_2i5psq22_a5g3_lr_high_seeds.sh
)"

echo ""
echo "=== peptides-func 2i5psq22 a5g3 high-LR×seed grid submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_LR} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS}), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-func-hybrid-2i5psq22-a5g3-unigcn-anchor.yaml"
echo "  Anchor run:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fpfl6ve9"
echo "  Logs:         logs_gnnplus/pf_2i5_lrhigh_${job_id}_<TASK>.log"
echo ""
echo "  LR 0 (tasks  1-10): base_lr=0.0005  → ${WANDB_GROUP_PREFIX}_b5"
echo "  LR 1 (tasks 11-20): base_lr=0.0007  → ${WANDB_GROUP_PREFIX}_b7"
echo "  LR 2 (tasks 21-30): base_lr=0.0009  → ${WANDB_GROUP_PREFIX}_b9"
echo ""
echo "Aggregate when done:"
echo "  for g in b5 b7 b9; do"
echo "    python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "      --group ${WANDB_GROUP_PREFIX}_\${g} --metric best_test_perf"
echo "  done"
echo ""
