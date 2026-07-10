#!/usr/bin/env bash
# Submit peptides-func UniGCN hybrid: 2i5psq22 a5g3 × 3 LR × 10 seeds (30 jobs).
#
# Source run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2i5psq22
# Sweep: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_hybrid_2i5psq22_a5g3_lr_seeds.sh
#
# Optional overrides:
#   PF_2I5_PARALLEL=2 PF_2I5_TIME=120:00:00 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_LR="${PF_2I5_NUM_LR:-3}"
NUM_SEEDS="${PF_2I5_NUM_SEEDS:-10}"
NUM_TASKS="${PF_2I5_NUM_TASKS:-$((NUM_LR * NUM_SEEDS))}"
ARRAY_SPEC="${PF_2I5_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PF_2I5_PARALLEL:-2}"
NICE="${PF_2I5_NICE:-10000}"
MEM="${PF_2I5_MEM:-64GB}"
TIME="${PF_2I5_TIME:-120:00:00}"
WANDB_GROUP_PREFIX="${PF_2I5_WANDB_GROUP:-peptides_func_2i5psq22_a5g3_lr_seeds}"

sbatch_args=(
    --parsable
    --job-name=pf_2i5psq22_lr
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/pf_2i5psq22_lr_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PF_2I5_NUM_LR="${NUM_LR}",PF_2I5_NUM_SEEDS="${NUM_SEEDS}",PF_2I5_NUM_TASKS="${NUM_TASKS}",PF_2I5_WANDB_GROUP="${WANDB_GROUP_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_hybrid_2i5psq22_a5g3_lr_seeds.sh
)"

echo ""
echo "=== peptides-func 2i5psq22 a5g3 LR×seed grid submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (${NUM_LR} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS}), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-func-hybrid-2i5psq22-a5g3-unigcn-anchor.yaml"
echo "  Source run:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2i5psq22"
echo "  Sweep:        https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz"
echo "  Logs:         logs_gnnplus/pf_2i5psq22_lr_${job_id}_<TASK>.log"
echo ""
echo "  LR 0 (tasks  1-10): base_lr=0.000455  → ${WANDB_GROUP_PREFIX}_b455"
echo "  LR 1 (tasks 11-20): base_lr=0.00045   → ${WANDB_GROUP_PREFIX}_b45"
echo "  LR 2 (tasks 21-30): base_lr=0.0005    → ${WANDB_GROUP_PREFIX}_b5"
echo ""
echo "  Arch: a5g3 GINE,GCNE,UNIGCN | d_h=128 | T=10 | headwise | LN | full mask | ep=300"
echo ""
