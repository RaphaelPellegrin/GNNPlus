#!/usr/bin/env bash
# Submit peptides-func UniGCN hybrid: 124caj93 a2g3 × 10 seeds (max 2 GPUs).
#
# Source run: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/124caj93
# Sweep: https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_hybrid_124caj93_a2g3_seeds.sh
#
# Optional: PF_124_PARALLEL=2 PF_124_TIME=120:00:00 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PF_124_NUM_SEEDS:-10}"
ARRAY_SPEC="${PF_124_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PF_124_PARALLEL:-2}"
NICE="${PF_124_NICE:-10000}"
MEM="${PF_124_MEM:-64GB}"
TIME="${PF_124_TIME:-120:00:00}"
WANDB_GROUP="${PF_124_WANDB_GROUP:-peptides_func_124caj93_a2g3_seeds}"

sbatch_args=(
    --parsable
    --job-name=pf_124caj93_seeds
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/pf_124caj93_seeds_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PF_124_NUM_SEEDS="${NUM_SEEDS}",PF_124_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_hybrid_124caj93_a2g3_seeds.sh
)"

echo ""
echo "=== peptides-func 124caj93 a2g3 × 10 seeds submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds 0..$((NUM_SEEDS - 1))), parallel=${PARALLEL}"
echo "  Config:       configs/gated_hybrid/peptides-func-hybrid-124caj93-a2g3-unigcn-anchor.yaml"
echo "  Source run:   https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/124caj93"
echo "  Sweep:        https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz"
echo "  Logs:         logs_gnnplus/pf_124caj93_seeds_${job_id}_<TASK>.log"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  LR (exact):   0.0005650212198206989"
echo "  Arch:         a2g3 GINE,GCNE,UNIGCN | d_h=128 | T=16 | headwise | LN | full | ep=300"
echo ""
echo "Aggregate when done:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf"
echo ""
