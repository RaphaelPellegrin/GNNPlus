#!/usr/bin/env bash
# Submit peptides-func Level 2bis (hybrid a0g1, no LN, no residual) × 10 seeds.
#
# Still hybrid_gnn (proj → gated GCNE MP → out_proj), without pre-norm / block skip.
#
# Defaults: max 2 concurrent GPUs, --nice=10000.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_level2bis_seeds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${LEVEL2BIS_NUM_TASKS:-10}"
ARRAY_SPEC="${LEVEL2BIS_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${LEVEL2BIS_PARALLEL:-2}"
NICE="${LEVEL2BIS_NICE:-10000}"
MEM="${LEVEL2BIS_MEM:-64GB}"
TIME="${LEVEL2BIS_TIME:-120:00:00}"
WANDB_GROUP="${LEVEL2BIS_WANDB_GROUP:-peptides_func_level2bis_seeds}"

sbatch_args=(
    --parsable
    --job-name=peptides_l2bis
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/peptides_l2bis_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,LEVEL2BIS_NUM_TASKS="${NUM_TASKS}",LEVEL2BIS_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_level2bis_seeds.sh
)"

echo ""
echo "=== Peptides-func Level 2bis (no LN / no residual) × 10 seeds submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds 0–9), parallel=${PARALLEL}"
echo "  SLURM nice:   ${NICE}"
echo "  Config:       configs/gated_hybrid/peptides-func-gcn-repro-a0g1-noln-nores.yaml"
echo "  model.type:   hybrid_gnn (a0g1, d_h=275, norm=none, residual=false)"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/peptides_l2bis_${job_id}_<TASK>.log"
echo ""
echo "Aggregate:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag level_2bis"
echo ""
