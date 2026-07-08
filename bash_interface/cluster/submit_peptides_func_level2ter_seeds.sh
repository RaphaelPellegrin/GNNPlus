#!/usr/bin/env bash
# Submit peptides-func Level 2ter (SiGMA hybrid collapsed ≈ Level 1) × 10 seeds.
#
# identity_proj + no LN/res: tests whether dropping hybrid projections closes the gap.
#
# Defaults: max 5 concurrent GPUs, --nice=10000.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_level2ter_seeds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${LEVEL2TER_NUM_TASKS:-10}"
ARRAY_SPEC="${LEVEL2TER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${LEVEL2TER_PARALLEL:-5}"
NICE="${LEVEL2TER_NICE:-10000}"
MEM="${LEVEL2TER_MEM:-64GB}"
TIME="${LEVEL2TER_TIME:-120:00:00}"
WANDB_GROUP="${LEVEL2TER_WANDB_GROUP:-peptides_func_level2ter_seeds}"

sbatch_args=(
    --parsable
    --job-name=peptides_l2ter
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/peptides_l2ter_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,LEVEL2TER_NUM_TASKS="${NUM_TASKS}",LEVEL2TER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_level2ter_seeds.sh
)"

echo ""
echo "=== Peptides-func Level 2ter (identity_proj ≈ Level 1) × 10 seeds ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds 0–9), parallel=${PARALLEL}"
echo "  SLURM nice:   ${NICE}"
echo "  Config:       configs/gated_hybrid/peptides-func-gcn-repro-a0g1-identity-proj.yaml"
echo "  model.type:   hybrid_gnn (a0g1, identity_proj, d_h=275, no LN/res)"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/peptides_l2ter_${job_id}_<TASK>.log"
echo ""
echo "Aggregate:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag level_2ter"
echo ""
