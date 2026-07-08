#!/usr/bin/env bash
# Submit peptides-func Level 1 (custom_gnn_gated) × 10 seeds + 2 LR spot checks.
#
#   Tasks 1–10: seeds 0–9 @ lr=1e-3
#   Task 11:    seed 0 @ lr=1.25e-3  (1.25×)
#   Task 12:    seed 0 @ lr=7.5e-4   (0.75×)
#
# Defaults: max 2 concurrent GPUs, --nice=10000 (lab-friendly).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_level1_seed_lr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${LEVEL1_SWEEP_NUM_TASKS:-12}"
ARRAY_SPEC="${LEVEL1_SWEEP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${LEVEL1_SWEEP_PARALLEL:-2}"
NICE="${LEVEL1_SWEEP_NICE:-10000}"
MEM="${LEVEL1_SWEEP_MEM:-64GB}"
TIME="${LEVEL1_SWEEP_TIME:-120:00:00}"
WANDB_GROUP="${LEVEL1_SWEEP_WANDB_GROUP:-peptides_func_level1_seed_lr}"

sbatch_args=(
    --parsable
    --job-name=peptides_l1_sweep
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/peptides_l1_sweep_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,LEVEL1_SWEEP_NUM_TASKS="${NUM_TASKS}",LEVEL1_SWEEP_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_level1_seed_lr.sh
)"

echo ""
echo "=== Peptides-func Level 1 seed/LR sweep submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (10 seeds + 2 LR ablations), parallel=${PARALLEL}"
echo "  SLURM nice:   ${NICE}"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  Config:       configs/gcn/peptides-func-gated.yaml (custom_gnn_gated)"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/peptides_l1_sweep_${job_id}_<TASK>.log"
echo ""
echo "  Tasks 1–10:  seeds 0–9 @ lr=1e-3"
echo "  Task 11:     seed 0 @ lr=1.25e-3"
echo "  Task 12:     seed 0 @ lr=7.5e-4"
echo ""
echo "Aggregate base-lr seeds:"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag lr_base"
echo ""
