#!/usr/bin/env bash
# Submit peptides-func fairness ladder: 5 levels × 3 seeds = 15 jobs.
#
# Defaults: max 5 concurrent GPUs, --nice=10000 so other lab users get priority
# (FASRC: higher nice = lower scheduling priority; override with NICE=0).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_func_fairness_ladder.sh
#
# Override concurrency / priority:
#   FAIRNESS_LADDER_PARALLEL=3 FAIRNESS_LADDER_NICE=20000 bash .../submit_peptides_func_fairness_ladder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${FAIRNESS_LADDER_NUM_TASKS:-15}"
ARRAY_SPEC="${FAIRNESS_LADDER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${FAIRNESS_LADDER_PARALLEL:-5}"
NICE="${FAIRNESS_LADDER_NICE:-10000}"
MEM="${PEPTIDES_FUNC_LADDER_MEM:-64GB}"
TIME="${PEPTIDES_FUNC_LADDER_TIME:-240:00:00}"
WANDB_GROUP="${FAIRNESS_LADDER_WANDB_GROUP:-peptides_func_fairness_ladder}"

sbatch_args=(
    --parsable
    --job-name=peptides_func_ladder
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/peptides_func_ladder_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,FAIRNESS_LADDER_NUM_TASKS="${NUM_TASKS}",FAIRNESS_LADDER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_fairness_ladder.sh
)"

echo ""
echo "=== Peptides-func fairness ladder submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (5 levels × 3 seeds), parallel=${PARALLEL}"
echo "  SLURM nice:   ${NICE} (0 = normal priority; higher = lower priority)"
echo "  Time limit:   ${TIME}  mem=${MEM}"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/peptides_func_ladder_${job_id}_<TASK>.log"
echo ""
echo "  Level 0 (tasks 1–3):   baseline GCNE @ 275, seeds 0–2"
echo "  Level 1 (tasks 4–6):   GCNE + headwise gate @ 275"
echo "  Level 2 (tasks 7–9):   hybrid a0g1 @ d_h=275"
echo "  Level 3 (tasks 10–12): zc371e1n paper anchor (1200 ep)"
echo "  Level 4 (tasks 13–15):  hybrid a1g1 @ d_h=275"
echo ""
echo "  W&B tags: level_0 … level_4 (+ fairness_ladder)"
echo ""
echo "Aggregate when done (per level):"
echo "  python scripts/api_wanndb_query/aggregate_paper_repro.py \\"
echo "    --group ${WANDB_GROUP} --metric best_test_perf --tag level_2"
echo ""
