#!/usr/bin/env bash
# Submit ENZYMES ogpkubk9 a4g4 seed grids: plateau ×5 + cosine ×5 = 10 jobs.
#
# Source: https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/ogpkubk9
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_enzymes_ogpkubk9_seed_grids.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${ENZ_OGPK_NUM_SEEDS:-5}"
NUM_VARIANTS="${ENZ_OGPK_NUM_VARIANTS:-2}"
NUM_TASKS="${ENZ_OGPK_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
PARALLEL="${ENZ_OGPK_PARALLEL:-5}"
NICE="${ENZ_OGPK_NICE:-10000}"
MEM="${ENZ_OGPK_MEM:-64GB}"
TIME="${ENZ_OGPK_TIME:-96:00:00}"

sbatch_args=(
    --parsable
    --job-name=enz_ogpkubk9
    --array="1-${NUM_TASKS}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/enz_ogpkubk9_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,ENZ_OGPK_NUM_SEEDS="${NUM_SEEDS}",ENZ_OGPK_NUM_VARIANTS="${NUM_VARIANTS}",ENZ_OGPK_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_enzymes_ogpkubk9_seed_grids.sh
)"

cat <<EOF

=== ENZYMES ogpkubk9 a4g4 seed grids submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         1-${NUM_TASKS}%${PARALLEL}  (plateau×${NUM_SEEDS} + cosine×${NUM_SEEDS})
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/enz_ogpkubk9_${job_id}_<TASK>.log

  W&B groups:
    enzymes_ogpkubk9_a4g4_plateau_seeds
    enzymes_ogpkubk9_a4g4_cosine_seeds

  Source: https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/ogpkubk9

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group enzymes_ogpkubk9_a4g4_plateau_seeds --metric best_test_perf
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group enzymes_ogpkubk9_a4g4_cosine_seeds --metric best_test_perf

  Paste JOBID into Paper_enzymes_ogpkubk9.md

EOF
