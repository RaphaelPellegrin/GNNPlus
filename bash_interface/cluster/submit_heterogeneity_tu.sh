#!/usr/bin/env bash
# Launch TU heterogeneity profiles: 3 datasets × {GCN,GIN,SiGMA} = 9 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch (≤5 GPUs by default, mweber_gpu):
#   bash bash_interface/cluster/submit_heterogeneity_tu.sh
#
# Cap parallelism / pick partition:
#   HETERO_PARALLEL=5 HETERO_PARTITION=gpu_h200 \
#     bash bash_interface/cluster/submit_heterogeneity_tu.sh
#
# Smoke (2 appearances):
#   HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=20 \
#     bash bash_interface/cluster/submit_heterogeneity_tu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${HETERO_NUM_TASKS:-9}"
ARRAY_SPEC="${HETERO_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${HETERO_PARALLEL:-5}"
PARTITION="${HETERO_PARTITION:-mweber_gpu}"
NICE="${HETERO_NICE:-10000}"
MEM="${HETERO_MEM:-64GB}"
TIME="${HETERO_TIME:-192:00:00}"
REQUIRED="${HETERO_REQUIRED_TEST_APPEARANCES:-100}"
MAX_TRIALS="${HETERO_MAX_TRIALS:-2000}"
SEED0="${HETERO_SEED0:-0}"
WANDB_USE="${HETERO_WANDB:-1}"

sbatch_args=(
    --parsable
    --job-name=hetero_tu
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/hetero_tu_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,HETERO_REQUIRED_TEST_APPEARANCES="${REQUIRED}",HETERO_MAX_TRIALS="${MAX_TRIALS}",HETERO_SEED0="${SEED0}",HETERO_WANDB="${WANDB_USE}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_heterogeneity_tu.sh
)"

cat <<EOF

=== Heterogeneity TU profiles submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (mutag/enzymes/proteins × gcn/gin/sigma)
  Parallel:      ${PARALLEL} GPUs max
  Appearances:   ≥${REQUIRED} per graph
  Max trials:    ${MAX_TRIALS}
  W&B:           ${WANDB_USE}  (groups: building_hetero_profile_<dataset>)
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/hetero_tu_${job_id}_<TASK>.log
  Local outs:    results/heterogeneity/<dataset>_<MODEL>/
                 (+ pickle, test_appearances.csv, profile PNGs → W&B artifact)

  Paste JOBID into Paper_heterogeneity.md + CLUSTER_LAUNCHES.md

EOF
