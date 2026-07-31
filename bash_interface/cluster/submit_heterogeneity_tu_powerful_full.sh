#!/usr/bin/env bash
# Launch full TU hetero grid (Xu et al. HPs): 6 datasets × 4 models = 24 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_heterogeneity_tu_powerful_full.sh
#
# Smoke:
#   HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=20 HETERO_NUM_TASKS=2 \
#     bash bash_interface/cluster/submit_heterogeneity_tu_powerful_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${HETERO_NUM_TASKS:-24}"
ARRAY_SPEC="${HETERO_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${HETERO_PARALLEL:-8}"
PARTITION="${HETERO_PARTITION:-mweber_gpu}"
NICE="${HETERO_NICE:-10000}"
MEM="${HETERO_MEM:-64GB}"
if [ -n "${HETERO_TIME:-}" ]; then
    TIME="${HETERO_TIME}"
elif [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="72:00:00"
else
    TIME="192:00:00"
fi
REQUIRED="${HETERO_REQUIRED_TEST_APPEARANCES:-100}"
MAX_TRIALS="${HETERO_MAX_TRIALS:-2000}"
SEED0="${HETERO_SEED0:-0}"
WANDB_USE="${HETERO_WANDB:-1}"

sbatch_args=(
    --parsable
    --job-name=hetero_tu_full
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/hetero_tu_full_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,HETERO_REQUIRED_TEST_APPEARANCES="${REQUIRED}",HETERO_MAX_TRIALS="${MAX_TRIALS}",HETERO_SEED0="${SEED0}",HETERO_WANDB="${WANDB_USE}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_heterogeneity_tu_powerful_full.sh
)"

cat <<EOF

=== Heterogeneity · full TU (Xu HPs) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (6 ds × 4 models = 24)
  Parallel:      ≤${PARALLEL} GPUs
  Appearances:   ≥${REQUIRED} per graph
  Max trials:    ${MAX_TRIALS}
  Models:        GCN · GIN · SAGE · SiGMA(a2g2 GIN,GIN)
  Datasets:      MUTAG ENZYMES PROTEINS DD NCI1 TRIANGLES
  W&B groups:    building_hetero_profile_<ds>_powerful_gnns
  Configs:       configs/heterogeneity/powerful_gnns/
  Outs:          \$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/<ds>_<model>/
  Logs:          logs_gnnplus/hetero_tu_full_${job_id}_<TASK>.log
  Mem / time:    ${MEM} / ${TIME}

  Task map (model cycles fastest):
    1-4   mutag_{gcn,gin,sage,sigma}
    5-8   enzymes_…
    9-12  proteins_…
    13-16 dd_…
    17-20 nci1_…
    21-24 triangles_…

  Note: loader supports only these 6 TU names (not all of PyG TUDataset).
  NCI1/DD are large — expect long walltime for 100 appearances.
  Paste JOBID into Paper_heterogeneity.md + CLUSTER_LAUNCHES.md

EOF
