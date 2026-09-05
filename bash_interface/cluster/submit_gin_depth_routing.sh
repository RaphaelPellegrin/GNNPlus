#!/usr/bin/env bash
# Submit GIN depth-routing synthetic (2-layer SiGMA gated vs ungated).
#
# Local machine first (git), then cluster:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=...   # parent that will contain GinDepthRouting/
#   export GNNPLUS_OUT_DIR=...
#   cd .../GNNPlus && git pull
#   bash bash_interface/cluster/submit_gin_depth_routing.sh
#
# Optional:
#   GIN_DEPTH_ROUTING_NUM_SEEDS=5
#   GIN_DEPTH_ROUTING_NUM_LRS=2
#   GIN_DEPTH_ROUTING_PARALLEL=8
#   GIN_DEPTH_DATASET_DIR=$PWD/results/gin_routing_depth/data   # local override

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

NUM_SEEDS="${GIN_DEPTH_ROUTING_NUM_SEEDS:-5}"
NUM_LRS="${GIN_DEPTH_ROUTING_NUM_LRS:-2}"
NUM_MODELS=2
NUM_TASKS=$((NUM_MODELS * NUM_LRS * NUM_SEEDS))
ARRAY_SPEC="${GIN_DEPTH_ROUTING_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${GIN_DEPTH_ROUTING_PARALLEL:-8}"
PARTITION="${GIN_DEPTH_ROUTING_PARTITION:-mweber_gpu}"
MEM="${GIN_DEPTH_ROUTING_MEM:-32GB}"
TIME="${GIN_DEPTH_ROUTING_TIME:-04:00:00}"
NICE="${GIN_DEPTH_ROUTING_NICE:-10000}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
  echo "[submit_gin_depth_routing] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
  echo "[submit_gin_depth_routing] GNNPLUS_DATASET_DIR unset → ${GNNPLUS_DATASET_DIR}"
fi

dataset_parent="${GIN_DEPTH_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"
dataset_root="${dataset_parent}/GinDepthRouting"
if [ ! -f "${dataset_root}/processed/train.pt" ]; then
  echo "[submit_gin_depth_routing] Generating dataset at ${dataset_root}..."
  python scripts/synthetic/generate_gin_depth_routing_dataset.py \
    --root "${dataset_root}" \
    --train "${GIN_DEPTH_TRAIN:-10000}" \
    --val "${GIN_DEPTH_VAL:-2000}" \
    --test "${GIN_DEPTH_TEST:-2000}"
fi

chmod +x bash_interface/cluster/run_gin_depth_routing.sh

sbatch_args=(
  --parsable
  --job-name=gin_depth_rt
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --gpus=1
  --output="logs_gnnplus/gin_depth_rt_%A_%a.log"
  --export=ALL,ENV_NAME=gnnplus,GIN_DEPTH_ROUTING_NUM_SEEDS="${NUM_SEEDS}",GIN_DEPTH_ROUTING_NUM_LRS="${NUM_LRS}",GIN_DEPTH_DATASET_DIR="${dataset_parent}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)
if [ "${NICE}" != "0" ]; then
  sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
  sbatch "${sbatch_args[@]}" \
    bash_interface/cluster/run_gin_depth_routing.sh
)"

cat <<EOF

=== GIN depth-routing submitted ===
  JOBID:     ${job_id}
  Tasks:     ${ARRAY_SPEC}  (2 models × ${NUM_LRS} lr × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Models:    l2_a0g1_gated · l2_a0g1_ungated  (always layers_mp=2)
  Dataset:   ${dataset_root}
  Out:       \$GNNPLUS_OUT_DIR/gin_routing_depth/toy/<model>_<lr>_seed<s>/
  W&B gates: gates_by_tau_depth/{val,test}/layer{k}/tau{0,1}/mean_gamma
  Logs:      logs_gnnplus/gin_depth_rt_${job_id}_<TASK>.log

Paste JOBID into CLUSTER_LAUNCHES.md / results/gin_routing_depth/README.md

EOF
