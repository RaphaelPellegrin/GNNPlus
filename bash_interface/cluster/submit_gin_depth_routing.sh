#!/usr/bin/env bash
# Submit GIN depth-routing (Track A=toy / Track B=sigma), like gcn_gin_routing.
#
# Usage:
#   bash bash_interface/cluster/submit_gin_depth_routing.sh toy
#   bash bash_interface/cluster/submit_gin_depth_routing.sh sigma
#   bash bash_interface/cluster/submit_gin_depth_routing.sh both
#
# Models per track (40 tasks = 4 × 2 lr × 5 seeds):
#   1–10:  l2_a0g1_gated
#  11–20:  l2_a0g1_ungated
#  21–30:  l1_a0g1          (1-GIN specialist)
#  31–40:  l2_a0g1_gin      (2-GIN specialist)
#
# Track A already ran gated+ungated (tasks 1–20). Specialists fill:
#   GIN_DEPTH_ROUTING_ARRAY=21-40 bash .../submit_gin_depth_routing.sh toy
#
# Track B (full):
#   bash .../submit_gin_depth_routing.sh sigma

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

track_arg="${1:-both}"
NUM_SEEDS="${GIN_DEPTH_ROUTING_NUM_SEEDS:-5}"
NUM_LRS="${GIN_DEPTH_ROUTING_NUM_LRS:-2}"
NUM_MODELS=4
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

missing=0
for track in toy sigma; do
  for slug in l2_a0g1_gated l2_a0g1_ungated l1_a0g1 l2_a0g1_gin; do
    cfg="configs/synthetic/gin_depth_routing_${track}_${slug}.yaml"
    if [ ! -f "${cfg}" ]; then
      echo "MISSING ${cfg}"
      missing=1
    fi
  done
done
if [ "${missing}" -ne 0 ]; then
  exit 1
fi

chmod +x bash_interface/cluster/run_gin_depth_routing.sh

_submit_one() {
  local track="$1"
  local job_name="gin_depth_rt_${track}"
  local export_list
  export_list="ALL,ENV_NAME=gnnplus"
  export_list+=",GIN_DEPTH_ROUTING_TRACK=${track}"
  export_list+=",GIN_DEPTH_ROUTING_NUM_SEEDS=${NUM_SEEDS}"
  export_list+=",GIN_DEPTH_ROUTING_NUM_LRS=${NUM_LRS}"
  export_list+=",GIN_DEPTH_DATASET_DIR=${dataset_parent}"
  export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
  export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"

  local sbatch_args=(
    --parsable
    --job-name="${job_name}"
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/${job_name}_%A_%a.log"
    --export="${export_list}"
  )
  if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
  fi

  local job_id
  job_id="$(
    sbatch "${sbatch_args[@]}" \
      bash_interface/cluster/run_gin_depth_routing.sh
  )"

  local track_label="A (toy, d_h=1, ROUTING_SUM)"
  if [[ "${track}" == "sigma" ]]; then
    track_label="B (sigma, d_h=4, GIN)"
  fi

  cat <<EOF

=== GIN depth-routing TRACK=${track} [${track_label}] submitted ===
  JOBID:     ${job_id}
  Tasks:     ${ARRAY_SPEC}  (4 models × ${NUM_LRS} lr × ${NUM_SEEDS} seeds = ${NUM_TASKS} full)
  Models:    l2_a0g1_gated · l2_a0g1_ungated · l1_a0g1 · l2_a0g1_gin
  Dataset:   ${dataset_root}
  Out:       \$GNNPLUS_OUT_DIR/gin_routing_depth/${track}/<model>_<lr>_seed<s>/
  Logs:      logs_gnnplus/${job_name}_${job_id}_<TASK>.log
  Array map: 1–10 gated · 11–20 ungated · 21–30 1-GIN · 31–40 2-GIN

EOF
}

case "${track_arg}" in
  toy|sigma)
    _submit_one "${track_arg}"
    ;;
  both)
    _submit_one toy
    _submit_one sigma
    ;;
  *)
    echo "Usage: $0 [toy|sigma|both]" >&2
    exit 1
    ;;
esac
