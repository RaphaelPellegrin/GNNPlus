#!/usr/bin/env bash
# Submit GCN/GIN routing synthetic benchmark (toy and/or sigma track).
#
# Usage:
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh toy
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh sigma
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh both
#
# On cluster: source ~/.gnnplus_env; export GNNPLUS_*; git pull; then:
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh both
# Prep uses common_env.sh (gnnplus conda, PYTHONNOUSERSITE=1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

# Use lab gnnplus env (not login-node ~/.local python) for config/dataset prep.
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

track_arg="${1:-both}"
NUM_SEEDS="${GCN_GIN_ROUTING_NUM_SEEDS:-5}"
NUM_LRS="${GCN_GIN_ROUTING_NUM_LRS:-2}"
NUM_MODELS=4
NUM_TASKS=$((NUM_MODELS * NUM_LRS * NUM_SEEDS))
ARRAY_SPEC="${GCN_GIN_ROUTING_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${GCN_GIN_ROUTING_PARALLEL:-10}"
PARTITION="${GCN_GIN_ROUTING_PARTITION:-mweber_gpu}"
MEM="${GCN_GIN_ROUTING_MEM:-32GB}"
TIME="${GCN_GIN_ROUTING_TIME:-04:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
  echo "[submit_gcn_gin_routing] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
  echo "[submit_gcn_gin_routing] GNNPLUS_DATASET_DIR unset → ${GNNPLUS_DATASET_DIR}"
fi

python scripts/synthetic/generate_gcn_gin_routing_configs.py

dataset_root="${GNNPLUS_DATASET_DIR}/GcnGinRouting"
if [ ! -f "${dataset_root}/processed/train.pt" ]; then
  if [ "${GCN_GIN_ROUTING_SKIP_DATA:-0}" = "1" ]; then
    echo "[submit_gcn_gin_routing] Dataset missing at ${dataset_root} (GCN_GIN_ROUTING_SKIP_DATA=1)"
    exit 1
  fi
  echo "[submit_gcn_gin_routing] Generating dataset at ${dataset_root} (~1 min CPU)..."
  python scripts/synthetic/generate_gcn_gin_routing_dataset.py \
    --root "${dataset_root}"
fi

missing=0
for track in toy sigma; do
  for slug in a0g2_gated a0g2_ungated a0g1_gcn a0g1_gin; do
    cfg="configs/synthetic/gcn_gin_routing_${track}_${slug}.yaml"
    if [ ! -f "${cfg}" ]; then
      echo "MISSING ${cfg}"
      missing=1
    fi
  done
done
if [ "${missing}" -ne 0 ]; then
  exit 1
fi

chmod +x bash_interface/cluster/run_gcn_gin_routing.sh

_submit_one() {
  local track="$1"
  local job_name="gcn_gin_route_${track}"
  local job_id
  job_id="$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --array="${ARRAY_SPEC}%${PARALLEL}" \
      --partition="${PARTITION}" \
      --mem="${MEM}" \
      --time="${TIME}" \
      --gpus=1 \
      --output="logs_gnnplus/${job_name}_%A_%a.log" \
      --export=ALL,ENV_NAME=gnnplus,GCN_GIN_ROUTING_TRACK="${track}",GCN_GIN_ROUTING_NUM_SEEDS="${NUM_SEEDS}",GCN_GIN_ROUTING_NUM_LRS="${NUM_LRS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}" \
      bash_interface/cluster/run_gcn_gin_routing.sh
  )"
  cat <<EOF

=== GCN/GIN routing TRACK=${track} submitted ===
  JOBID:     ${job_id}
  Tasks:     ${ARRAY_SPEC} (${NUM_MODELS} models × ${NUM_LRS} LRs × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:  ${PARALLEL}
  Logs:      logs_gnnplus/${job_name}_${job_id}_<TASK>.log
  Out:       \$GNNPLUS_OUT_DIR/gcn_gin_routing/${track}/<model>_<lr>_seed<s>/
  W&B tag:   gcn_gin_routing_synthetic

EOF
}

case "${track_arg}" in
  toy) _submit_one toy ;;
  sigma) _submit_one sigma ;;
  both) _submit_one toy; _submit_one sigma ;;
  *)
    echo "Usage: $0 {toy|sigma|both}"
    exit 1
    ;;
esac
