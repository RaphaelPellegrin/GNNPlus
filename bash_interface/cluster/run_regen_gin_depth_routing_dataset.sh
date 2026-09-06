#!/usr/bin/env bash
# Force-regenerate GinDepthRouting with residual-faithful labels (R1/R2).
#
#   bash bash_interface/cluster/run_regen_gin_depth_routing_dataset.sh
#
# Then retrain (labels changed ⇒ all models):
#   bash bash_interface/cluster/submit_gin_depth_routing.sh toy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

dataset_parent="${GIN_DEPTH_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"
dataset_root="${dataset_parent}/GinDepthRouting"

echo "Regenerating ${dataset_root} with --force (residual_r2_v1)..."
python scripts/synthetic/generate_gin_depth_routing_dataset.py \
  --root "${dataset_root}" \
  --train "${GIN_DEPTH_TRAIN:-10000}" \
  --val "${GIN_DEPTH_VAL:-2000}" \
  --test "${GIN_DEPTH_TEST:-2000}" \
  --force

echo "Done. Next: bash bash_interface/cluster/submit_gin_depth_routing.sh toy"
