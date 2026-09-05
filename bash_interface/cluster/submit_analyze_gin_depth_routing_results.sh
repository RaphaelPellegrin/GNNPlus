#!/usr/bin/env bash
# Submit GIN depth-routing analyze (per-τ acc + layer×τ gates).
#
#   bash bash_interface/cluster/submit_analyze_gin_depth_routing_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

results_root="${GIN_DEPTH_ANALYZE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
out_dir="${GIN_DEPTH_ANALYZE_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
tracks="${GIN_DEPTH_ANALYZE_TRACKS:-toy}"

chmod +x bash_interface/cluster/run_analyze_gin_depth_routing_results.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GIN_DEPTH_ANALYZE_RESULTS_ROOT=${results_root}"
export_list+=",GIN_DEPTH_ANALYZE_OUT_DIR=${out_dir}"
export_list+=",GIN_DEPTH_ANALYZE_TRACKS=${tracks}"
if [ -n "${GIN_DEPTH_ANALYZE_LR_TAG:-}" ]; then
  export_list+=",GIN_DEPTH_ANALYZE_LR_TAG=${GIN_DEPTH_ANALYZE_LR_TAG}"
fi

job_id="$(
  sbatch --parsable \
    --job-name=gin_depth_analyze \
    --partition="${GIN_DEPTH_ANALYZE_PARTITION:-mweber_gpu}" \
    --mem="${GIN_DEPTH_ANALYZE_MEM:-32GB}" \
    --time="${GIN_DEPTH_ANALYZE_TIME:-02:00:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gin_depth_analyze_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_analyze_gin_depth_routing_results.sh
)"

cat <<EOF

=== GIN depth-routing analyze submitted ===
  JOBID:  ${job_id}
  Root:   ${results_root}
  Out:    ${out_dir}
  Log:    logs_gnnplus/gin_depth_analyze_${job_id}.log

Append to CLUSTER_LAUNCHES.md:
  | $(date +%Y-%m-%d) | gin_depth_analyze | ${job_id} | per-τ acc + layer×τ gates |

EOF
