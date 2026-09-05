#!/usr/bin/env bash
# Submit the full GIN depth-routing analysis pack (one GPU job).
#
#   bash bash_interface/cluster/submit_gin_depth_routing_full_analysis.sh

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
lr_tag="${GIN_DEPTH_ANALYZE_LR_TAG:-lr001}"

chmod +x bash_interface/cluster/run_gin_depth_routing_full_analysis.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GIN_DEPTH_ANALYZE_RESULTS_ROOT=${results_root}"
export_list+=",GIN_DEPTH_ANALYZE_OUT_DIR=${out_dir}"
export_list+=",GIN_DEPTH_ANALYZE_LR_TAG=${lr_tag}"

job_id="$(
  sbatch --parsable \
    --job-name=gin_depth_full_an \
    --partition="${GIN_DEPTH_FULL_PARTITION:-mweber_gpu}" \
    --mem="${GIN_DEPTH_FULL_MEM:-32GB}" \
    --time="${GIN_DEPTH_FULL_TIME:-04:00:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gin_depth_full_an_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_gin_depth_routing_full_analysis.sh
)"

cat <<EOF

=== GIN depth FULL analysis submitted ===
  JOBID:  ${job_id}
  Root:   ${results_root}
  Out:    ${out_dir}
  Log:    logs_gnnplus/gin_depth_full_an_${job_id}.log

After COMPLETED, pull:
  bash bash_interface/local/pull_gin_depth_routing_results.sh --bundle

Append to CLUSTER_LAUNCHES.md:
  | $(date +%Y-%m-%d) | gin_depth_full_an | ${job_id} | full gcn_gin-style analysis for depth |

EOF
