#!/usr/bin/env bash
# Submit opposite-sign τ pair analysis for GIN depth-routing.
#
#   bash bash_interface/cluster/submit_analyze_gin_depth_opposite_sign_pairs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

results_root="${GIN_DEPTH_OPP_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
out_dir="${GIN_DEPTH_OPP_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
lr_tag="${GIN_DEPTH_OPP_LR_TAG:-lr001}"

chmod +x bash_interface/cluster/run_analyze_gin_depth_opposite_sign_pairs.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GIN_DEPTH_OPP_RESULTS_ROOT=${results_root}"
export_list+=",GIN_DEPTH_OPP_OUT_DIR=${out_dir}"
export_list+=",GIN_DEPTH_OPP_LR_TAG=${lr_tag}"

job_id="$(
  sbatch --parsable \
    --job-name=gin_depth_opp \
    --partition="${GIN_DEPTH_OPP_PARTITION:-mweber_gpu}" \
    --mem="${GIN_DEPTH_OPP_MEM:-16GB}" \
    --time="${GIN_DEPTH_OPP_TIME:-01:00:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gin_depth_opp_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_analyze_gin_depth_opposite_sign_pairs.sh
)"

cat <<EOF

=== GIN depth opposite-sign pairs submitted ===
  JOBID:  ${job_id}
  Out:    ${out_dir}/opposite_sign_pair_summary.csv
  Log:    logs_gnnplus/gin_depth_opp_${job_id}.log

EOF
