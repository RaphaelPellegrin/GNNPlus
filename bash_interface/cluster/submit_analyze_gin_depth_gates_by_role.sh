#!/usr/bin/env bash
# Submit role×layer×τ gate analysis for GIN depth-routing.
#
#   bash bash_interface/cluster/submit_analyze_gin_depth_gates_by_role.sh

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

results_root="${GIN_DEPTH_ROLE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
out_dir="${GIN_DEPTH_ROLE_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
lr_tag="${GIN_DEPTH_ROLE_LR_TAG:-lr001}"
tracks="${GIN_DEPTH_ROLE_TRACKS:-toy}"

chmod +x bash_interface/cluster/run_analyze_gin_depth_gates_by_role.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GIN_DEPTH_ROLE_RESULTS_ROOT=${results_root}"
export_list+=",GIN_DEPTH_ROLE_OUT_DIR=${out_dir}"
export_list+=",GIN_DEPTH_ROLE_LR_TAG=${lr_tag}"
export_list+=",GIN_DEPTH_ROLE_TRACKS=${tracks}"

job_id="$(
  sbatch --parsable \
    --job-name=gin_depth_role_g \
    --partition="${GIN_DEPTH_ROLE_PARTITION:-mweber_gpu}" \
    --mem="${GIN_DEPTH_ROLE_MEM:-32GB}" \
    --time="${GIN_DEPTH_ROLE_TIME:-01:30:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gin_depth_role_g_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_analyze_gin_depth_gates_by_role.sh
)"

cat <<EOF

=== GIN depth gates-by-role submitted ===
  JOBID:  ${job_id}
  Out:    ${out_dir}/fig_gates_by_role_layer_tau.png
  Log:    logs_gnnplus/gin_depth_role_g_${job_id}.log

EOF
