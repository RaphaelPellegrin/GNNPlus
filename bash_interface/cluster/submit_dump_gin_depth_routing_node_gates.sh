#!/usr/bin/env bash
# Submit gate dump for GIN depth-routing.
#
#   bash bash_interface/cluster/submit_dump_gin_depth_routing_node_gates.sh

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

results_root="${GIN_DEPTH_GDUMP_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"

chmod +x bash_interface/cluster/run_dump_gin_depth_routing_node_gates.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GIN_DEPTH_GDUMP_RESULTS_ROOT=${results_root}"

job_id="$(
  sbatch --parsable \
    --job-name=gin_depth_gdump \
    --partition="${GIN_DEPTH_GDUMP_PARTITION:-mweber_gpu}" \
    --mem="${GIN_DEPTH_GDUMP_MEM:-32GB}" \
    --time="${GIN_DEPTH_GDUMP_TIME:-01:30:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gin_depth_gdump_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_dump_gin_depth_routing_node_gates.sh
)"

cat <<EOF

=== GIN depth gate dump submitted ===
  JOBID:  ${job_id}
  Root:   ${results_root}
  Log:    logs_gnnplus/gin_depth_gdump_${job_id}.log

EOF
