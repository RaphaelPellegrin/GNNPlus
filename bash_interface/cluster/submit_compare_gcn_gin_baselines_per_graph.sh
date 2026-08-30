#!/usr/bin/env bash
# Submit pairwise GCN-only vs GIN-only per-graph comparison.
#
# Usage (login node):
#   export GNNPLUS_OUT_DIR=...
#   export GNNPLUS_DATASET_DIR=...
#   bash bash_interface/cluster/submit_compare_gcn_gin_baselines_per_graph.sh

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

results_root="${GCN_GIN_PAIRWISE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
out_dir="${GCN_GIN_PAIRWISE_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
lr_tag="${GCN_GIN_PAIRWISE_LR_TAG:-lr001}"

chmod +x bash_interface/cluster/run_compare_gcn_gin_baselines_per_graph.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GCN_GIN_PAIRWISE_RESULTS_ROOT=${results_root}"
export_list+=",GCN_GIN_PAIRWISE_OUT_DIR=${out_dir}"
export_list+=",GCN_GIN_PAIRWISE_LR_TAG=${lr_tag}"

job_id="$(
  sbatch --parsable \
    --job-name=gcn_gin_pairwise \
    --partition="${GCN_GIN_PAIRWISE_PARTITION:-mweber_gpu}" \
    --mem="${GCN_GIN_PAIRWISE_MEM:-16GB}" \
    --time="${GCN_GIN_PAIRWISE_TIME:-01:00:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gcn_gin_pairwise_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_compare_gcn_gin_baselines_per_graph.sh
)"

cat <<EOF

=== GCN/GIN pairwise per-graph comparison submitted ===
  JOBID:     ${job_id}
  Results:   ${results_root}
  Output:    ${out_dir}/pairwise_baseline_*.csv
  Figure:    ${out_dir}/paper_figures/fig05_pairwise_baseline_comparison.png
  Logs:      logs_gnnplus/gcn_gin_pairwise_${job_id}.log

EOF
