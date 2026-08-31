#!/usr/bin/env bash
# Submit opposite-sign τ pair analysis (fig07).
#
# Usage (login node, fast path — uses existing pairwise CSV):
#   export GNNPLUS_OUT_DIR=...
#   export GNNPLUS_DATASET_DIR=...
#   bash bash_interface/cluster/submit_analyze_opposite_sign_pairs.sh
#
# Include SiGMA gated eval (GPU):
#   export GCN_GIN_OPPOSITE_INCLUDE_GATED=1

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

results_root="${GCN_GIN_OPPOSITE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
out_dir="${GCN_GIN_OPPOSITE_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
lr_tag="${GCN_GIN_OPPOSITE_LR_TAG:-lr001}"
from_csv="${GCN_GIN_OPPOSITE_FROM_CSV:-${out_dir}/pairwise_baseline_per_graph.csv}"
include_gated="${GCN_GIN_OPPOSITE_INCLUDE_GATED:-0}"

chmod +x bash_interface/cluster/run_analyze_opposite_sign_pairs.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GCN_GIN_OPPOSITE_RESULTS_ROOT=${results_root}"
export_list+=",GCN_GIN_OPPOSITE_OUT_DIR=${out_dir}"
export_list+=",GCN_GIN_OPPOSITE_LR_TAG=${lr_tag}"
export_list+=",GCN_GIN_OPPOSITE_FROM_CSV=${from_csv}"
export_list+=",GCN_GIN_OPPOSITE_INCLUDE_GATED=${include_gated}"

# Login-node fast path: no GPU if CSV exists and gated eval is off.
partition="${GCN_GIN_OPPOSITE_PARTITION:-mweber_gpu}"
mem="${GCN_GIN_OPPOSITE_MEM:-16GB}"
time_limit="${GCN_GIN_OPPOSITE_TIME:-01:00:00}"
sbatch_extra=()
if [[ -f "${from_csv}" && "${include_gated}" != "1" ]]; then
  partition="${GCN_GIN_OPPOSITE_PARTITION:-serial_requeue}"
  mem="${GCN_GIN_OPPOSITE_MEM:-8GB}"
  time_limit="${GCN_GIN_OPPOSITE_TIME:-00:15:00}"
else
  sbatch_extra+=(--gpus=1)
fi

sbatch_args=(
  --parsable
  --job-name=gcn_gin_opp_pairs
  --partition="${partition}"
  --mem="${mem}"
  --time="${time_limit}"
  --output="logs_gnnplus/gcn_gin_opp_pairs_%j.log"
  --export="${export_list}"
  "${sbatch_extra[@]}"
)

job_id="$(sbatch "${sbatch_args[@]}" bash_interface/cluster/run_analyze_opposite_sign_pairs.sh)"

cat <<EOF

=== Opposite-sign τ pair analysis submitted ===
  JOBID:     ${job_id}
  CSV:       ${from_csv}
  Gated:     ${include_gated}
  Output:    ${out_dir}/opposite_sign_pair_summary.csv
  Figure:    ${out_dir}/paper_figures/fig07_opposite_sign_pair_outcomes.png
  Table:     ${out_dir}/paper_figures/fig07_opposite_sign_pair_table.png
  Logs:      logs_gnnplus/gcn_gin_opp_pairs_${job_id}.log

EOF
