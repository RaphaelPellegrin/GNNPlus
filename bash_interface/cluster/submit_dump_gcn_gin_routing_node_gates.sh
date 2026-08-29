#!/usr/bin/env bash
# Submit per-node gate dumps for GCN/GIN routing gated runs (20 array tasks).
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_dump_gcn_gin_routing_node_gates.sh
#
# Optional:
#   GCN_GIN_GATE_DUMP_SKIP_EXISTING=1
#   GCN_GIN_GATE_DUMP_SPLITS=test
#   GCN_GIN_GATE_DUMP_DEPENDENCY=afterok:<analyze_jobid>

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

NUM_SEEDS="${GCN_GIN_GATE_DUMP_NUM_SEEDS:-5}"
NUM_LRS="${GCN_GIN_GATE_DUMP_NUM_LRS:-2}"
NUM_TRACKS="${GCN_GIN_GATE_DUMP_NUM_TRACKS:-2}"
NUM_TASKS=$((NUM_TRACKS * NUM_LRS * NUM_SEEDS))
PARTITION="${GCN_GIN_GATE_DUMP_PARTITION:-mweber_gpu}"
MEM="${GCN_GIN_GATE_DUMP_MEM:-16GB}"
TIME="${GCN_GIN_GATE_DUMP_TIME:-01:00:00}"

results_root="${GCN_GIN_GATE_DUMP_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"

chmod +x bash_interface/cluster/run_dump_gcn_gin_routing_node_gates.sh

_splits_display="${GCN_GIN_GATE_DUMP_SPLITS:-train,val,test}"
_splits_export="${_splits_display//,/\;}"

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GCN_GIN_GATE_DUMP_RESULTS_ROOT=${results_root}"
export_list+=",GCN_GIN_GATE_DUMP_NUM_SEEDS=${NUM_SEEDS}"
export_list+=",GCN_GIN_GATE_DUMP_NUM_LRS=${NUM_LRS}"
export_list+=",GCN_GIN_GATE_DUMP_NUM_TRACKS=${NUM_TRACKS}"
export_list+=",GCN_GIN_GATE_DUMP_SPLITS=${_splits_export}"
export_list+=",GCN_GIN_GATE_DUMP_SKIP_EXISTING=${GCN_GIN_GATE_DUMP_SKIP_EXISTING:-0}"

job_id="$(
  sbatch --parsable \
    --job-name=gcn_gin_gdump \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --array="1-${NUM_TASKS}" \
    --gpus=1 \
    --output="logs_gnnplus/gcn_gin_gdump_%A_%a.log" \
    --export="${export_list}" \
    ${GCN_GIN_GATE_DUMP_DEPENDENCY:+--dependency="${GCN_GIN_GATE_DUMP_DEPENDENCY}"} \
    bash_interface/cluster/run_dump_gcn_gin_routing_node_gates.sh
)"

cat <<EOF

=== GCN/GIN routing node gate dump submitted ===
  JOBID:         ${job_id} (array 1-${NUM_TASKS})
  Partition:     ${PARTITION}
  Results:       ${results_root}
  Dataset:       ${GNNPLUS_DATASET_DIR}
  Per-run out:   gate_values_per_node.pt + gate_graph_summary.csv
  Logs:          logs_gnnplus/gcn_gin_gdump_${job_id}_<TASK>.log

Inspect (after dump):
  python scripts/synthetic/inspect_gcn_gin_routing_node_gates.py \\
    --results-root ${results_root}/toy --split test
  python scripts/synthetic/inspect_gcn_gin_routing_node_gates.py \\
    --results-root ${results_root}/sigma --split test \\
    --export-csv results/gcn_gin_routing/analysis/gate_node_summary_test.csv

EOF
