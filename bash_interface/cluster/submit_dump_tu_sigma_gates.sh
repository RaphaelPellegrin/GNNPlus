#!/usr/bin/env bash
# Submit per-graph gate dumps for TU SiGMA gate-viz runs (after training).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_dump_tu_sigma_gates.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${GATE_DUMP_NUM_TASKS:-6}"
ARRAY_SPEC="${GATE_DUMP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${GATE_DUMP_PARALLEL:-6}"
PARTITION="${GATE_DUMP_PARTITION:-mweber_gpu}"
MEM="${GATE_DUMP_MEM:-32GB}"
TIME="${GATE_DUMP_TIME:-02:00:00}"
SEED="${GATE_VIZ_SEED:-2}"
EPOCH="${GATE_DUMP_EPOCH:--1}"

export_vars="ALL,ENV_NAME=gnnplus"
export_vars+=",GATE_VIZ_SEED=${SEED}"
export_vars+=",GATE_DUMP_EPOCH=${EPOCH}"
export_vars+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_vars+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR:-}"

job_id="$(
    sbatch --parsable \
        --job-name=tu_gate_dump \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/tu_gate_dump_%A_%a.log" \
        --export="${export_vars}" \
        bash_interface/cluster/run_dump_tu_sigma_gates.sh
)"

cat <<EOF

=== TU SiGMA gate dump submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}
  Seed:          ${SEED}
  Epoch:         ${EPOCH} (-1 = latest)
  Output:        \$GNNPLUS_OUT_DIR/gate_viz_<ds>_sigma_powerful_seed${SEED}/gate_values_per_graph.pt
  Logs:          logs_gnnplus/tu_gate_dump_${job_id}_<TASK>.log

EOF
