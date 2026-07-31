#!/usr/bin/env bash
# Submit ENZYMES per-graph gate dump (reload ckpt → gate_values_per_graph.pt).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_dump_enzymes_ogpkubk9_gates.sh
#
# Optional:
#   GATE_VIZ_SEED=2 GATE_DUMP_EPOCH=999 \
#     bash bash_interface/cluster/submit_dump_enzymes_ogpkubk9_gates.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

seed="${GATE_VIZ_SEED:-2}"
scheduler="${GATE_VIZ_SCHEDULER:-plateau}"
epoch="${GATE_DUMP_EPOCH:--1}"
mem="${GATE_DUMP_MEM:-32GB}"
time_limit="${GATE_DUMP_TIME:-02:00:00}"
partition="${GATE_DUMP_PARTITION:-mweber_gpu}"

if [ -n "${GATE_VIZ_OUT_DIR:-}" ]; then
    out_dir="${GATE_VIZ_OUT_DIR}"
elif [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
else
    out_dir="results/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
fi

if [ ! -d "${out_dir}/ckpt" ]; then
    echo "ERROR: no ckpt/ under ${out_dir}"
    echo "Train first: bash bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh"
    exit 1
fi

export_vars="ALL,ENV_NAME=gnnplus"
export_vars+=",GATE_VIZ_SEED=${seed}"
export_vars+=",GATE_VIZ_SCHEDULER=${scheduler}"
export_vars+=",GATE_VIZ_OUT_DIR=${out_dir}"
export_vars+=",GATE_DUMP_EPOCH=${epoch}"
export_vars+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_vars+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR:-}"

job_id="$(
    sbatch --parsable \
        --job-name=enz_gate_dump \
        --partition="${partition}" \
        --mem="${mem}" \
        --time="${time_limit}" \
        --gpus=1 \
        --output="logs_gnnplus/enz_gate_dump_%j.log" \
        --export="${export_vars}" \
        bash_interface/cluster/run_dump_enzymes_ogpkubk9_gates.sh
)"

echo ""
echo "=== ENZYMES gate dump submitted ==="
echo "  JOBID:      ${job_id}"
echo "  Seed:       ${seed}"
echo "  Scheduler:  ${scheduler}"
echo "  Epoch:      ${epoch}  (-1 = latest ckpt)"
echo "  run_dir:    ${out_dir}"
echo "  Output:     ${out_dir}/gate_values_per_graph.pt"
echo "  Log:        logs_gnnplus/enz_gate_dump_${job_id}.log"
echo ""
echo "When done:"
echo "  ls -lh ${out_dir}/gate_values_per_graph.pt"
echo "  tail -n 40 logs_gnnplus/enz_gate_dump_${job_id}.log"
echo ""
