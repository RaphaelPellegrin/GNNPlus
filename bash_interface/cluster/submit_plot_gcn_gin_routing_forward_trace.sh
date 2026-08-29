#!/usr/bin/env bash
# Submit GCN/GIN routing forward-trace plots (SLURM GPU worker — no python on login).
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_plot_gcn_gin_routing_forward_trace.sh
#
# Optional overrides:
#   GCN_GIN_FORWARD_RUN_DIR   (default: $GNNPLUS_OUT_DIR/gcn_gin_routing/toy/a0g2_gated_lr001_seed0)
#   GCN_GIN_FORWARD_OUT_DIR   (default: $REPO/results/gcn_gin_routing/analysis/forward_traces)
#   GCN_GIN_FORWARD_SPLIT     (test|val|train — script also scans other splits if needed)
#   GCN_GIN_FORWARD_DEVICE    (auto|cpu|cuda)
#   GCN_GIN_FORWARD_PARTITION / _MEM / _TIME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
  echo "[submit_plot_gcn_gin_forward_trace] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
  echo "[submit_plot_gcn_gin_forward_trace] GNNPLUS_DATASET_DIR unset → ${GNNPLUS_DATASET_DIR}"
fi

run_dir="${GCN_GIN_FORWARD_RUN_DIR:-${GNNPLUS_OUT_DIR}/gcn_gin_routing/toy/a0g2_gated_lr001_seed0}"
out_dir="${GCN_GIN_FORWARD_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis/forward_traces}"

if [ ! -d "${run_dir}/ckpt" ]; then
  echo "ERROR: run missing ckpt/: ${run_dir}"
  exit 1
fi
if [ ! -f "${GNNPLUS_DATASET_DIR}/GcnGinRouting/processed/train.pt" ]; then
  echo "ERROR: dataset missing at ${GNNPLUS_DATASET_DIR}/GcnGinRouting"
  exit 1
fi
if [ ! -f "scripts/synthetic/plot_gcn_gin_routing_forward_trace.py" ]; then
  echo "ERROR: plot script missing — git pull after push"
  exit 1
fi

chmod +x bash_interface/cluster/run_plot_gcn_gin_routing_forward_trace.sh

PARTITION="${GCN_GIN_FORWARD_PARTITION:-mweber_gpu}"
MEM="${GCN_GIN_FORWARD_MEM:-16GB}"
TIME="${GCN_GIN_FORWARD_TIME:-01:00:00}"

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GCN_GIN_FORWARD_RUN_DIR=${run_dir}"
export_list+=",GCN_GIN_FORWARD_OUT_DIR=${out_dir}"
export_list+=",GCN_GIN_FORWARD_SPLIT=${GCN_GIN_FORWARD_SPLIT:-test}"
export_list+=",GCN_GIN_FORWARD_DEVICE=${GCN_GIN_FORWARD_DEVICE:-auto}"

job_id="$(
  sbatch --parsable \
    --job-name=gcn_gin_fwd_trace \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --gpus=1 \
    --output="logs_gnnplus/gcn_gin_fwd_trace_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_plot_gcn_gin_routing_forward_trace.sh
)"

cat <<EOF

=== GCN/GIN routing forward-trace plots submitted ===
  JOBID:     ${job_id}
  Partition: ${PARTITION} (1 GPU)
  Run:       ${run_dir}
  Dataset:   ${GNNPLUS_DATASET_DIR}
  Output:    ${out_dir}
  Logs:      logs_gnnplus/gcn_gin_fwd_trace_${job_id}.log

Monitor:
  squeue -j ${job_id}
  tail -f logs_gnnplus/gcn_gin_fwd_trace_${job_id}.log

EOF
