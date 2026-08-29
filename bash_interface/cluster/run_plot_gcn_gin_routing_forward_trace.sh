#!/usr/bin/env bash
# =============================================================================
# GCN/GIN routing — forward-trace figures (τ × correct/incorrect)
#
# One PNG per case under $GCN_GIN_FORWARD_OUT_DIR:
#   fig_forward_tau{0,1}_{correct,incorrect}.png
#
# Submit from login node (no python there):
#   bash bash_interface/cluster/submit_plot_gcn_gin_routing_forward_trace.sh
# =============================================================================

#SBATCH --job-name=gcn_gin_fwd_trace
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=logs_gnnplus/%x_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

run_dir="${GCN_GIN_FORWARD_RUN_DIR:-${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gcn_gin_routing/toy/a0g2_gated_lr001_seed0}"
dataset_dir="${GCN_GIN_FORWARD_DATASET_DIR:-${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}}"
out_dir="${GCN_GIN_FORWARD_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis/forward_traces}"
split="${GCN_GIN_FORWARD_SPLIT:-test}"
device="${GCN_GIN_FORWARD_DEVICE:-auto}"
dpi="${GCN_GIN_FORWARD_DPI:-160}"

if [ ! -d "${run_dir}/ckpt" ]; then
  log_message "Run directory missing ckpt/: ${run_dir}"
  exit 1
fi
if [ -z "$(ls -A "${run_dir}/ckpt"/*.ckpt 2>/dev/null || true)" ]; then
  log_message "No checkpoints in ${run_dir}/ckpt"
  exit 1
fi
dataset_pt="${dataset_dir}/GcnGinRouting/processed/train.pt"
if [ ! -f "${dataset_pt}" ]; then
  log_message "Dataset missing: ${dataset_pt}"
  exit 1
fi
if [ ! -f "scripts/synthetic/plot_gcn_gin_routing_forward_trace.py" ]; then
  log_message "Script missing — git pull on cluster?"
  exit 1
fi

mkdir -p "${out_dir}" logs_gnnplus

log_message "gcn_gin_routing forward_trace job=${SLURM_JOB_ID:-local}"
log_message "run_dir=${run_dir}"
log_message "dataset_dir=${dataset_dir}"
log_message "out_dir=${out_dir}"
log_message "split=${split} device=${device}"

python scripts/synthetic/plot_gcn_gin_routing_forward_trace.py \
  --run-dir "${run_dir}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --split "${split}" \
  --device "${device}" \
  --dpi "${dpi}"

log_message "Forward traces complete → ${out_dir}"
log_message "  fig_forward_tau0_correct.png"
log_message "  fig_forward_tau0_incorrect.png"
log_message "  fig_forward_tau1_correct.png"
log_message "  fig_forward_tau1_incorrect.png"
