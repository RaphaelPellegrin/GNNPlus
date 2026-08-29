#!/usr/bin/env bash
# =============================================================================
# GCN/GIN routing — aggregate per-type test accuracy + root gate plots
#
# Evaluates best checkpoints under:
#   $GCN_GIN_ANALYZE_RESULTS_ROOT/{toy,sigma}/
#
# Writes CSVs + figures to $GCN_GIN_ANALYZE_OUT_DIR (default: repo results/).
#
# Submit from login node (no python there):
#   bash bash_interface/cluster/submit_analyze_gcn_gin_routing_results.sh
# =============================================================================

#SBATCH --job-name=gcn_gin_analyze
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
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

results_root="${GCN_GIN_ANALYZE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gcn_gin_routing}"
dataset_dir="${GCN_GIN_ANALYZE_DATASET_DIR:-${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}}"
out_dir="${GCN_GIN_ANALYZE_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
# SLURM --export splits on commas; submit script passes toy;sigma when both tracks.
tracks="${GCN_GIN_ANALYZE_TRACKS:-toy,sigma}"
tracks="${tracks//;/,}"
device="${GCN_GIN_ANALYZE_DEVICE:-auto}"

if [ ! -d "${results_root}" ]; then
  log_message "Results root missing: ${results_root}"
  exit 1
fi
dataset_pt="${dataset_dir}/GcnGinRouting/processed/train.pt"
if [ ! -f "${dataset_pt}" ]; then
  log_message "Dataset missing: ${dataset_pt}"
  exit 1
fi
if [ ! -f "scripts/synthetic/analyze_gcn_gin_routing_results.py" ]; then
  log_message "Analyze script missing — git pull on cluster?"
  exit 1
fi

mkdir -p "${out_dir}" logs_gnnplus

log_message "gcn_gin_routing analyze job=${SLURM_JOB_ID:-local}"
log_message "results_root=${results_root}"
log_message "dataset_dir=${dataset_dir}"
log_message "out_dir=${out_dir}"
log_message "tracks=${tracks} device=${device}"

extra_args=()
if [ -n "${GCN_GIN_ANALYZE_LR_TAG:-}" ]; then
  extra_args+=(--lr-tag "${GCN_GIN_ANALYZE_LR_TAG}")
fi
if [ "${GCN_GIN_ANALYZE_PLOTS_ONLY:-0}" = "1" ]; then
  extra_args+=(--plots-only)
fi

python scripts/synthetic/analyze_gcn_gin_routing_results.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --tracks "${tracks}" \
  --device "${device}" \
  "${extra_args[@]}"

log_message "Analysis complete → ${out_dir}"
log_message "  per_run_metrics.csv"
log_message "  summary_by_model.csv"
log_message "  fig_baseline_per_type.png/.pdf"
log_message "  fig_gate_by_type.png/.pdf"
