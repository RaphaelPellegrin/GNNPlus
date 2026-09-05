#!/usr/bin/env bash
# =============================================================================
# GIN depth-routing — aggregate per-τ test accuracy + per-layer root gates
#
# Evaluates best checkpoints under:
#   $GIN_DEPTH_ANALYZE_RESULTS_ROOT/toy/
#
# Writes CSVs + figures to $GIN_DEPTH_ANALYZE_OUT_DIR.
#
# Submit from login node:
#   bash bash_interface/cluster/submit_analyze_gin_depth_routing_results.sh
# =============================================================================

#SBATCH --job-name=gin_depth_analyze
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

results_root="${GIN_DEPTH_ANALYZE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gin_routing_depth}"
dataset_dir="${GIN_DEPTH_ANALYZE_DATASET_DIR:-${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}}"
out_dir="${GIN_DEPTH_ANALYZE_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
tracks="${GIN_DEPTH_ANALYZE_TRACKS:-toy}"
tracks="${tracks//;/,}"
device="${GIN_DEPTH_ANALYZE_DEVICE:-auto}"

if [ ! -d "${results_root}" ]; then
  log_message "Results root missing: ${results_root}"
  exit 1
fi
dataset_pt="${dataset_dir}/GinDepthRouting/processed/train.pt"
if [ ! -f "${dataset_pt}" ]; then
  log_message "Dataset missing: ${dataset_pt}"
  exit 1
fi
if [ ! -f "scripts/synthetic/analyze_gin_depth_routing_results.py" ]; then
  log_message "Analyze script missing — git pull on cluster?"
  exit 1
fi

mkdir -p "${out_dir}" logs_gnnplus

log_message "gin_depth_routing analyze job=${SLURM_JOB_ID:-local}"
log_message "results_root=${results_root}"
log_message "dataset_dir=${dataset_dir}"
log_message "out_dir=${out_dir}"

extra_args=()
if [ -n "${GIN_DEPTH_ANALYZE_LR_TAG:-}" ]; then
  extra_args+=(--lr-tag "${GIN_DEPTH_ANALYZE_LR_TAG}")
fi
if [ "${GIN_DEPTH_ANALYZE_PLOTS_ONLY:-0}" = "1" ]; then
  extra_args+=(--plots-only)
fi

python scripts/synthetic/analyze_gin_depth_routing_results.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --tracks "${tracks}" \
  --device "${device}" \
  "${extra_args[@]}"

log_message "Analysis complete → ${out_dir}"
