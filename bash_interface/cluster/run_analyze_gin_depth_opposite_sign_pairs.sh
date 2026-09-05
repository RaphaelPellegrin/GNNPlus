#!/usr/bin/env bash
# GIN depth-routing — opposite-sign pair analysis (+ gated/ungated eval).
#
# Submit: bash bash_interface/cluster/submit_analyze_gin_depth_opposite_sign_pairs.sh

#SBATCH --job-name=gin_depth_opp
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

results_root="${GIN_DEPTH_OPP_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
out_dir="${GIN_DEPTH_OPP_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
lr_tag="${GIN_DEPTH_OPP_LR_TAG:-lr001}"
dataset_dir="${GIN_DEPTH_OPP_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"

mkdir -p "${out_dir}" logs_gnnplus

log_message "opposite-sign depth pairs results_root=${results_root} lr=${lr_tag}"

python scripts/synthetic/analyze_gin_depth_opposite_sign_pairs.py \
  --dataset-dir "${dataset_dir}" \
  --results-root "${results_root}" \
  --out-dir "${out_dir}" \
  --lr-tag "${lr_tag}" \
  --tracks toy \
  --device auto

log_message "Done → ${out_dir}/opposite_sign_pair_summary.csv"
