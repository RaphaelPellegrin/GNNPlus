#!/usr/bin/env bash
# Full GIN depth-routing analysis pack (run on cluster GPU node or locally).
#
# Order:
#   1) per-τ accuracy + layer×τ gates
#   2) opposite-sign pair outcomes
#   3) layer-mask ablation
#   4) gate dump
#   5) ranked gate plots
#   6) gates by role × layer × τ
#
# Submit wrapper:
#   bash bash_interface/cluster/submit_gin_depth_routing_full_analysis.sh
#
# Or run directly after training finishes:
#   bash bash_interface/cluster/run_gin_depth_routing_full_analysis.sh

#SBATCH --job-name=gin_depth_full_an
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
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

results_root="${GIN_DEPTH_ANALYZE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
dataset_dir="${GIN_DEPTH_ANALYZE_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"
out_dir="${GIN_DEPTH_ANALYZE_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
lr_tag="${GIN_DEPTH_ANALYZE_LR_TAG:-lr001}"

mkdir -p "${out_dir}" logs_gnnplus

log_message "=== [1/6] per-τ accuracy + gates ==="
python scripts/synthetic/analyze_gin_depth_routing_results.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --tracks toy \
  --device auto

log_message "=== [2/6] opposite-sign pairs ==="
python scripts/synthetic/analyze_gin_depth_opposite_sign_pairs.py \
  --dataset-dir "${dataset_dir}" \
  --results-root "${results_root}" \
  --out-dir "${out_dir}" \
  --lr-tag "${lr_tag}" \
  --tracks toy \
  --device auto

log_message "=== [3/6] layer-mask ablation ==="
python scripts/synthetic/eval_gin_depth_routing_layer_masks.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --lr-tag "${lr_tag}" \
  --model l2_a0g1_gated \
  --device auto

log_message "=== [4/6] gate dump ==="
python scripts/synthetic/dump_gin_depth_routing_node_gates.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --tracks toy \
  --device auto \
  --skip-existing

log_message "=== [5/6] ranked gate plots ==="
python scripts/synthetic/plot_gin_depth_routing_ranked_gates.py \
  --results-root "${results_root}" \
  --out-dir "${out_dir}/ranked_gates" \
  --lr-tag "${lr_tag}" \
  --split test

log_message "=== [6/6] gates by role × layer × τ ==="
python scripts/synthetic/analyze_gin_depth_gates_by_role.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --tracks toy \
  --lr-tag "${lr_tag}" \
  --model l2_a0g1_gated \
  --device auto

log_message "Full analysis complete → ${out_dir}"
ls -lh "${out_dir}"/*.png "${out_dir}"/paper_figures/*.png 2>/dev/null || true
