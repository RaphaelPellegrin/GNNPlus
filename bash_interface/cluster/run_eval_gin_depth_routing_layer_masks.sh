#!/usr/bin/env bash
# GIN depth-routing — layer-mask ablation (gated).
#
# Submit: bash bash_interface/cluster/submit_eval_gin_depth_routing_layer_masks.sh

#SBATCH --job-name=gin_depth_lmask
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
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

results_root="${GIN_DEPTH_LMASK_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gin_routing_depth}"
out_dir="${GIN_DEPTH_LMASK_OUT_DIR:-${REPO_ROOT}/results/gin_routing_depth/analysis}"
lr_tag="${GIN_DEPTH_LMASK_LR_TAG:-lr001}"
dataset_dir="${GIN_DEPTH_LMASK_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"

mkdir -p "${out_dir}" logs_gnnplus

python scripts/synthetic/eval_gin_depth_routing_layer_masks.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --lr-tag "${lr_tag}" \
  --model l2_a0g1_gated \
  --device auto

log_message "Done → ${out_dir}/layer_mask_ablation_summary.csv"
