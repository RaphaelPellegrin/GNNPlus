#!/usr/bin/env bash
# SLURM worker: SiGMA gated MP head masking ablation at eval.

#SBATCH --job-name=gcn_gin_mask
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

results_root="${GCN_GIN_MASK_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
dataset_dir="${GCN_GIN_MASK_DATASET_DIR:-${GNNPLUS_DATASET_DIR}}"
out_dir="${GCN_GIN_MASK_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
tracks="${GCN_GIN_MASK_TRACKS:-toy,sigma}"
tracks="${tracks//;/,}"
lr_tag="${GCN_GIN_MASK_LR_TAG:-lr001}"

log_message "gcn_gin mask ablation job=${SLURM_JOB_ID:-local}"
log_message "results_root=${results_root} out_dir=${out_dir} tracks=${tracks}"

python scripts/synthetic/eval_gcn_gin_routing_masks.py \
  --results-root "${results_root}" \
  --dataset-dir "${dataset_dir}" \
  --out-dir "${out_dir}" \
  --tracks "${tracks}" \
  --lr-tag "${lr_tag}" \
  --device auto

log_message "Done → ${out_dir}/fig_mask_ablation.png"
