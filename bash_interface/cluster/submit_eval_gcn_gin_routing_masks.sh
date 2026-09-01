#!/usr/bin/env bash
# Submit SiGMA gated head-masking ablation (eval only).
#
# Usage (login node):
#   export GNNPLUS_OUT_DIR=...
#   export GNNPLUS_DATASET_DIR=...
#   bash bash_interface/cluster/submit_eval_gcn_gin_routing_masks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

results_root="${GCN_GIN_MASK_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
out_dir="${GCN_GIN_MASK_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"
tracks_display="${GCN_GIN_MASK_TRACKS:-toy,sigma}"
# sbatch --export splits on commas; use semicolons in the exported value.
tracks_export="${tracks_display//,/\;}"
lr_tag="${GCN_GIN_MASK_LR_TAG:-lr001}"

if [ ! -f "${GNNPLUS_DATASET_DIR}/GcnGinRouting/processed/train.pt" ]; then
  echo "ERROR: dataset missing at ${GNNPLUS_DATASET_DIR}/GcnGinRouting"
  exit 1
fi
if [ ! -f "scripts/synthetic/eval_gcn_gin_routing_masks.py" ]; then
  echo "ERROR: eval script missing — git pull"
  exit 1
fi

chmod +x bash_interface/cluster/run_eval_gcn_gin_routing_masks.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GCN_GIN_MASK_RESULTS_ROOT=${results_root}"
export_list+=",GCN_GIN_MASK_OUT_DIR=${out_dir}"
export_list+=",GCN_GIN_MASK_TRACKS=${tracks_export}"
export_list+=",GCN_GIN_MASK_LR_TAG=${lr_tag}"

job_id="$(
  sbatch --parsable \
    --job-name=gcn_gin_mask \
    --partition="${GCN_GIN_MASK_PARTITION:-mweber_gpu}" \
    --mem="${GCN_GIN_MASK_MEM:-16GB}" \
    --time="${GCN_GIN_MASK_TIME:-01:00:00}" \
    --gpus=1 \
    --output="logs_gnnplus/gcn_gin_mask_%j.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_eval_gcn_gin_routing_masks.sh
)"

cat <<EOF

=== GCN/GIN routing mask ablation submitted ===
  JOBID:     ${job_id}
  Model:     a0g2_gated (${lr_tag}, tracks: ${tracks_display})
  Masks:     none | mask_gin | mask_gcn
  Output:    ${out_dir}/mask_ablation_*.csv
  Figure:    ${out_dir}/fig_mask_ablation.png
  Logs:      logs_gnnplus/gcn_gin_mask_${job_id}.log
  Note:      sbatch export uses tracks=${tracks_export} (semicolons)

EOF
