#!/usr/bin/env bash
# Submit toy-track GCN/GIN routing runs WITHOUT node encoder (pedagogy ablation).
#
# Usage (cluster login):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_gcn_gin_routing_noxenc.sh
#
# Defaults: 4 models × 1 LR × 1 seed = 4 GPU tasks (~minutes each on toy).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

NUM_SEEDS="${GCN_GIN_NOXENC_NUM_SEEDS:-1}"
NUM_LRS="${GCN_GIN_NOXENC_NUM_LRS:-1}"
NUM_MODELS=4
NUM_TASKS=$((NUM_MODELS * NUM_LRS * NUM_SEEDS))
ARRAY_SPEC="${GCN_GIN_NOXENC_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${GCN_GIN_NOXENC_PARALLEL:-4}"
PARTITION="${GCN_GIN_NOXENC_PARTITION:-mweber_gpu}"
MEM="${GCN_GIN_NOXENC_MEM:-32GB}"
TIME="${GCN_GIN_NOXENC_TIME:-02:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

python scripts/synthetic/generate_gcn_gin_routing_configs.py --noxenc

missing=0
for slug in a0g2_gated a0g2_ungated a0g1_gcn a0g1_gin; do
  cfg="configs/synthetic/gcn_gin_routing_toy_${slug}_noxenc.yaml"
  if [ ! -f "${cfg}" ]; then
    echo "MISSING ${cfg}"
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  exit 1
fi

chmod +x bash_interface/cluster/run_gcn_gin_routing_noxenc.sh

job_id="$(
  sbatch --parsable \
    --job-name=gcn_gin_noxenc \
    --array="${ARRAY_SPEC}%${PARALLEL}" \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --gpus=1 \
    --output="logs_gnnplus/gcn_gin_noxenc_%A_%a.log" \
    --export=ALL,ENV_NAME=gnnplus,GCN_GIN_NOXENC_NUM_SEEDS="${NUM_SEEDS}",GCN_GIN_NOXENC_NUM_LRS="${NUM_LRS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}" \
    bash_interface/cluster/run_gcn_gin_routing_noxenc.sh
)"

cat <<EOF

=== GCN/GIN routing no-encoder (toy) submitted ===
  JOBID:     ${job_id}
  Tasks:     ${ARRAY_SPEC} (4 models × ${NUM_LRS} LR × ${NUM_SEEDS} seeds)
  Out:       \$GNNPLUS_OUT_DIR/gcn_gin_routing/toy/<model>_noxenc_<lr>_seed<s>/
  Forward traces after training:
    GCN_GIN_FORWARD_RUN_DIR=\$GNNPLUS_OUT_DIR/gcn_gin_routing/toy/a0g2_gated_noxenc_lr001_seed0 \\
      GCN_GIN_FORWARD_OUT_DIR=\$PWD/results/gcn_gin_routing/analysis/forward_traces/noxenc_gated \\
      bash bash_interface/cluster/submit_plot_gcn_gin_routing_forward_trace.sh

EOF
