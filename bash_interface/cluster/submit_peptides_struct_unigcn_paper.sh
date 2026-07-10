#!/usr/bin/env bash
# Submit peptides-struct custom_gnn UniGCN with paper hyperparams
# (arXiv:2410.05499 / peptides-struct-UnitaryGCN-final.yaml).
#
# Defaults: 3 seeds, max 5 concurrent GPUs, --nice=10000.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_peptides_struct_unigcn_paper.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PS_UNIGCN_PAPER_NUM_SEEDS:-3}"
NUM_TASKS="${PS_UNIGCN_PAPER_NUM_TASKS:-${NUM_SEEDS}}"
ARRAY_SPEC="${PS_UNIGCN_PAPER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PS_UNIGCN_PAPER_PARALLEL:-5}"
NICE="${PS_UNIGCN_PAPER_NICE:-10000}"
MEM="${PS_UNIGCN_PAPER_MEM:-64GB}"
TIME="${PS_UNIGCN_PAPER_TIME:-120:00:00}"
WANDB_GROUP="${PS_UNIGCN_PAPER_WANDB_GROUP:-peptides_struct_unigcn_paper}"

sbatch_args=(
    --parsable
    --job-name=ps_unigcn_paper
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/ps_unigcn_paper_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PS_UNIGCN_PAPER_NUM_TASKS="${NUM_TASKS}",PS_UNIGCN_PAPER_NUM_SEEDS="${NUM_SEEDS}",PS_UNIGCN_PAPER_WANDB_GROUP="${WANDB_GROUP}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_struct_unigcn_paper.sh
)"

echo ""
echo "=== Peptides-struct paper UniGCN (custom_gnn) submitted ==="
echo "  ARRAY JOBID:  ${job_id}"
echo "  Tasks:        ${ARRAY_SPEC} (seeds 0..$((NUM_SEEDS - 1))), parallel=${PARALLEL}"
echo "  Config:       configs/gcn/peptides-struct-unigcn-paper.yaml"
echo "  Paper:        arXiv:2410.05499 / peptides-struct-UnitaryGCN-final.yaml"
echo "  Key params:   Atom+LapPE, L8, H160, residual=True, drop=0.1, bs=200, ep=250"
echo "  W&B group:    ${WANDB_GROUP}"
echo "  Logs:         logs_gnnplus/ps_unigcn_paper_${job_id}_<TASK>.log"
echo ""
