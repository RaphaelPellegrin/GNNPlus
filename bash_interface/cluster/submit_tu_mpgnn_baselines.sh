#!/usr/bin/env bash
# Launch standalone GIN / SAGE / GAT on the paper TU table set.
# Same depth/width/optim as existing GCN (L12, H64, lr=1e-3).
#
# 6 datasets × 3 models × 5 seeds = 90 jobs, up to 20 GPUs.
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_tu_mpgnn_baselines.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_MPGNN_NUM_SEEDS:-5}"
NUM_MODELS="${TU_MPGNN_NUM_MODELS:-3}"
NUM_DATASETS="${TU_MPGNN_NUM_DATASETS:-6}"
NUM_TASKS="${TU_MPGNN_NUM_TASKS:-$((NUM_DATASETS * NUM_MODELS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_MPGNN_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_MPGNN_PARALLEL:-20}"
PARTITION="${TU_MPGNN_PARTITION:-mweber_gpu}"
NICE="${TU_MPGNN_NICE:-10000}"
MEM="${TU_MPGNN_MEM:-128GB}"
TIME="${TU_MPGNN_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_mpgnn_baselines] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_mpgnn_baselines.sh

sbatch_args=(
    --parsable
    --job-name=tu_mpgnn
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_mpgnn_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_MPGNN_NUM_SEEDS="${NUM_SEEDS}",TU_MPGNN_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_mpgnn_baselines.sh
)"

cat <<EOF

=== TU MPGNN baselines (GIN / SAGE / GAT) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (6 ds × 3 models × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_mpgnn_${job_id}_<TASK>.log
  Out:           \${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/<ds>_{GIN,SAGE,GAT}_lr001_seed<s>/
  Docs:          Paper_tu_sigma_homo_hetero.md

  Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
  Models (per ds, lr=0.001, L12/H64, same as GCN):
    +0  GIN
    +1  SAGE
    +2  GAT
  Batches: bio=64, COLLAB=32, IMDB=64, REDDIT=16
  W&B: tu_hh_<ds>_{GIN,SAGE,GAT}_lr001

  Paste JOBID into Paper_tu_sigma_homo_hetero.md

EOF
