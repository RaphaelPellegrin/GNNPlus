#!/usr/bin/env bash
# Launch COLLAB / IMDB-BINARY / REDDIT-BINARY (PyG TUDataset stats table).
# Completes Lukas's recommended set with MUTAG/ENZYMES/PROTEINS already run.
#
# 3 datasets × 5 variants × 5 seeds = 75 jobs, up to 20 GPUs.
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_tu_sigma_social.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_SOC_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_SOC_NUM_VARIANTS:-5}"
NUM_DATASETS="${TU_SOC_NUM_DATASETS:-3}"
NUM_TASKS="${TU_SOC_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_SOC_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_SOC_PARALLEL:-20}"
PARTITION="${TU_SOC_PARTITION:-mweber_gpu}"
NICE="${TU_SOC_NICE:-10000}"
MEM="${TU_SOC_MEM:-128GB}"
TIME="${TU_SOC_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_social] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=tu_sigma_soc
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_sigma_soc_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_SOC_NUM_SEEDS="${NUM_SEEDS}",TU_SOC_NUM_VARIANTS="${NUM_VARIANTS}",TU_SOC_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_social.sh
)"

cat <<EOF

=== TU social (COLLAB / IMDB-BINARY / REDDIT-BINARY) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (3 ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_sigma_soc_${job_id}_<TASK>.log
  Loader:        Constant() node features (0-feat social graphs)
  Docs:          https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html

  Datasets: COLLAB → IMDB-BINARY → REDDIT-BINARY  (25 tasks each)
  Variants (per ds):
    +0  GCN              lr=0.001
    +1  SiGMA_homo a2g4  lr=0.001
    +2  SiGMA_homo a2g4  lr=0.01
    +3  SiGMA_hetero a2g4 lr=0.001
    +4  SiGMA_hetero a2g4 lr=0.01

  Batches: COLLAB=32, IMDB-BINARY=64, REDDIT-BINARY=16
  W&B: tu_hh_{collab,imdb_binary,reddit_binary}_{GCN,SiGMA_homo,SiGMA_hetero}_{lr001,lr01}

  Paste JOBID into Paper_tu_sigma_homo_hetero.md

EOF
