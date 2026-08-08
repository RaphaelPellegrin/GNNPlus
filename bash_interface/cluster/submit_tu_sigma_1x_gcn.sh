#!/usr/bin/env bash
# Submit TU param-matched SiGMA (~1× GCN) + GPS-style a1g1 seed grids.
#
# 6 datasets × 6 variants × 5 seeds = 180 jobs, up to 20 GPUs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_tu_sigma_1x_gcn.sh
#
# Smoke (MUTAG, all 6 variants, 1 seed → tasks 1,6,11,16,21,26):
#   TU_1X_ARRAY=1,6,11,16,21,26 TU_1X_PARALLEL=6 \
#     bash bash_interface/cluster/submit_tu_sigma_1x_gcn.sh
#
# Paste printed JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_1X_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_1X_NUM_VARIANTS:-6}"
NUM_DATASETS="${TU_1X_NUM_DATASETS:-6}"
NUM_TASKS="${TU_1X_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_1X_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_1X_PARALLEL:-20}"
PARTITION="${TU_1X_PARTITION:-mweber_gpu}"
NICE="${TU_1X_NICE:-10000}"
MEM="${TU_1X_MEM:-128GB}"
TIME="${TU_1X_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_1x_gcn] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_sigma_1x_gcn.sh

sbatch_args=(
    --parsable
    --job-name=tu_1x_gcn
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_1x_gcn_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_1X_NUM_SEEDS="${NUM_SEEDS}",TU_1X_NUM_VARIANTS="${NUM_VARIANTS}",TU_1X_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_1x_gcn.sh
)"

cat <<EOF

=== TU SiGMA ~1× GCN + GPS submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_1x_gcn_${job_id}_<TASK>.log
  Outs:          \$GNNPLUS_OUT_DIR/tu_sigma_1x_gcn/<ds>_<variant>_<lr>_seed<s>/
  Configs:       configs/tu_sigma_homo_hetero/{sigma-*-matched,gps-a1g1}-anchor.yaml
  Docs:          Paper_tu_sigma_homo_hetero.md

  Datasets: MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY
  Variants (per ds, blocks of ${NUM_SEEDS} seeds):
    +0  SiGMA_homo   a2g4 d_h=4   lr=0.001   (~1.02× GCN params)
    +1  SiGMA_homo   a2g4 d_h=4   lr=0.01
    +2  SiGMA_hetero a2g4 d_h=4   lr=0.001
    +3  SiGMA_hetero a2g4 d_h=4   lr=0.01
    +4  GPS          a1g1 GATEDGCN+attn d_h=8  lr=0.001   (~1.01× GCN)
    +5  GPS          a1g1 GATEDGCN+attn d_h=8  lr=0.01
  Batches: bio=64, COLLAB=32, IMDB=64, REDDIT=16

  W&B groups: tu_1x_<ds>_{SiGMA_homo,SiGMA_hetero,GPS}_{lr001,lr01}
  (distinct from prior tu_hh_* ~1.65× SiGMA runs)

  Aggregate example:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group tu_1x_enzymes_SiGMA_hetero_lr001 --metric best_test_perf --state finished

  Paste JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

EOF
