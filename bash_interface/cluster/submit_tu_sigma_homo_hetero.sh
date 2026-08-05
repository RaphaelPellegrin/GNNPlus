#!/usr/bin/env bash
# Submit TU GCN vs SiGMA(homo) vs SiGMA(hetero) seed grids.
#
# Default: 6 datasets × 5 variants × 5 seeds = 150 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_tu_sigma_homo_hetero.sh
#
# Smoke (1 dataset × 1 seed × all variants = 5 tasks):
#   TU_SIGMA_HH_ARRAY=1-5 TU_SIGMA_HH_PARALLEL=5 \
#     bash bash_interface/cluster/submit_tu_sigma_homo_hetero.sh
#
# Paste printed JOBID into Paper_tu_sigma_homo_hetero.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_SIGMA_HH_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_SIGMA_HH_NUM_VARIANTS:-5}"
NUM_DATASETS="${TU_SIGMA_HH_NUM_DATASETS:-6}"
NUM_TASKS="${TU_SIGMA_HH_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_SIGMA_HH_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_SIGMA_HH_PARALLEL:-8}"
PARTITION="${TU_SIGMA_HH_PARTITION:-mweber_gpu}"
NICE="${TU_SIGMA_HH_NICE:-10000}"
MEM="${TU_SIGMA_HH_MEM:-64GB}"
TIME="${TU_SIGMA_HH_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_sigma_homo_hetero] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=tu_sigma_hh
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_sigma_hh_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,TU_SIGMA_HH_NUM_SEEDS="${NUM_SEEDS}",TU_SIGMA_HH_NUM_VARIANTS="${NUM_VARIANTS}",TU_SIGMA_HH_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_sigma_homo_hetero.sh
)"

cat <<EOF

=== TU SiGMA homo vs hetero submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/tu_sigma_hh_${job_id}_<TASK>.log
  Outs:          \$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_<variant>_<lr>_seed<s>/
                 ├── ckpt/                      (best-val model)
                 ├── gate_values_per_graph.pt   (SiGMA only; auto after train)
                 ├── config_used.yaml
                 └── train_meta.txt
  Configs:       configs/tu_sigma_homo_hetero/
  Docs:          Paper_tu_sigma_homo_hetero.md

  Datasets: MUTAG ENZYMES PROTEINS DD NCI1 TRIANGLES
  Variants (per ds, blocks of ${NUM_SEEDS} seeds):
    +0  GCN              lr=0.001
    +1  SiGMA_homo a2g4  lr=0.001   (GCN×4)
    +2  SiGMA_homo a2g4  lr=0.01
    +3  SiGMA_hetero a2g4 lr=0.001  (GCN,GIN,SAGE,GAT)
    +4  SiGMA_hetero a2g4 lr=0.01

  W&B groups: tu_hh_<ds>_{GCN,SiGMA_homo,SiGMA_hetero}_{lr001,lr01}

  Gate dump: auto for SiGMA (set TU_SIGMA_HH_GATE_DUMP=0 to skip).
  Re-dump offline: bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh

  Aggregate (per group):
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group tu_hh_enzymes_SiGMA_hetero_lr001 --metric best_test_perf --state finished

  Paste JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

EOF
