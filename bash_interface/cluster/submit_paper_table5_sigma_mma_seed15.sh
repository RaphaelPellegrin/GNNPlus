#!/usr/bin/env bash
# Submit Table 5 SiGMA vs MMA — 15 new seeds × 7 datasets × 2 variants = 210 jobs.
# Max 6 concurrent GPUs. Default: wait until TU MMA + SiGMA-lite arrays finish.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch (after TU jobs 53595181 / 53595192):
#   bash bash_interface/cluster/submit_paper_table5_sigma_mma_seed15.sh
#
# Override dependency / parallel:
#   PAPER_T5_S15_DEPENDENCY=afterany:53595181,53595192 \
#   PAPER_T5_S15_PARALLEL=6 \
#     bash bash_interface/cluster/submit_paper_table5_sigma_mma_seed15.sh
#
# Smoke (peptides_func SiGMA seed5 + MMA seed5 → tasks 1,16):
#   PAPER_T5_S15_ARRAY=1,16 PAPER_T5_S15_PARALLEL=2 PAPER_T5_S15_DEPENDENCY= \
#     bash bash_interface/cluster/submit_paper_table5_sigma_mma_seed15.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_S15_NUM_SEEDS:-15}"
NUM_VARIANTS=2
NUM_DATASETS=7
NUM_TASKS="${PAPER_T5_S15_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_S15_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_S15_PARALLEL:-6}"
PARTITION="${PAPER_T5_S15_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_S15_NICE:-10000}"
MEM="${PAPER_T5_S15_MEM:-128GB}"
TIME="${PAPER_T5_S15_TIME:-120:00:00}"
# afterany: start once both arrays are no longer pending/running (success or fail).
DEPENDENCY="${PAPER_T5_S15_DEPENDENCY:-afterany:53595181,53595192}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_paper_table5_sigma_mma_seed15] GNNPLUS_OUT_DIR unset -> ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_paper_table5_sigma_mma_seed15.sh

sbatch_args=(
    --parsable
    --job-name=t5_sigma_mma_s15
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/t5_sigma_mma_s15_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_S15_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_S15_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ -n "${DEPENDENCY}" ]; then
    sbatch_args+=(--dependency="${DEPENDENCY}")
fi
if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(sbatch "${sbatch_args[@]}" bash_interface/cluster/run_paper_table5_sigma_mma_seed15.sh)"

cat <<EOF

=== Table 5 SiGMA vs MMA (15 new seeds) submitted ===
  ARRAY JOBID:   ${job_id}
  Dependency:    ${DEPENDENCY:-<none>}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (7 ds × 2 variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Logs:          logs_gnnplus/t5_sigma_mma_s15_${job_id}_<TASK>.log

  MMA = SiGMA recipe + gnn.hybrid.gate=none (same heads/width/LR/epochs).
  Peptides-func: a1g2 Homog_MP (GCN×2) for BOTH SiGMA and MMA
                 (historical MMA table runs were a1g1; this fill matches SiGMA).

  Seed blocks:
    peptides_func/struct, voc, mnist, cifar10  → 5–19
    pattern (GRIT+VN4)                         → 25–39
    cluster                                    → 20–34

  Paste JOBID into CLUSTER_LAUNCHES.md / Paper_table5_sigma_mma_seed15.md
EOF
