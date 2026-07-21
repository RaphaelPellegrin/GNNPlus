#!/usr/bin/env bash
# Launch SiGMA + GRIT attention (PATTERN + CLUSTER) on mweber_gpu.
#
# Defaults: 1 variant (no VN) × 2 datasets × 5 seeds = 10 jobs.
#
# Resubmit seeds 5–9 + VN=4 (20 jobs):
#   SIGMA_GRIT_ATTN_SEED_OFFSET=5 \
#   SIGMA_GRIT_ATTN_NUM_VARIANTS=2 \
#   SIGMA_GRIT_ATTN_NUM_VN=4 \
#   bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_GRIT_ATTN_NUM_SEEDS:-5}"
NUM_DATASETS="${SIGMA_GRIT_ATTN_NUM_DATASETS:-2}"
NUM_VARIANTS="${SIGMA_GRIT_ATTN_NUM_VARIANTS:-1}"
SEED_OFFSET="${SIGMA_GRIT_ATTN_SEED_OFFSET:-0}"
NUM_VN="${SIGMA_GRIT_ATTN_NUM_VN:-4}"
NUM_TASKS="${SIGMA_GRIT_ATTN_NUM_TASKS:-$((NUM_VARIANTS * NUM_DATASETS * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_GRIT_ATTN_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_GRIT_ATTN_PARALLEL:-5}"
NICE="${SIGMA_GRIT_ATTN_NICE:-10000}"
MEM="${SIGMA_GRIT_ATTN_MEM:-128GB}"
TIME="${SIGMA_GRIT_ATTN_TIME:-120:00:00}"

sbatch_args=(
    --parsable
    --job-name=sigma_grit_attn
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_grit_attn_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_GRIT_ATTN_NUM_SEEDS="${NUM_SEEDS}",SIGMA_GRIT_ATTN_NUM_DATASETS="${NUM_DATASETS}",SIGMA_GRIT_ATTN_NUM_VARIANTS="${NUM_VARIANTS}",SIGMA_GRIT_ATTN_SEED_OFFSET="${SEED_OFFSET}",SIGMA_GRIT_ATTN_NUM_VN="${NUM_VN}",SIGMA_GRIT_ATTN_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_grit_attn_pattern_cluster.sh
)"

cat <<EOF

=== SiGMA + GRIT attention (PATTERN + CLUSTER) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_VARIANTS} var × ${NUM_DATASETS} ds × ${NUM_SEEDS} seeds)
  Seed offset:   ${SEED_OFFSET}  (seeds ${SEED_OFFSET}..$((SEED_OFFSET + NUM_SEEDS - 1)))
  VN (variant1): ${NUM_VN}  (variants=${NUM_VARIANTS}; 1=no-VN only)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/sigma_grit_attn_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B groups:         paper_sigma_grit_attn_{pattern,cluster}
                      paper_sigma_grit_attn_{pattern,cluster}_vn${NUM_VN}  (if variants>=2)

  CLI force: gnn.hybrid.attn_type grit

  Paste JOBID into Paper_sigma_grit_attn.md + CLUSTER_LAUNCHES.md

EOF
