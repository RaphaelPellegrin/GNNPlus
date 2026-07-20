#!/usr/bin/env bash
# Launch SiGMA + GRIT attention (PATTERN + CLUSTER) on mweber_gpu.
#
# 2 datasets × 5 seeds = 10 jobs.
# Max concurrent GPUs: SIGMA_GRIT_ATTN_PARALLEL (default 5).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh
#
# Then paste ARRAY JOBID into Paper_sigma_grit_attn.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_GRIT_ATTN_NUM_SEEDS:-5}"
NUM_DATASETS="${SIGMA_GRIT_ATTN_NUM_DATASETS:-2}"
NUM_TASKS="${SIGMA_GRIT_ATTN_NUM_TASKS:-$((NUM_DATASETS * NUM_SEEDS))}"
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
    --export=ALL,ENV_NAME=gnnplus,SIGMA_GRIT_ATTN_NUM_SEEDS="${NUM_SEEDS}",SIGMA_GRIT_ATTN_NUM_DATASETS="${NUM_DATASETS}",SIGMA_GRIT_ATTN_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
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
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_grit_attn_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B groups:         paper_sigma_grit_attn_pattern
                      paper_sigma_grit_attn_cluster
  W&B tags:           sigma_grit_attn, attn_type_grit, grit_attn, <ds>, seed<k>

  Configs:
    pattern  pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml  (a2g2 GCNE, RRWP k=21)
    cluster  cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml  (a1g1 GATEDGCN, RRWP k=32)

  CLI force: gnn.hybrid.attn_type grit

  Paste JOBID into Paper_sigma_grit_attn.md + CLUSTER_LAUNCHES.md

EOF
