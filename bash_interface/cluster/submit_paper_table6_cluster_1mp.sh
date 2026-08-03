#!/usr/bin/env bash
# CLUSTER Table 6/7 — +1 MP head homog/hetero on ht9bntg2 (a1g1).
# 4 variants × seeds 0–4 = 20 jobs. SiGMA = reuse paper_bestmodel (78.956%).
#
#   bash bash_interface/cluster/submit_paper_table6_cluster_1mp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T6_CLUSTER_NUM_SEEDS:-5}"
SEED_OFFSET="${PAPER_T6_CLUSTER_SEED_OFFSET:-0}"
NUM_VARIANTS=4
NUM_TASKS="${PAPER_T6_CLUSTER_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T6_CLUSTER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T6_CLUSTER_PARALLEL:-5}"
NICE="${PAPER_T6_CLUSTER_NICE:-10000}"
MEM="${PAPER_T6_CLUSTER_MEM:-128GB}"
TIME="${PAPER_T6_CLUSTER_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T6_CLUSTER_WANDB_PREFIX:-paper_T6}"

sbatch_args=(
    --parsable
    --job-name=sigma_T6_cluster
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T6_cluster_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T6_CLUSTER_NUM_SEEDS="${NUM_SEEDS}",PAPER_T6_CLUSTER_SEED_OFFSET="${SEED_OFFSET}",PAPER_T6_CLUSTER_NUM_TASKS="${NUM_TASKS}",PAPER_T6_CLUSTER_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table6_cluster_1mp.sh
)"

cat <<EOF

=== CLUSTER Table 6 (+1 MP) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (Homog_MP / Hetero_MP / Homog_ungated / Hetero_ungated × 5)
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T6_cluster_${job_id}_<TASK>.log

  SiGMA:   paper_bestmodel_v1_cluster_ht9bntg2 (78.956±0.112%)
  Homog:   GATEDGCN → GATEDGCN,GATEDGCN
  Hetero:  GATEDGCN → GATEDGCN,GCN
  W&B:     ${WANDB_PREFIX}_cluster_{Homog_MP,Hetero_MP,Homog_MP_ungated,Hetero_MP_ungated}

EOF
