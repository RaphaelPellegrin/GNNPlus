#!/usr/bin/env bash
# CLUSTER Table 5 ablations on ht9bntg2 SiGMA (78.956±0.112%).
# Default: 5 variants × seeds 0–4 = 25 jobs (skip SiGMA reuse).
#
#   bash bash_interface/cluster/submit_paper_table5_cluster_ablations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_CLUSTER_NUM_SEEDS:-5}"
SEED_OFFSET="${PAPER_T5_CLUSTER_SEED_OFFSET:-0}"
INCLUDE_SIGMA="${PAPER_T5_CLUSTER_INCLUDE_SIGMA:-0}"
WANDB_PREFIX="${PAPER_T5_CLUSTER_WANDB_PREFIX:-paper_T5}"

if [ "${INCLUDE_SIGMA}" = "1" ]; then
    NUM_VARIANTS=6
else
    NUM_VARIANTS=5
fi
NUM_TASKS="${PAPER_T5_CLUSTER_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_CLUSTER_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_CLUSTER_PARALLEL:-5}"
NICE="${PAPER_T5_CLUSTER_NICE:-10000}"
MEM="${PAPER_T5_CLUSTER_MEM:-128GB}"
TIME="${PAPER_T5_CLUSTER_TIME:-120:00:00}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_cluster
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_cluster_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_CLUSTER_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_CLUSTER_SEED_OFFSET="${SEED_OFFSET}",PAPER_T5_CLUSTER_INCLUDE_SIGMA="${INCLUDE_SIGMA}",PAPER_T5_CLUSTER_NUM_TASKS="${NUM_TASKS}",PAPER_T5_CLUSTER_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_cluster_ablations.sh
)"

cat <<EOF

=== CLUSTER Table 5 ablations submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_VARIANTS} variants × seeds ${SEED_OFFSET}–$((SEED_OFFSET + NUM_SEEDS - 1)))
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_cluster_${job_id}_<TASK>.log

  SiGMA:  paper_bestmodel_v1_cluster_ht9bntg2  (78.956±0.112%)
  Cfg:    cluster-hybrid-ht9bntg2-anchor.yaml  (vanilla attn a1g1)
  W&B:    ${WANDB_PREFIX}_cluster_{SiGMA_ungated,SiGMA_attn_gate,SiGMA_ungated_attn,Attn_only,MP_only}

  Note: GRIT CLUSTER is higher (~79.11%); this matches the paper SiGMA number you cited.

EOF
