#!/usr/bin/env bash
# CLUSTER Table 5 — MP_only ungated (a0g2 GATEDGCN×2, gate=none).
# Twin of gated paper_T5_cluster_MP_only (79.087±0.158%).
#
#   bash bash_interface/cluster/submit_paper_table5_cluster_mp_only_ungated.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_CLUSTER_MPUNG_NUM_SEEDS:-5}"
SEED_OFFSET="${PAPER_T5_CLUSTER_MPUNG_SEED_OFFSET:-0}"
ARRAY_SPEC="${PAPER_T5_CLUSTER_MPUNG_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PAPER_T5_CLUSTER_MPUNG_PARALLEL:-5}"
NICE="${PAPER_T5_CLUSTER_MPUNG_NICE:-10000}"
MEM="${PAPER_T5_CLUSTER_MPUNG_MEM:-128GB}"
TIME="${PAPER_T5_CLUSTER_MPUNG_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T5_CLUSTER_MPUNG_WANDB_PREFIX:-paper_T5}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_cluster_mpung
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_cluster_mpung_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_CLUSTER_MPUNG_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_CLUSTER_MPUNG_SEED_OFFSET="${SEED_OFFSET}",PAPER_T5_CLUSTER_MPUNG_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_cluster_mp_only_ungated.sh
)"

cat <<EOF

=== CLUSTER T5 MP_only_ungated submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (seeds ${SEED_OFFSET}–$((SEED_OFFSET + NUM_SEEDS - 1)))
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_cluster_mpung_${job_id}_<TASK>.log

  Arch:   a0g2 GATEDGCN,GATEDGCN  · gate=none
  Twin:   paper_T5_cluster_MP_only  (gated headwise)  79.087±0.158%
  W&B:    ${WANDB_PREFIX}_cluster_MP_only_ungated

EOF
