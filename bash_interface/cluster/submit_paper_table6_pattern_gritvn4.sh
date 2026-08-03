#!/usr/bin/env bash
# PATTERN Table 6/7 on GRIT+VN4 SiGMA anchor (~87.4%).
# Default: 3 variants × seeds 5–9 = 15 jobs (Homog gated = reuse VN4 SiGMA).
#
#   bash bash_interface/cluster/submit_paper_table6_pattern_gritvn4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T6_PATTERN_GRITVN4_NUM_SEEDS:-5}"
SEED_OFFSET="${PAPER_T6_PATTERN_GRITVN4_SEED_OFFSET:-5}"
INCLUDE_HOMOG_GATED="${PAPER_T6_PATTERN_GRITVN4_INCLUDE_HOMOG_GATED:-0}"
NUM_VN="${PAPER_T6_PATTERN_GRITVN4_NUM_VN:-4}"
WANDB_PREFIX="${PAPER_T6_PATTERN_GRITVN4_WANDB_PREFIX:-paper_T6_pattern_gritvn4}"

if [ "${INCLUDE_HOMOG_GATED}" = "1" ]; then
    NUM_VARIANTS=4
else
    NUM_VARIANTS=3
fi
NUM_TASKS="${PAPER_T6_PATTERN_GRITVN4_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T6_PATTERN_GRITVN4_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T6_PATTERN_GRITVN4_PARALLEL:-5}"
NICE="${PAPER_T6_PATTERN_GRITVN4_NICE:-10000}"
MEM="${PAPER_T6_PATTERN_GRITVN4_MEM:-128GB}"
TIME="${PAPER_T6_PATTERN_GRITVN4_TIME:-120:00:00}"

sbatch_args=(
    --parsable
    --job-name=sigma_T6_pat_gritvn4
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T6_pat_gritvn4_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T6_PATTERN_GRITVN4_NUM_SEEDS="${NUM_SEEDS}",PAPER_T6_PATTERN_GRITVN4_SEED_OFFSET="${SEED_OFFSET}",PAPER_T6_PATTERN_GRITVN4_INCLUDE_HOMOG_GATED="${INCLUDE_HOMOG_GATED}",PAPER_T6_PATTERN_GRITVN4_NUM_VN="${NUM_VN}",PAPER_T6_PATTERN_GRITVN4_NUM_TASKS="${NUM_TASKS}",PAPER_T6_PATTERN_GRITVN4_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table6_pattern_gritvn4.sh
)"

cat <<EOF

=== PATTERN Table 6 on GRIT+VN${NUM_VN} submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_VARIANTS} variants × seeds ${SEED_OFFSET}–$((SEED_OFFSET + NUM_SEEDS - 1)))
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T6_pat_gritvn4_${job_id}_<TASK>.log

  SiGMA / Homog_MP gated: reuse paper_sigma_grit_attn_pattern_vn4 (~87.4%)
  New W&B: ${WANDB_PREFIX}_{Homog_MP_ungated,Hetero_MP,Hetero_MP_ungated}
  Hetero swap: GCNE,GCNE → GCNE,GINE

  Paste JOBID into Paper_table6_mnist_cifar_pattern.md + CLUSTER_LAUNCHES.md

EOF
