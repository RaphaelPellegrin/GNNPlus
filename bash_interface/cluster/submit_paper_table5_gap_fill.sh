#!/usr/bin/env bash
# Table 5 gap-fill: CIFAR10 MP_only (5) + COCO ungated_attn seeds 1–4 (4).
#
#   bash bash_interface/cluster/submit_paper_table5_gap_fill.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${PAPER_T5_GAP_NUM_TASKS:-9}"
ARRAY_SPEC="${PAPER_T5_GAP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_GAP_PARALLEL:-5}"
PARTITION="${PAPER_T5_GAP_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_GAP_NICE:-10000}"
MEM="${PAPER_T5_GAP_MEM:-128GB}"
TIME="${PAPER_T5_GAP_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T5_GAP_WANDB_PREFIX:-paper_T5}"
NAME_SUFFIX="${PAPER_T5_GAP_NAME_SUFFIX:-_gapfill}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_gap
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_gap_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_GAP_NUM_TASKS="${NUM_TASKS}",PAPER_T5_GAP_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_GAP_NAME_SUFFIX="${NAME_SUFFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_gap_fill.sh
)"

cat <<EOF

=== Table 5 gap-fill submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (1–5 CIFAR MP_only; 6–9 COCO ungated_attn seeds1–4)
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_gap_${job_id}_<TASK>.log
  W&B:           ${WANDB_PREFIX}_cifar10_MP_only
                 ${WANDB_PREFIX}_coco_SiGMA_ungated_attn
  Note:          COCO seed0 still running on prior job — not relaunched here
  Paste JOBID into Paper_ablations.md + CLUSTER_LAUNCHES.md

EOF
