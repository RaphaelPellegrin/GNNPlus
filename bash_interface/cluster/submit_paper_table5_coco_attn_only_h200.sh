#!/usr/bin/env bash
# Relaunch Table 5 COCO Attn_only (5 seeds) on gpu_h200.
# Does not cancel old 32232124_71..75 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_coco_attn_only_h200.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_COCO_ATTN_NUM_SEEDS:-5}"
ARRAY_SPEC="${PAPER_T5_COCO_ATTN_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${PAPER_T5_COCO_ATTN_PARALLEL:-5}"
PARTITION="${PAPER_T5_COCO_ATTN_PARTITION:-gpu_h200}"
NICE="${PAPER_T5_COCO_ATTN_NICE:-10000}"
MEM="${PAPER_T5_COCO_ATTN_MEM:-128GB}"
TIME="${PAPER_T5_COCO_ATTN_TIME:-72:00:00}"
WANDB_PREFIX="${PAPER_T5_COCO_ATTN_WANDB_PREFIX:-paper_T5}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_coco_attn
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_coco_attn_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_COCO_ATTN_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_COCO_ATTN_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_coco_attn_only_h200.sh
)"

cat <<EOF

=== Table 5 COCO Attn_only H200 relaunch submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (seeds 0..$((NUM_SEEDS - 1)); ≤${PARALLEL} GPUs)
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_coco_attn_${job_id}_<TASK>.log

  W&B group:     ${WANDB_PREFIX}_coco_Attn_only
  Anchor:        coco-hybrid-5b4z9l3u-a1g1-anchor.yaml (Attn_only a2g0)
  Note:          old 32232124_71..75 left running; new names have _h200 suffix

  Paste JOBID into Paper_ablations.md + CLUSTER_LAUNCHES.md

EOF
