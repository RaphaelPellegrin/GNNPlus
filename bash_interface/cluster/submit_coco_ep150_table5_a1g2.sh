#!/usr/bin/env bash
# Submit COCO Table 5 full a1g2 family @ ep150 (6 variants × 5 seeds = 30).
#
# Anchor: 5b4z9l3u a1g1 SiGMA (paper ~0.42) expanded to a1g2 / a3 / a0g3.
#
#   bash bash_interface/cluster/submit_coco_ep150_table5_a1g2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_COCO_A1G2_EP150_NUM_SEEDS:-5}"
NUM_VARIANTS="${PAPER_T5_COCO_A1G2_EP150_NUM_VARIANTS:-6}"
NUM_TASKS=$((NUM_VARIANTS * NUM_SEEDS))
ARRAY_SPEC="${PAPER_T5_COCO_A1G2_EP150_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_COCO_A1G2_EP150_PARALLEL:-5}"
PARTITION="${PAPER_T5_COCO_A1G2_EP150_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_COCO_A1G2_EP150_NICE:-10000}"
MEM="${PAPER_T5_COCO_A1G2_EP150_MEM:-128GB}"
TIME="${PAPER_T5_COCO_A1G2_EP150_TIME:-120:00:00}"
MAX_EPOCH="${PAPER_T5_COCO_A1G2_EP150_MAX_EPOCH:-150}"
WANDB_PREFIX="${PAPER_T5_COCO_A1G2_EP150_WANDB_PREFIX:-paper_T5_ep150}"
NAME_SUFFIX="${PAPER_T5_COCO_A1G2_EP150_NAME_SUFFIX:-_ep150_a1g2}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_ep150_table5_a1g2] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=sigma_T5_coco_a1g2_ep150
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_coco_a1g2_ep150_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_COCO_A1G2_EP150_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_COCO_A1G2_EP150_NUM_VARIANTS="${NUM_VARIANTS}",PAPER_T5_COCO_A1G2_EP150_MAX_EPOCH="${MAX_EPOCH}",PAPER_T5_COCO_A1G2_EP150_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_COCO_A1G2_EP150_NAME_SUFFIX="${NAME_SUFFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_coco_ep150_table5_a1g2.sh
)"

cat <<EOF

=== COCO Table 5 a1g2 @ ep${MAX_EPOCH} submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (6 variants × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_coco_a1g2_ep150_${job_id}_<TASK>.log

  Anchor:  coco-hybrid-5b4z9l3u-a1g1-anchor.yaml  (SiGMA a1g1 ~0.42)
  Twin:    a1g2 / a3 / a0g3  (total_heads=3)

  Task map:
    1–5   SiGMA_a1g2
    6–10  SiGMA_ungated_a1g2
    11–15 SiGMA_attn_gate_a1g2      (Hybrid, ungated MP)
    16–20 SiGMA_ungated_attn_a1g2   (Hybrid, ungated Att)
    21–25 Attn_only_a3
    26–30 MP_only_a0g3

  W&B: ${WANDB_PREFIX}_coco_{SiGMA_a1g2,SiGMA_ungated_a1g2,SiGMA_attn_gate_a1g2,
        SiGMA_ungated_attn_a1g2,Attn_only_a3,MP_only_a0g3}

EOF
