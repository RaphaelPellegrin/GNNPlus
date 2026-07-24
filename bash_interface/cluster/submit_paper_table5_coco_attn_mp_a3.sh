#!/usr/bin/env bash
# Launch paper Table 6 COCO Attn_only (a3g0) + MP_only (a0g3 GATEDGCN×3).
# 2 variants × 5 seeds = 10 jobs. Does not cancel existing a2 Attn/MP runs.
#
# Defaults: mweber_gpu, ≤5 GPUs, 96h walltime, **optim.max_epoch=150**.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_coco_attn_mp_a3.sh
#
# Overrides:
#   PAPER_T5_COCO_A3_PARTITION=gpu_h200 PAPER_T5_COCO_A3_PARALLEL=10 \
#     PAPER_T5_COCO_A3_TIME=72:00:00 bash bash_interface/cluster/submit_paper_table5_coco_attn_mp_a3.sh
#   PAPER_T5_COCO_A3_MAX_EPOCH=300  # restore full length if needed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_COCO_A3_NUM_SEEDS:-5}"
NUM_TASKS=$((2 * NUM_SEEDS))
ARRAY_SPEC="${PAPER_T5_COCO_A3_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_COCO_A3_PARALLEL:-5}"
PARTITION="${PAPER_T5_COCO_A3_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_COCO_A3_NICE:-10000}"
MEM="${PAPER_T5_COCO_A3_MEM:-128GB}"
if [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="${PAPER_T5_COCO_A3_TIME:-72:00:00}"
else
    TIME="${PAPER_T5_COCO_A3_TIME:-96:00:00}"
fi
WANDB_PREFIX="${PAPER_T5_COCO_A3_WANDB_PREFIX:-paper_T5}"
NAME_SUFFIX="${PAPER_T5_COCO_A3_NAME_SUFFIX:-_ep150}"
HEADS="${PAPER_T5_COCO_A3_HEADS:-3}"
MAX_EPOCH="${PAPER_T5_COCO_A3_MAX_EPOCH:-150}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_a3] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=sigma_T5_coco_a3
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_coco_a3_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_COCO_A3_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_COCO_A3_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_COCO_A3_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T5_COCO_A3_HEADS="${HEADS}",PAPER_T5_COCO_A3_MAX_EPOCH="${MAX_EPOCH}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_coco_attn_mp_a3.sh
)"

cat <<EOF

=== Paper Table 6 COCO Attn_only a${HEADS} + MP_only a0g${HEADS} @ ep${MAX_EPOCH} submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (1–5 Attn_only a${HEADS}g0; 6–10 MP_only a0g${HEADS} GATEDGCN×${HEADS})
  Parallel:      ≤${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  max_epoch:     ${MAX_EPOCH}
  Name suffix:   ${NAME_SUFFIX:-"(none)"}
  Out:           ${GNNPLUS_OUT_DIR}
  Logs:          logs_gnnplus/sigma_T5_coco_a3_${job_id}_<TASK>.log

  W&B groups:    ${WANDB_PREFIX}_coco_Attn_only_a${HEADS}
                 ${WANDB_PREFIX}_coco_MP_only_a0g${HEADS}
  Anchor:        coco-hybrid-5b4z9l3u-a1g1-anchor.yaml
  Note:          old a2 Attn/MP jobs left alone; new groups are distinct

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${WANDB_PREFIX}_coco_Attn_only_a${HEADS} --metric best_test_perf
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${WANDB_PREFIX}_coco_MP_only_a0g${HEADS} --metric best_test_perf

  Paste JOBID into Paper_ablations.md + CLUSTER_LAUNCHES.md

EOF
