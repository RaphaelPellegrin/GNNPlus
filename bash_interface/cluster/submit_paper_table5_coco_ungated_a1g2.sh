#!/usr/bin/env bash
# Launch paper Table 6 COCO a1g2 twins: Hybrid ungated + Hybrid ungated-MP (attn_gate).
# 2 variants × 5 seeds = 10 jobs. Full **300** epochs (anchor length).
#
# Distinct from main a1g1 ungated/attn_gate and from paper_T6 Homog_* groups.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_coco_ungated_a1g2.sh
#
# Overrides:
#   PAPER_T5_COCO_A1G2_PARTITION=gpu_h200 PAPER_T5_COCO_A1G2_PARALLEL=10 \
#     PAPER_T5_COCO_A1G2_TIME=72:00:00 \
#     bash bash_interface/cluster/submit_paper_table5_coco_ungated_a1g2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_COCO_A1G2_NUM_SEEDS:-5}"
NUM_TASKS=$((2 * NUM_SEEDS))
ARRAY_SPEC="${PAPER_T5_COCO_A1G2_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_COCO_A1G2_PARALLEL:-5}"
PARTITION="${PAPER_T5_COCO_A1G2_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_COCO_A1G2_NICE:-10000}"
MEM="${PAPER_T5_COCO_A1G2_MEM:-128GB}"
if [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="${PAPER_T5_COCO_A1G2_TIME:-72:00:00}"
else
    TIME="${PAPER_T5_COCO_A1G2_TIME:-192:00:00}"
fi
WANDB_PREFIX="${PAPER_T5_COCO_A1G2_WANDB_PREFIX:-paper_T5}"
NAME_SUFFIX="${PAPER_T5_COCO_A1G2_NAME_SUFFIX:-}"
MAX_EPOCH="${PAPER_T5_COCO_A1G2_MAX_EPOCH:-300}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_a1g2] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=sigma_T5_coco_a1g2
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_coco_a1g2_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_COCO_A1G2_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_COCO_A1G2_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_COCO_A1G2_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T5_COCO_A1G2_MAX_EPOCH="${MAX_EPOCH}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_coco_ungated_a1g2.sh
)"

cat <<EOF

=== Paper Table 6 COCO a1g2 ungated + attn_gate @ ep${MAX_EPOCH} submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}
                 1–5  SiGMA_ungated_a1g2   (gate=none, a1g2 GATEDGCN×2)
                 6–10 SiGMA_attn_gate_a1g2 (mp_gate=none, a1g2)
  Parallel:      ≤${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  max_epoch:     ${MAX_EPOCH}
  Out:           ${GNNPLUS_OUT_DIR}
  Logs:          logs_gnnplus/sigma_T5_coco_a1g2_${job_id}_<TASK>.log

  W&B groups:    ${WANDB_PREFIX}_coco_SiGMA_ungated_a1g2
                 ${WANDB_PREFIX}_coco_SiGMA_attn_gate_a1g2
  Anchor:        coco-hybrid-5b4z9l3u-a1g1-anchor.yaml

  Note:          Table 7 Homog_MP_ungated is the same arch as ungated_a1g2
                 but a different W&B group; this twin is for Table 6 extras.

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${WANDB_PREFIX}_coco_SiGMA_ungated_a1g2 --metric best_test_perf
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${WANDB_PREFIX}_coco_SiGMA_attn_gate_a1g2 --metric best_test_perf

  Paste JOBID into Paper_ablations.md + CLUSTER_LAUNCHES.md

EOF
