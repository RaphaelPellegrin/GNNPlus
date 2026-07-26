#!/usr/bin/env bash
# Launch Table 6 attention-only gating ablation (code paper_T5 groups).
#
# 4 LRGB datasets × SiGMA_attn_gate × 5 seeds = 20 jobs.
# Keeps yaml attention ``gate``; sets ``gnn.hybrid.mp_gate none``.
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_attn_gate_only.sh
#
# COCO-only on H200 (optional):
#   PAPER_T5_ATTN_GATE_ARRAY=16-20 PAPER_T5_ATTN_GATE_PARALLEL=5 \
#     PAPER_T5_ATTN_GATE_PARTITION=gpu_h200 PAPER_T5_ATTN_GATE_TIME=72:00:00 \
#     PAPER_T5_ATTN_GATE_NAME_SUFFIX=_h200 \
#     bash bash_interface/cluster/submit_paper_table5_attn_gate_only.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_ATTN_GATE_NUM_SEEDS:-5}"
NUM_DATASETS="${PAPER_T5_ATTN_GATE_NUM_DATASETS:-4}"
NUM_TASKS="${PAPER_T5_ATTN_GATE_NUM_TASKS:-$((NUM_DATASETS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_ATTN_GATE_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_ATTN_GATE_PARALLEL:-10}"
PARTITION="${PAPER_T5_ATTN_GATE_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_ATTN_GATE_NICE:-10000}"
MEM="${PAPER_T5_ATTN_GATE_MEM:-128GB}"
TIME="${PAPER_T5_ATTN_GATE_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T5_ATTN_GATE_WANDB_PREFIX:-paper_T5}"
NAME_SUFFIX="${PAPER_T5_ATTN_GATE_NAME_SUFFIX:-}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_attn_gate
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_attn_gate_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_ATTN_GATE_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_ATTN_GATE_NUM_DATASETS="${NUM_DATASETS}",PAPER_T5_ATTN_GATE_NUM_TASKS="${NUM_TASKS}",PAPER_T5_ATTN_GATE_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_ATTN_GATE_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T5_ATTN_GATE_MAX_EPOCH="${PAPER_T5_ATTN_GATE_MAX_EPOCH:-}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_attn_gate_only.sh
)"

cat <<EOF

=== SiGMA Table 6 attn-gate-only (paper_T5_*_SiGMA_attn_gate) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × SiGMA_attn_gate × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Name suffix:   ${NAME_SUFFIX:-<none>}
  Max epoch:     ${PAPER_T5_ATTN_GATE_MAX_EPOCH:-<cfg default>}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/sigma_T5_attn_gate_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B group pattern:  ${WANDB_PREFIX}_<dataset>_SiGMA_attn_gate
  W&B tags:           paper_table5, paper_table6, SiGMA_attn_gate, <dataset>, seed<k>

  Variant:
    SiGMA_attn_gate = same SiGMA heads; yaml gate on attention; mp_gate=none

  Task map (seed = (task-1) % 5):
    1-5   peptides_func
    6-10  peptides_struct
    11-15 voc
    16-20 coco

  Paste JOBID into CLUSTER_LAUNCHES.md / Paper_ablations.md

  Aggregate when done:
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5

EOF
