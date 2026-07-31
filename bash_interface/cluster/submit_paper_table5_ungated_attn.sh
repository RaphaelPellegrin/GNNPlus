#!/usr/bin/env bash
# Launch Table 6 Hybrid ungated-Att ablation (MP gated, attention ungated).
#
# 7 datasets × SiGMA_ungated_attn × 5 seeds = 35 jobs.
# Overrides: gnn.hybrid.gate none + gnn.hybrid.mp_gate <yaml style>.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_ungated_attn.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_UNGATED_ATTN_NUM_SEEDS:-5}"
NUM_DATASETS="${PAPER_T5_UNGATED_ATTN_NUM_DATASETS:-7}"
NUM_TASKS="${PAPER_T5_UNGATED_ATTN_NUM_TASKS:-$((NUM_DATASETS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_UNGATED_ATTN_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_UNGATED_ATTN_PARALLEL:-10}"
PARTITION="${PAPER_T5_UNGATED_ATTN_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_UNGATED_ATTN_NICE:-10000}"
MEM="${PAPER_T5_UNGATED_ATTN_MEM:-128GB}"
TIME="${PAPER_T5_UNGATED_ATTN_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T5_UNGATED_ATTN_WANDB_PREFIX:-paper_T5}"
NAME_SUFFIX="${PAPER_T5_UNGATED_ATTN_NAME_SUFFIX:-}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_ungated_attn
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_ungated_attn_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_UNGATED_ATTN_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_UNGATED_ATTN_NUM_DATASETS="${NUM_DATASETS}",PAPER_T5_UNGATED_ATTN_NUM_TASKS="${NUM_TASKS}",PAPER_T5_UNGATED_ATTN_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_UNGATED_ATTN_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T5_UNGATED_ATTN_MAX_EPOCH="${PAPER_T5_UNGATED_ATTN_MAX_EPOCH:-}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_ungated_attn.sh
)"

cat <<EOF

=== SiGMA Table 6 ungated-Att (paper_T5_*_SiGMA_ungated_attn) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ≤${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/sigma_T5_ungated_attn_${job_id}_<TASK>.log

  Variant:  SiGMA_ungated_attn = gate=none on attention; mp_gate keeps yaml style
            (opposite of SiGMA_attn_gate)

  Task map (seed = (task-1) % 5):
    1-5   peptides_func   (mp_gate=elementwise)
    6-10  peptides_struct (mp_gate=elementwise)
    11-15 voc             (mp_gate=headwise)
    16-20 coco            (mp_gate=headwise)
    21-25 mnist           (mp_gate=elementwise)
    26-30 cifar10         (mp_gate=headwise)
    31-35 pattern         (mp_gate=elementwise)

  W&B: ${WANDB_PREFIX}_<ds>_SiGMA_ungated_attn
  Paste JOBID into Paper_ablations.md + CLUSTER_LAUNCHES.md

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc

EOF
