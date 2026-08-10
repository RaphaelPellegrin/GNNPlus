#!/usr/bin/env bash
# Submit TU attention-sink training (paper TU set × gated/ungated × SiGMA/GPS).
#
# 6 datasets × 4 variants × seed 2 = 24 jobs.
#   mutag enzymes proteins collab imdb_binary reddit_binary
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus && git pull
#
# Launch (full):
#   bash bash_interface/cluster/submit_tu_attention_sinks.sh
#
# GPS ungated only — cheapest test of ×uniform vs |V|:
#   AS_ARRAY=4,8,12,16,20,24 AS_PARALLEL=6 bash bash_interface/cluster/submit_tu_attention_sinks.sh
#
# Skip MUTAG (already done), all variants on remaining 5 ds (tasks 5-24):
#   AS_ARRAY=5-24 AS_PARALLEL=10 bash bash_interface/cluster/submit_tu_attention_sinks.sh
#
# Also dump attention on the GPU node after train (skip REDDIT unless AS_DUMP_REDDIT=1):
#   AS_DUMP_ATTN=1 bash bash_interface/cluster/submit_tu_attention_sinks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_VARIANTS="${AS_NUM_VARIANTS:-4}"
NUM_DATASETS="${AS_NUM_DATASETS:-6}"
NUM_TASKS="${AS_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS))}"
ARRAY_SPEC="${AS_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${AS_PARALLEL:-8}"
PARTITION="${AS_PARTITION:-mweber_gpu}"
NICE="${AS_NICE:-10000}"
MEM="${AS_MEM:-64GB}"
TIME="${AS_TIME:-48:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_attention_sinks] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_attention_sinks.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",AS_NUM_VARIANTS=${NUM_VARIANTS}"
export_list+=",AS_NUM_TASKS=${NUM_TASKS}"
export_list+=",AS_DUMP_ATTN=${AS_DUMP_ATTN:-0}"
export_list+=",AS_DUMP_REDDIT=${AS_DUMP_REDDIT:-0}"
export_list+=",AS_SEED=${AS_SEED:-2}"
export_list+=",AS_SINK_EVERY=${AS_SINK_EVERY:-50}"
export_list+=",AS_SINK_MAX_NODES=${AS_SINK_MAX_NODES:-512}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
if [ -n "${AS_BASE_LR:-}" ]; then
    export_list+=",AS_BASE_LR=${AS_BASE_LR}"
fi
if [ -n "${AS_BATCH:-}" ]; then
    export_list+=",AS_BATCH=${AS_BATCH}"
fi

sbatch_args=(
    --parsable
    --job-name=tu_attn_sinks
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_attn_sinks_%A_%a.log"
    --export="${export_list}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_attention_sinks.sh
)"

cat <<EOF

=== TU attention-sink campaign submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (6 ds × 4 variants = ${NUM_TASKS} max)
  Parallel:      ${PARALLEL} GPUs max
  Datasets:      MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY
  Outs:          \$GNNPLUS_OUT_DIR/tu_attention_sinks/
  W&B sinks:     log_attention_sinks=True · every AS_SINK_EVERY=${AS_SINK_EVERY:-50} epochs
  Dump attn:     AS_DUMP_ATTN=${AS_DUMP_ATTN:-0}  (REDDIT needs AS_DUMP_REDDIT=1)
  Logs:          logs_gnnplus/tu_attn_sinks_${job_id}_<TASK>.log
  Tracker:       Paper_attention_sinks.md

Task map (variant 0..3 × dataset):
  1-4   MUTAG     5-8   ENZYMES    9-12  PROTEINS
  13-16 COLLAB   17-20  IMDB-BIN   21-24 REDDIT-BIN
  GPS ungated = tasks 4,8,12,16,20,24

EOF
