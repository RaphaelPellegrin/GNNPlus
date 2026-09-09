#!/usr/bin/env bash
# Submit vanilla full-attn (a4g0) attention-sink runs on all 6 TU AS datasets.
#
# Partition: gpu_h200, ≤5 GPUs, 72h MaxTime.
# Sink PNGs on disk + W&B every max_epoch/4 (quarters mode; 1000 → 0/250/500/750/999).
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
# Tasks (AS_NUM_VARIANTS=5): 5,10,15,20,25,30
#
# Usage (cluster, after git pull):
#   bash bash_interface/cluster/submit_tu_attention_sinks_vanilla_h200.sh
#
# Env knobs:
#   AS_PARALLEL / AS_ARRAY / AS_TIME / AS_MEM / AS_NICE / AS_DUMP_ATTN / AS_DUMP_REDDIT
#   AS_SINK_MAX_NODES (default 512) AS_SEED (default 2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

export AS_NUM_VARIANTS="${AS_NUM_VARIANTS:-5}"
export AS_NUM_TASKS="${AS_NUM_TASKS:-30}"
export AS_ARRAY="${AS_ARRAY:-5,10,15,20,25,30}"
export AS_PARALLEL="${AS_PARALLEL:-5}"
export AS_PARTITION="${AS_PARTITION:-gpu_h200}"
export AS_TIME="${AS_TIME:-72:00:00}"
export AS_MEM="${AS_MEM:-128GB}"
export AS_NICE="${AS_NICE:-0}"
export AS_SEED="${AS_SEED:-2}"

# Mid-train sink panels → W&B + disk PNGs under run_dir/attention_sinks/epXXXX/
export AS_LOG_SINKS="${AS_LOG_SINKS:-1}"
export AS_SINK_SAVE_DISK="${AS_SINK_SAVE_DISK:-1}"
export AS_SINK_EPOCHS="${AS_SINK_EPOCHS:-quarters}"
export AS_SINK_MAX_NODES="${AS_SINK_MAX_NODES:-512}"
# Full attention_matrices dump still opt-in (REDDIT needs AS_DUMP_REDDIT=1).
export AS_DUMP_ATTN="${AS_DUMP_ATTN:-0}"
export AS_DUMP_REDDIT="${AS_DUMP_REDDIT:-0}"

LOGDIR="${GNNPLUS_OUT_DIR}/logs_tu_attention_sinks_vanilla"
mkdir -p "${LOGDIR}" logs_gnnplus

echo "[vanilla_h200] array=${AS_ARRAY}%${AS_PARALLEL} partition=${AS_PARTITION} time=${AS_TIME}"
echo "[vanilla_h200] sinks: LOG=${AS_LOG_SINKS} SAVE_DISK=${AS_SINK_SAVE_DISK} EPOCHS=${AS_SINK_EPOCHS}"
echo "[vanilla_h200] logs → ${LOGDIR}/  (also logs_gnnplus/)"

# Reuse generic submit; override log path via sbatch is awkward, so call sbatch here.
NUM_VARIANTS="${AS_NUM_VARIANTS}"
NUM_TASKS="${AS_NUM_TASKS}"
ARRAY_SPEC="${AS_ARRAY}"
PARALLEL="${AS_PARALLEL}"
PARTITION="${AS_PARTITION}"
NICE="${AS_NICE}"
MEM="${AS_MEM}"
TIME="${AS_TIME}"

chmod +x bash_interface/cluster/run_tu_attention_sinks.sh

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",AS_NUM_VARIANTS=${NUM_VARIANTS}"
export_list+=",AS_NUM_TASKS=${NUM_TASKS}"
export_list+=",AS_DUMP_ATTN=${AS_DUMP_ATTN}"
export_list+=",AS_DUMP_REDDIT=${AS_DUMP_REDDIT}"
export_list+=",AS_SEED=${AS_SEED}"
export_list+=",AS_LOG_SINKS=${AS_LOG_SINKS}"
export_list+=",AS_SINK_SAVE_DISK=${AS_SINK_SAVE_DISK}"
export_list+=",AS_SINK_EPOCHS=${AS_SINK_EPOCHS}"
export_list+=",AS_SINK_EVERY=${AS_SINK_EVERY:-50}"
export_list+=",AS_SINK_MAX_NODES=${AS_SINK_MAX_NODES}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"

job_id="$(
  sbatch --parsable \
    --job-name=tu_as_vanilla_a4g0 \
    --array="${ARRAY_SPEC}%${PARALLEL}" \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --nice="${NICE}" \
    --gpus=1 \
    --output="${LOGDIR}/vanilla_%A_%a.log" \
    --export="${export_list}" \
    bash_interface/cluster/run_tu_attention_sinks.sh
)"

cat <<EOF

=== Vanilla a4g0 attention-sinks (gpu_h200) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}  · parallel ≤${PARALLEL}
  Tasks:         ${ARRAY_SPEC}  (MUTAG/ENZYMES/PROTEINS/COLLAB/IMDB/REDDIT)
  Sink schedule: ${AS_SINK_EPOCHS}  (PNG every max_epoch/4 + ep0 + last)
  Disk PNGs:     \$GNNPLUS_OUT_DIR/tu_attention_sinks/<ds>_vanilla_full_attn_*/attention_sinks/
  Logs:          ${LOGDIR}/vanilla_${job_id}_<TASK>.log
  Tracker:       Paper_attention_sinks.md / CLUSTER_LAUNCHES.md

Monitor: squeue -j ${job_id}; sacct -j ${job_id} -X --format=State -n | sort | uniq -c

EOF
