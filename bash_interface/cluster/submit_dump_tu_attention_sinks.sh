#!/usr/bin/env bash
# Submit offline attention dumps for existing TU AS ckpts (no retrain).
#
# Prerequisites: source ~/.gnnplus_env; export GNNPLUS_*; cd …/GNNPlus && git pull
#
# Full SiGMA gated+ungated (skip REDDIT; MUTAG already dumped but safe to re-run):
#   AS_DUMP_ARRAY=1,2,5,6,9,10,13,14,17,18 AS_DUMP_PARALLEL=8 \
#     bash bash_interface/cluster/submit_dump_tu_attention_sinks.sh
#
# SiGMA only, skip MUTAG (already have mats):
#   AS_DUMP_ARRAY=5,6,9,10,13,14,17,18 AS_DUMP_PARALLEL=8 \
#     bash bash_interface/cluster/submit_dump_tu_attention_sinks.sh
#
# After COLLAB GPS lr001 retrain:
#   AS_DUMP_ARRAY=15,16 AS_BASE_LR=0.001 AS_LR_TAG=lr001 \
#     bash bash_interface/cluster/submit_dump_tu_attention_sinks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus
chmod +x bash_interface/cluster/run_dump_tu_attention_sinks.sh

ARRAY_SPEC="${AS_DUMP_ARRAY:-5,6,9,10,13,14,17,18}"
PARALLEL="${AS_DUMP_PARALLEL:-8}"
PARTITION="${AS_DUMP_PARTITION:-mweber_gpu}"
NICE="${AS_DUMP_NICE:-10000}"
MEM="${AS_DUMP_MEM:-64GB}"
TIME="${AS_DUMP_TIME:-12:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_dump_tu_attention_sinks] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",AS_NUM_VARIANTS=${AS_NUM_VARIANTS:-4}"
export_list+=",AS_NUM_TASKS=${AS_NUM_TASKS:-24}"
export_list+=",AS_SEED=${AS_SEED:-2}"
export_list+=",AS_DUMP_BATCH=${AS_DUMP_BATCH:-8}"
export_list+=",AS_DUMP_REDDIT=${AS_DUMP_REDDIT:-0}"
# Use + not commas: Slurm --export splits on ','.
export_list+=",AS_DUMP_SPLITS=${AS_DUMP_SPLITS:-train+val+test}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
if [ -n "${AS_BASE_LR:-}" ]; then
    export_list+=",AS_BASE_LR=${AS_BASE_LR}"
fi
if [ -n "${AS_LR_TAG:-}" ]; then
    export_list+=",AS_LR_TAG=${AS_LR_TAG}"
fi

sbatch_args=(
    --parsable
    --job-name=tu_as_dump
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_as_dump_%A_%a.log"
    --export="${export_list}"
)
if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_dump_tu_attention_sinks.sh
)"

cat <<EOF

=== TU AS offline attention dump submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}
  Parallel:      ${PARALLEL}
  Out:           \$GNNPLUS_OUT_DIR/tu_attention_sinks/*/attention_matrices/
  Logs:          logs_gnnplus/tu_as_dump_${job_id}_<TASK>.log
  Default array: SiGMA × ENZYMES…IMDB (skip MUTAG+REDDIT)

Task map: 1-4 MUTAG · 5-8 ENZYMES · 9-12 PROTEINS · 13-16 COLLAB · 17-20 IMDB · 21-24 REDDIT
  +0 SiGMA gated · +1 SiGMA ungated · +2 GPS gated · +3 GPS ungated

EOF
