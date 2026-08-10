#!/usr/bin/env bash
# Submit cluster-safe mechanism CSVs (summarize_nop_broadcast.py — no GraphGym).
#
# After dumps finish:
#   AS_MECH_ARRAY=1-20 AS_MECH_PARALLEL=10 \
#     bash bash_interface/cluster/submit_summarize_tu_attention_mech.sh
#
# SiGMA dumps only (post AS_DUMP_ARRAY=5,6,9,...):
#   AS_MECH_ARRAY=5,6,9,10,13,14,17,18 \
#     bash bash_interface/cluster/submit_summarize_tu_attention_mech.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus
chmod +x bash_interface/cluster/run_summarize_tu_attention_mech.sh

ARRAY_SPEC="${AS_MECH_ARRAY:-1-20}"
PARALLEL="${AS_MECH_PARALLEL:-10}"
PARTITION="${AS_MECH_PARTITION:-shared}"
MEM="${AS_MECH_MEM:-32GB}"
TIME="${AS_MECH_TIME:-04:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",AS_SEED=${AS_SEED:-2}"
export_list+=",AS_SINK_TAU=${AS_SINK_TAU:-1.5}"
export_list+=",AS_MECH_MAX_TEST=${AS_MECH_MAX_TEST:-120}"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
if [ -n "${AS_BASE_LR:-}" ]; then
    export_list+=",AS_BASE_LR=${AS_BASE_LR}"
fi
if [ -n "${AS_LR_TAG:-}" ]; then
    export_list+=",AS_LR_TAG=${AS_LR_TAG}"
fi

job_id="$(
    sbatch --parsable \
        --job-name=tu_as_mech \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --cpus-per-task=4 \
        --output="logs_gnnplus/tu_as_mech_%A_%a.log" \
        --export="${export_list}" \
        bash_interface/cluster/run_summarize_tu_attention_mech.sh
)"

cat <<EOF

=== TU AS mechanism summarize submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}
  Partition:     ${PARTITION} (CPU — no GraphGym import)
  Out CSVs:      \$GNNPLUS_OUT_DIR/tu_attention_sinks/analysis/*_mech.csv
  Logs:          logs_gnnplus/tu_as_mech_${job_id}_<TASK>.log

Requires inlined scripts/attention_sinks/summarize_nop_broadcast.py on cluster
(git pull after push). Old import of GNNPlus.attention_sink_tracking will fail.

EOF
