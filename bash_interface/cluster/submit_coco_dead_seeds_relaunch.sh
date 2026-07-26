#!/usr/bin/env bash
# Relaunch ONLY COCO seeds that crashed/failed with no finished twin.
#
# Does NOT touch seeds that are finished or still running / queued.
# Uses the same W&B groups as the main paper cells (paper_T6_coco_* /
# paper_T*_ep150_coco_*) so finished retries count toward the 5-seed fill.
#
# Main paper Table 7 (old T6) gaps @ full epochs (cfg default, usually 300):
#   Homog_MP seeds 1,4          → tasks 57,60
#   Hetero_MP seeds 0,2,4       → tasks 61,63,65
#   Homog_MP_ungated seed 1     → task 67
#
# Optional ep150 SiGMA insurance gaps (set COCO_DEAD_INCLUDE_EP150=1):
#   T5 ep150 SiGMA seeds 1,2    → tasks 62,63
#   T6 ep150 SiGMA seed 4       → task 55
#
# Skipped on purpose (still running or queued elsewhere):
#   T5 Attn/MP a2, Attn a3, MP a0g3, Hetero ungated, Homog 2–3, etc.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch (main gaps only, mweber 192h):
#   bash bash_interface/cluster/submit_coco_dead_seeds_relaunch.sh
#
# Main gaps + ep150 SiGMA holes on H200, ≤10 GPUs:
#   COCO_DEAD_PARTITION=gpu_h200 COCO_DEAD_PARALLEL=10 \
#     COCO_DEAD_INCLUDE_EP150=1 \
#     bash bash_interface/cluster/submit_coco_dead_seeds_relaunch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PARTITION="${COCO_DEAD_PARTITION:-mweber_gpu}"
# Prefer long walltime: several Homog/Hetero kills were ~62h into 300-ep H200 jobs.
if [ -n "${COCO_DEAD_TIME:-}" ]; then
    TIME="${COCO_DEAD_TIME}"
elif [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="72:00:00"
else
    TIME="192:00:00"
fi
MEM="${COCO_DEAD_MEM:-128GB}"
NICE="${COCO_DEAD_NICE:-10000}"
PARALLEL="${COCO_DEAD_PARALLEL:-6}"
INCLUDE_EP150="${COCO_DEAD_INCLUDE_EP150:-0}"
if [ -n "${COCO_DEAD_NAME_SUFFIX:-}" ]; then
    NAME_SUFFIX="${COCO_DEAD_NAME_SUFFIX}"
elif [ "${PARTITION}" = "gpu_h200" ]; then
    NAME_SUFFIX="_h200_retry"
else
    NAME_SUFFIX="_retry"
fi

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_dead] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

# COCO is dataset_idx=2 in the 75-task T6 layout → tasks 51-75.
# Homog_MP=56-60, Hetero_MP=61-65, Homog_ungated=66-70.
T6_ARRAY="57,60,61,63,65,67"

echo "=== COCO dead-seed relaunch (no finished twin) ==="
echo "  partition=${PARTITION}  time=${TIME}  parallel≤${PARALLEL}"
echo "  name suffix=${NAME_SUFFIX}"
echo "  T6 main gaps array: ${T6_ARRAY}"
echo "    57,60 = Homog_MP seeds 1,4"
echo "    61,63,65 = Hetero_MP seeds 0,2,4"
echo "    67 = Homog_MP_ungated seed 1"
echo

echo "--- Paper Table 7 (old T6) COCO gaps @ full epochs ---"
PAPER_T6_1MP_ARRAY="${T6_ARRAY}" \
PAPER_T6_1MP_PARALLEL="${PARALLEL}" \
PAPER_T6_1MP_PARTITION="${PARTITION}" \
PAPER_T6_1MP_TIME="${TIME}" \
PAPER_T6_1MP_MEM="${MEM}" \
PAPER_T6_1MP_NICE="${NICE}" \
PAPER_T6_1MP_NAME_SUFFIX="${NAME_SUFFIX}" \
PAPER_T6_1MP_WANDB_PREFIX="paper_T6" \
bash "${SCRIPT_DIR}/submit_paper_table6_lrgb_1mp_hetero.sh"

if [ "${INCLUDE_EP150}" = "1" ]; then
    echo
    echo "--- ep150 SiGMA insurance gaps only ---"
    # T5 COCO SiGMA = tasks 61-65 → dead seeds 1,2 → 62,63
    PAPER_T5_ARRAY="62,63" \
    PAPER_T5_PARALLEL="${PARALLEL}" \
    PAPER_T5_PARTITION="${PARTITION}" \
    PAPER_T5_TIME="${TIME}" \
    PAPER_T5_MEM="${MEM}" \
    PAPER_T5_NICE="${NICE}" \
    PAPER_T5_NAME_SUFFIX="_ep150${NAME_SUFFIX}" \
    PAPER_T5_WANDB_PREFIX="paper_T5_ep150" \
    PAPER_T5_MAX_EPOCH="150" \
    bash "${SCRIPT_DIR}/submit_paper_table5_ablations.sh"

    # T6 COCO SiGMA seed 4 → task 55
    PAPER_T6_1MP_ARRAY="55" \
    PAPER_T6_1MP_PARALLEL="${PARALLEL}" \
    PAPER_T6_1MP_PARTITION="${PARTITION}" \
    PAPER_T6_1MP_TIME="${TIME}" \
    PAPER_T6_1MP_MEM="${MEM}" \
    PAPER_T6_1MP_NICE="${NICE}" \
    PAPER_T6_1MP_NAME_SUFFIX="_ep150${NAME_SUFFIX}" \
    PAPER_T6_1MP_WANDB_PREFIX="paper_T6_ep150" \
    PAPER_T6_1MP_MAX_EPOCH="150" \
    bash "${SCRIPT_DIR}/submit_paper_table6_lrgb_1mp_hetero.sh"
fi

cat <<EOF

=== Done: paste JOBID(s) into CLUSTER_LAUNCHES.md ===
  Main gaps (always): Homog_MP 1/4, Hetero_MP 0/2/4, Homog_ungated 1
  ep150 SiGMA gaps:   $([ "${INCLUDE_EP150}" = "1" ] && echo "T5 seeds 1,2 + T6 seed 4" || echo "skipped (COCO_DEAD_INCLUDE_EP150=0)")

  Not relaunched (still running or queued):
    T5 Attn/MP a2, Attn a3, MP a0g3, Hetero_MP_ungated,
    Homog_MP 2–3, Homog_ungated 0/2/3/4, ep150 Attn/MP/Hetero queues

EOF
