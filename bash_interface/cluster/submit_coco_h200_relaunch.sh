#!/usr/bin/env bash
# Relaunch ALL COCO paper jobs on gpu_h200 without cancelling mweber_gpu runs.
#
# Submits two arrays (same task IDs / recipes as the original grids):
#   Table 5 COCO  tasks 61-80  (4 variants × 5 seeds)
#   Table 6 COCO  tasks 51-75  (5 variants × 5 seeds)
#
# Defaults: partition=gpu_h200, ≤25 GPUs total (split T5/T6), 72h (H200 MaxTime),
# W&B name suffix=_h200, GNNPLUS_OUT_DIR → netscratch (holylabs quota).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_coco_h200_relaunch.sh
#
# Overrides:
#   COCO_H200_PARALLEL=25 COCO_H200_PARTITION=gpu_h200 COCO_H200_TIME=72:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PARTITION="${COCO_H200_PARTITION:-gpu_h200}"
# Total concurrent GPUs across both arrays (split 12+13 by default).
PARALLEL_TOTAL="${COCO_H200_PARALLEL:-25}"
T5_PARALLEL="${COCO_H200_T5_PARALLEL:-$(( PARALLEL_TOTAL / 2 ))}"
T6_PARALLEL="${COCO_H200_T6_PARALLEL:-$(( PARALLEL_TOTAL - T5_PARALLEL ))}"
# gpu_h200 MaxTime is typically 3 days — 192h fails with "time limit is invalid".
TIME="${COCO_H200_TIME:-72:00:00}"
MEM="${COCO_H200_MEM:-128GB}"
NICE="${COCO_H200_NICE:-10000}"
NAME_SUFFIX="${COCO_H200_NAME_SUFFIX:-_h200}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_h200] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

echo "=== COCO H200 relaunch (keep existing mweber jobs) ==="
echo "  partition=${PARTITION}  time=${TIME}  suffix=${NAME_SUFFIX}"
echo "  parallel total≤${PARALLEL_TOTAL}  (T5%${T5_PARALLEL} + T6%${T6_PARALLEL})"
echo "  out_dir=${GNNPLUS_OUT_DIR}"
echo

echo "--- Table 5 COCO tasks 61-80 ---"
PAPER_T5_ARRAY=61-80 \
PAPER_T5_PARALLEL="${T5_PARALLEL}" \
PAPER_T5_PARTITION="${PARTITION}" \
PAPER_T5_TIME="${TIME}" \
PAPER_T5_MEM="${MEM}" \
PAPER_T5_NICE="${NICE}" \
PAPER_T5_NAME_SUFFIX="${NAME_SUFFIX}" \
bash "${SCRIPT_DIR}/submit_paper_table5_ablations.sh"

echo
echo "--- Table 6 COCO tasks 51-75 ---"
PAPER_T6_1MP_ARRAY=51-75 \
PAPER_T6_1MP_PARALLEL="${T6_PARALLEL}" \
PAPER_T6_1MP_PARTITION="${PARTITION}" \
PAPER_T6_1MP_TIME="${TIME}" \
PAPER_T6_1MP_MEM="${MEM}" \
PAPER_T6_1MP_NICE="${NICE}" \
PAPER_T6_1MP_NAME_SUFFIX="${NAME_SUFFIX}" \
bash "${SCRIPT_DIR}/submit_paper_table6_lrgb_1mp_hetero.sh"

cat <<EOF

=== Done: paste BOTH JOBIDs into CLUSTER_LAUNCHES.md + Paper_ablations.md / Paper_table6_lrgb_1mp.md ===
  Does NOT cancel mweber jobs (34070241 / 34081524 / 34070245).
  W&B groups unchanged (paper_T5_coco_*, paper_T6_coco_*); names/tags have ${NAME_SUFFIX}.

EOF
