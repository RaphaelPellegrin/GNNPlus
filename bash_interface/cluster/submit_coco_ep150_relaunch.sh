#!/usr/bin/env bash
# Relaunch ALL COCO Table 5 + Table 6 jobs at max_epoch=150 (insurance run).
#
# Does NOT cancel existing 300-epoch jobs (H200 / mweber). Same variants/seeds
# as submit_coco_h200_relaunch.sh, but:
#   - optim.max_epoch=150
#   - distinct W&B groups: paper_T5_ep150_coco_*, paper_T6_ep150_coco_*
#   - name/tag suffix _ep150
#
# Defaults: mweber_gpu (H200 already busy with 300-ep twins), ≤10 GPUs total,
# 96h walltime (~40h expected for 150 ep from current ~1d20h @ ep170).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_coco_ep150_relaunch.sh
#
# Overrides:
#   COCO_EP150_PARTITION=gpu_h200 COCO_EP150_PARALLEL=20 COCO_EP150_TIME=72:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PARTITION="${COCO_EP150_PARTITION:-mweber_gpu}"
PARALLEL_TOTAL="${COCO_EP150_PARALLEL:-10}"
T5_PARALLEL="${COCO_EP150_T5_PARALLEL:-$(( PARALLEL_TOTAL / 2 ))}"
T6_PARALLEL="${COCO_EP150_T6_PARALLEL:-$(( PARALLEL_TOTAL - T5_PARALLEL ))}"
if [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="${COCO_EP150_TIME:-72:00:00}"
else
    TIME="${COCO_EP150_TIME:-96:00:00}"
fi
MEM="${COCO_EP150_MEM:-128GB}"
NICE="${COCO_EP150_NICE:-10000}"
MAX_EPOCH="${COCO_EP150_MAX_EPOCH:-150}"
NAME_SUFFIX="${COCO_EP150_NAME_SUFFIX:-_ep150}"
T5_WANDB_PREFIX="${COCO_EP150_T5_WANDB_PREFIX:-paper_T5_ep150}"
T6_WANDB_PREFIX="${COCO_EP150_T6_WANDB_PREFIX:-paper_T6_ep150}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_ep150] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

echo "=== COCO ep=${MAX_EPOCH} relaunch (keep existing 300-ep jobs) ==="
echo "  partition=${PARTITION}  time=${TIME}  suffix=${NAME_SUFFIX}"
echo "  parallel total≤${PARALLEL_TOTAL}  (T5%${T5_PARALLEL} + T6%${T6_PARALLEL})"
echo "  W&B: ${T5_WANDB_PREFIX}_* / ${T6_WANDB_PREFIX}_*"
echo "  out_dir=${GNNPLUS_OUT_DIR}"
echo

echo "--- Table 5 COCO tasks 61-80 (4×5) @ ep${MAX_EPOCH} ---"
PAPER_T5_ARRAY=61-80 \
PAPER_T5_PARALLEL="${T5_PARALLEL}" \
PAPER_T5_PARTITION="${PARTITION}" \
PAPER_T5_TIME="${TIME}" \
PAPER_T5_MEM="${MEM}" \
PAPER_T5_NICE="${NICE}" \
PAPER_T5_NAME_SUFFIX="${NAME_SUFFIX}" \
PAPER_T5_WANDB_PREFIX="${T5_WANDB_PREFIX}" \
PAPER_T5_MAX_EPOCH="${MAX_EPOCH}" \
bash "${SCRIPT_DIR}/submit_paper_table5_ablations.sh"

echo
echo "--- Table 6 COCO tasks 51-75 (5×5) @ ep${MAX_EPOCH} ---"
PAPER_T6_1MP_ARRAY=51-75 \
PAPER_T6_1MP_PARALLEL="${T6_PARALLEL}" \
PAPER_T6_1MP_PARTITION="${PARTITION}" \
PAPER_T6_1MP_TIME="${TIME}" \
PAPER_T6_1MP_MEM="${MEM}" \
PAPER_T6_1MP_NICE="${NICE}" \
PAPER_T6_1MP_NAME_SUFFIX="${NAME_SUFFIX}" \
PAPER_T6_1MP_WANDB_PREFIX="${T6_WANDB_PREFIX}" \
PAPER_T6_1MP_MAX_EPOCH="${MAX_EPOCH}" \
bash "${SCRIPT_DIR}/submit_paper_table6_lrgb_1mp_hetero.sh"

cat <<EOF

=== Done: paste BOTH JOBIDs into CLUSTER_LAUNCHES.md ===
  Existing 300-ep jobs left alone (34098505 / 34098527 / mweber twins).
  Same recipes: T5 SiGMA/ungated/Attn_only/MP_only; T6 SiGMA/Homog/Hetero ±ungated.
  Only change: optim.max_epoch=${MAX_EPOCH}
  W&B groups: ${T5_WANDB_PREFIX}_coco_* / ${T6_WANDB_PREFIX}_coco_*

  Aggregate e.g.:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${T5_WANDB_PREFIX}_coco_SiGMA --metric best_test_perf --state finished

EOF
