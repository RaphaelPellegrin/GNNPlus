#!/usr/bin/env bash
# Fill COCO architectural Table 6 @ ep150 with a1g1 SiGMA baseline.
#
# Baseline: coco-hybrid-5b4z9l3u-a1g1-anchor.yaml (1 attn + 1 GATEDGCN).
# Matched Attn/MP: a2g0 / a0g2 (total_heads=2). Does NOT launch a3/a0g3.
#
# Gap inventory (W&B, 2026-07-28):
#   SiGMA            seed 1 crashed          → T5 task 62
#   SiGMA_ungated    5/5 done                → skip
#   Attn_only a2     seeds 0–3 still running → skip; seed 4 crashed → task 75
#   MP_only a0g2     never launched @ ep150  → T5 tasks 76-80
#   SiGMA_attn_gate  never launched @ ep150  → attn_gate tasks 16-20
#
# W&B groups: paper_T5_ep150_coco_{SiGMA,Attn_only,MP_only,SiGMA_attn_gate}
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_coco_ep150_table6_a1g1_fill.sh
#
# Optional H200:
#   COCO_EP150_A1G1_PARTITION=gpu_h200 COCO_EP150_A1G1_PARALLEL=10 \
#     COCO_EP150_A1G1_TIME=72:00:00 \
#     bash bash_interface/cluster/submit_coco_ep150_table6_a1g1_fill.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PARTITION="${COCO_EP150_A1G1_PARTITION:-mweber_gpu}"
PARALLEL="${COCO_EP150_A1G1_PARALLEL:-10}"
if [ -n "${COCO_EP150_A1G1_TIME:-}" ]; then
    TIME="${COCO_EP150_A1G1_TIME}"
elif [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="72:00:00"
else
    TIME="96:00:00"
fi
MEM="${COCO_EP150_A1G1_MEM:-128GB}"
NICE="${COCO_EP150_A1G1_NICE:-10000}"
MAX_EPOCH="${COCO_EP150_A1G1_MAX_EPOCH:-150}"
NAME_SUFFIX="${COCO_EP150_A1G1_NAME_SUFFIX:-_ep150_a1g1_fill}"
T5_WANDB_PREFIX="${COCO_EP150_A1G1_T5_PREFIX:-paper_T5_ep150}"

# T5 COCO layout (dataset_idx=3): SiGMA 61-65, ungated 66-70, Attn 71-75, MP 76-80.
T5_ARRAY="${COCO_EP150_A1G1_T5_ARRAY:-62,75,76-80}"
# attn_gate layout: coco = tasks 16-20.
ATTN_GATE_ARRAY="${COCO_EP150_A1G1_ATTN_GATE_ARRAY:-16-20}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_coco_ep150_a1g1_fill] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

echo "=== COCO Table 6 ep${MAX_EPOCH} a1g1 fill (crashed + never launched) ==="
echo "  partition=${PARTITION}  time=${TIME}  parallel≤${PARALLEL}"
echo "  W&B prefix=${T5_WANDB_PREFIX}_coco_*  suffix=${NAME_SUFFIX}"
echo "  Skip: SiGMA_ungated (done); Attn seeds 0–3 (still running)"
echo

echo "--- T5 gaps: SiGMA s1 + Attn s4 + MP_only a0g2 ×5 ---"
PAPER_T5_ARRAY="${T5_ARRAY}" \
PAPER_T5_PARALLEL="${PARALLEL}" \
PAPER_T5_PARTITION="${PARTITION}" \
PAPER_T5_TIME="${TIME}" \
PAPER_T5_MEM="${MEM}" \
PAPER_T5_NICE="${NICE}" \
PAPER_T5_NAME_SUFFIX="${NAME_SUFFIX}" \
PAPER_T5_WANDB_PREFIX="${T5_WANDB_PREFIX}" \
PAPER_T5_MAX_EPOCH="${MAX_EPOCH}" \
bash "${SCRIPT_DIR}/submit_paper_table5_ablations.sh"

echo
echo "--- SiGMA_attn_gate COCO ×5 @ ep${MAX_EPOCH} ---"
PAPER_T5_ATTN_GATE_ARRAY="${ATTN_GATE_ARRAY}" \
PAPER_T5_ATTN_GATE_PARALLEL="${PARALLEL}" \
PAPER_T5_ATTN_GATE_PARTITION="${PARTITION}" \
PAPER_T5_ATTN_GATE_TIME="${TIME}" \
PAPER_T5_ATTN_GATE_MEM="${MEM}" \
PAPER_T5_ATTN_GATE_NICE="${NICE}" \
PAPER_T5_ATTN_GATE_NAME_SUFFIX="${NAME_SUFFIX}" \
PAPER_T5_ATTN_GATE_WANDB_PREFIX="${T5_WANDB_PREFIX}" \
PAPER_T5_ATTN_GATE_MAX_EPOCH="${MAX_EPOCH}" \
bash "${SCRIPT_DIR}/submit_paper_table5_attn_gate_only.sh"

cat <<EOF

=== Done: paste BOTH JOBIDs into CLUSTER_LAUNCHES.md / Paper_ablations.md ===
  Baseline:     a1g1 (1 attn + 1 GATEDGCN)
  T5 array:     ${T5_ARRAY}
                62 = SiGMA seed1
                75 = Attn_only a2 seed4
                76-80 = MP_only a0g2 seeds 0–4
  attn_gate:    ${ATTN_GATE_ARRAY} → ${T5_WANDB_PREFIX}_coco_SiGMA_attn_gate
  Left running: paper_T5_ep150_coco_Attn_only seeds 0–3 (tasks 71–74)
  Not in this fill: Attn_a3 / MP_a0g3 (separate campaign 34869787)

  Aggregate:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${T5_WANDB_PREFIX}_coco_SiGMA --metric best_test_perf --state finished

EOF
