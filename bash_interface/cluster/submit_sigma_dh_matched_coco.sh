#!/usr/bin/env bash
# SiGMA d_h-matched — COCO tier (super-slow; ~56h/seed at full width on H200).
#
# 1 family × 2 LRs × 5 seeds = 10 jobs. Default parallel=2.
#
#   bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh
#
# Smoke (one seed, both LRs):
#   SIGMA_DH_MATCHED_ARRAY=1,6 SIGMA_DH_MATCHED_PARALLEL=2 \
#     bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIER=coco
NUM_FAMILIES=1
DEFAULT_PARALLEL=2
DEFAULT_MEM=128GB
DEFAULT_TIME=168:00:00

CFG_LIST=(
  configs/gated_hybrid/dh_matched/coco-a1g1-dh34.yaml
)

FAMILY_BLURB="$(cat <<'EOF'
  Task map (10 tasks: lr001 seeds 0–4, then lr01 seeds 0–4):
    1–5      COCO dh34 lr001
    6–10     COCO dh34 lr01
EOF
)"

# shellcheck source=_submit_sigma_dh_matched_common.sh
source "${SCRIPT_DIR}/_submit_sigma_dh_matched_common.sh"
