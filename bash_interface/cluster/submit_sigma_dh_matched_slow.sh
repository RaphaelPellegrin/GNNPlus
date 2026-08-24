#!/usr/bin/env bash
# SiGMA d_h-matched — SLOW tier (CIFAR10, VOC).
#
# Paper-anchor wall times ~7–42h/seed on H200; keep fewer parallel jobs.
# 4 families × 2 LRs × 5 seeds = 40 jobs.
#
#   bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh
#
# Smoke:
#   SIGMA_DH_MATCHED_ARRAY=1,6,11,16 SIGMA_DH_MATCHED_PARALLEL=4 \
#     bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIER=slow
NUM_FAMILIES=4
DEFAULT_PARALLEL=8
DEFAULT_MEM=128GB
DEFAULT_TIME=120:00:00

CFG_LIST=(
  configs/gated_hybrid/dh_matched/cifar10-a8g4-dh20.yaml
  configs/gated_hybrid/dh_matched/cifar10-a8g4-dh34.yaml
  configs/gated_hybrid/dh_matched/voc-a2g2-dh15.yaml
  configs/gated_hybrid/dh_matched/voc-a2g2-h64-dh12.yaml
)

FAMILY_BLURB="$(cat <<'EOF'
  Task map (10 tasks/family: lr001 seeds 0–4, then lr01 seeds 0–4):
    1–10     CIFAR10   dh20   (≤500k)
    11–20    CIFAR10   dh34   (≤1M)
    21–30    VOC       dh15   (≤1M)
    31–40    VOC       H64/dh12 (≤500k)
EOF
)"

# shellcheck source=_submit_sigma_dh_matched_common.sh
source "${SCRIPT_DIR}/_submit_sigma_dh_matched_common.sh"
