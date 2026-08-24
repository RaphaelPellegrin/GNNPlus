#!/usr/bin/env bash
# SiGMA d_h-matched — FAST tier (PATTERN, CLUSTER, MNIST, Pep-*, MalNet).
#
# 10 families × 2 LRs × 5 seeds = 100 jobs.
#
#   bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh
#
# Smoke (seed 0, both LRs, first 2 families):
#   SIGMA_DH_MATCHED_ARRAY=1,6,11,16 SIGMA_DH_MATCHED_PARALLEL=4 \
#     bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIER=fast
NUM_FAMILIES=10
DEFAULT_PARALLEL=20
DEFAULT_MEM=128GB
DEFAULT_TIME=48:00:00

CFG_LIST=(
  configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh16.yaml
  configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml
  configs/gated_hybrid/dh_matched/cluster-a1g1-dh36.yaml
  configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml
  configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml
  configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml
  configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh75.yaml
  configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh43.yaml
  configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh92.yaml
  configs/gated_hybrid/dh_matched/malnet-a1g1-dh57.yaml
)

FAMILY_BLURB="$(cat <<'EOF'
  Task map (10 tasks/family: lr001 seeds 0–4, then lr01 seeds 0–4):
    1–10     PATTERN   dh16
    11–20    PATTERN   dh4
    21–30    CLUSTER   dh36
    31–40    CLUSTER   dh24
    41–50    MNIST     dh37
    51–60    Pep-func  dh23
    61–70    Pep-func  dh75
    71–80    Pep-struct dh43
    81–90    Pep-struct dh92
    91–100   MalNet    dh57
EOF
)"

# shellcheck source=_submit_sigma_dh_matched_common.sh
source "${SCRIPT_DIR}/_submit_sigma_dh_matched_common.sh"
