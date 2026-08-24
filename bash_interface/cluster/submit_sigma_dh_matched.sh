#!/usr/bin/env bash
# Convenience wrapper: print the three tier submit commands (does not launch).
#
# Prefer launching tiers separately:
#   bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh
#   bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh
#   bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh
#
# To launch all three from here:
#   SIGMA_DH_MATCHED_LAUNCH_ALL=1 bash bash_interface/cluster/submit_sigma_dh_matched.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<'EOF'
SiGMA d_h-matched is split into three tiers — run them separately:

  # Fast (~min–few hours/seed): PATTERN, CLUSTER, MNIST, Pep-*, MalNet
  # 10 families × 2 LRs × 5 seeds = 100
  bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh

  # Slow (~7–42h/seed): CIFAR10, VOC
  # 4 families × 2 LRs × 5 seeds = 40
  bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh

  # Super-slow (~56h/seed): COCO
  # 1 family × 2 LRs × 5 seeds = 10
  bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh

Docs: Paper_sigma_dh_matched.md
EOF

if [ "${SIGMA_DH_MATCHED_LAUNCH_ALL:-0}" = "1" ]; then
  bash "${SCRIPT_DIR}/submit_sigma_dh_matched_fast.sh"
  bash "${SCRIPT_DIR}/submit_sigma_dh_matched_slow.sh"
  bash "${SCRIPT_DIR}/submit_sigma_dh_matched_coco.sh"
fi
