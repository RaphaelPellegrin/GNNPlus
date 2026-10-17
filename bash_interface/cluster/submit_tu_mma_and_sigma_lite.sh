#!/usr/bin/env bash
# Submit both TU MMA and SiGMA-lite arrays (Tab.17 + Tab.18; 120 jobs each = 240).
#
# Env overrides are forwarded:
#   TU_MMA_ARRAY / SIGMA_LITE_ARRAY, *_PARALLEL, *_PARTITION, *_NICE, *_MEM, *_TIME
#   SIGMA_LITE_GATE_ACT (default sigmoid)
#
# Example — Tab.18 only for both (tasks 61-120):
#   TU_MMA_ARRAY=61-120 SIGMA_LITE_ARRAY=61-120 \
#     bash bash_interface/cluster/submit_tu_mma_and_sigma_lite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

echo "=== Submitting MMA ==="
bash bash_interface/cluster/submit_tu_mma.sh
echo
echo "=== Submitting SiGMA-lite ==="
bash bash_interface/cluster/submit_sigma_lite_tu.sh
