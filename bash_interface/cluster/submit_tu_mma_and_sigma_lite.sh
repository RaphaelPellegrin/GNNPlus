#!/usr/bin/env bash
# Submit both TU MMA and SiGMA-lite arrays (Tab.17 + Tab.18; 120 jobs each = 240).
#
# Default: 3 GPUs per array so the combined ceiling is 6 GPUs.
# Override with TU_MMA_PARALLEL / SIGMA_LITE_PARALLEL (or TU_COMBINED_PARALLEL
# to set both). Other env overrides are forwarded (*_ARRAY, *_PARTITION, …).
#
# Example — Tab.18 only for both (tasks 61-120):
#   TU_MMA_ARRAY=61-120 SIGMA_LITE_ARRAY=61-120 \
#     bash bash_interface/cluster/submit_tu_mma_and_sigma_lite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -n "${TU_COMBINED_PARALLEL:-}" ]; then
    export TU_MMA_PARALLEL="${TU_MMA_PARALLEL:-${TU_COMBINED_PARALLEL}}"
    export SIGMA_LITE_PARALLEL="${SIGMA_LITE_PARALLEL:-${TU_COMBINED_PARALLEL}}"
else
    # Keep total concurrent GPUs at 6 when both arrays run together.
    export TU_MMA_PARALLEL="${TU_MMA_PARALLEL:-3}"
    export SIGMA_LITE_PARALLEL="${SIGMA_LITE_PARALLEL:-3}"
fi

echo "=== Submitting MMA (parallel=${TU_MMA_PARALLEL}) ==="
bash bash_interface/cluster/submit_tu_mma.sh
echo
echo "=== Submitting SiGMA-lite (parallel=${SIGMA_LITE_PARALLEL}) ==="
bash bash_interface/cluster/submit_sigma_lite_tu.sh
