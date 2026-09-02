#!/usr/bin/env bash
# Pull TU gate-bridge heterogeneity pickles from cluster scratch.
#
# Usage (laptop):
#   bash bash_interface/local/pull_tu_gate_bridge_hetero.sh
#
# Optional:
#   REMOTE_HOST=rpellegrinext@holylogin.rc.fas.harvard.edu
#   REMOTE_ROOT=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/heterogeneity/powerful_gnns/tu_gate_bridge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-rpellegrinext@holylogin.rc.fas.harvard.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/heterogeneity/powerful_gnns/tu_gate_bridge}"
LOCAL_ROOT="${LOCAL_ROOT:-results/heterogeneity/powerful_gnns/tu_gate_bridge}"

mkdir -p "${LOCAL_ROOT}"

echo "Pulling ${REMOTE_HOST}:${REMOTE_ROOT}/ → ${LOCAL_ROOT}/"

rsync -avz --progress \
  --include='*/' \
  --include='*.pickle' \
  --include='test_appearances.csv' \
  --include='*.png' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL_ROOT}/"

echo "Done. Local tree:"
find "${LOCAL_ROOT}" -maxdepth 2 -type f \( -name '*.pickle' -o -name 'test_appearances.csv' \) | sort
