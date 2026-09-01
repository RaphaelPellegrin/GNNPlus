#!/usr/bin/env bash
# Build per-fold SiGMA grids + manifest for Errica sigma_grid_select.
#
# Requires gin_per_fold.json from:
#   python scripts/tu_errica/aggregate_hp_selection.py --model gin
#
# Usage (login node or interactive):
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh
#
# Then submit:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

selection="${REPO_ROOT}/configs/tu_errica/selections/gin_per_fold.json"
if [ ! -f "${selection}" ]; then
    log_message "Missing ${selection}"
    log_message "Run: bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_gin"
    exit 1
fi

log_message "Generating SiGMA grids from ${selection}"
python scripts/tu_errica/generate_sigma_errica_grids.py "$@"

manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids/manifest.json"
if [ -f "${manifest}" ]; then
    num_tasks="$(python3 -c "import json; print(json.load(open('${manifest}'))['num_tasks'])")"
    log_message "Wrote ${manifest} (${num_tasks} sigma_grid_select tasks)"
    log_message "Next: bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select"
fi
