#!/usr/bin/env bash
# Build per-fold SiGMA grids + manifest for Errica sigma_grid_select.
#
# Default: fixed 8-config SIGMA_GRID on all datasets (no GIN param ceiling).
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh
#   # or:  .../run_generate_sigma_errica_grids.sh --mode budget_bio
#
# Legacy budget_bio still needs gin_per_fold.json:
#   python scripts/tu_errica/aggregate_hp_selection.py --model gin
#
# Then submit:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

MODE="fixed8"
for ((i = 1; i <= $#; i++)); do
    if [ "${!i}" = "--mode" ]; then
        j=$((i + 1))
        MODE="${!j:-fixed8}"
    fi
done

if [ "${MODE}" = "budget_bio" ]; then
    selection="${REPO_ROOT}/configs/tu_errica/selections/gin_per_fold.json"
    if [ ! -f "${selection}" ]; then
        log_message "Missing ${selection} (required for --mode budget_bio)"
        log_message "Run: bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_gin"
        exit 1
    fi
    log_message "Generating SiGMA grids (budget_bio) from ${selection}"
else
    log_message "Generating SiGMA grids (fixed8 SIGMA_GRID, no param ceiling)"
fi

python scripts/tu_errica/generate_sigma_errica_grids.py "$@"

manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids/manifest.json"
if [ -f "${manifest}" ]; then
    num_tasks="$(python3 -c "import json; print(json.load(open('${manifest}'))['num_tasks'])")"
    log_message "Wrote ${manifest} (${num_tasks} sigma_grid_select tasks)"
    log_message "Next: bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select"
fi
