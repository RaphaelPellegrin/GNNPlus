#!/usr/bin/env bash
# Build per-fold SiGMA grids + manifest for Errica sigma_grid_select.
#
# Default: fixed 8-config SIGMA_GRID on all datasets (no GIN param ceiling).
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode full64
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode anchor_boost
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode a1g2_micro
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode a1g2_nci1_micro
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode anchor_refine
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode nci1_refine
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode native_fair
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode a0g_pnr
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode tiny_pnr
#   bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode budget_bio
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
elif [ "${MODE}" = "full64" ]; then
    log_message "Generating SiGMA grids (full64 GIN-isomorphic, 64 configs)"
elif [ "${MODE}" = "anchor_boost" ]; then
    log_message "Generating SiGMA grids (anchor_boost: PROTEINS+REDDIT, 24 configs)"
elif [ "${MODE}" = "a1g2_micro" ]; then
    log_message "Generating SiGMA grids (a1g2_micro: PROTEINS+REDDIT, 4 configs)"
elif [ "${MODE}" = "a1g2_nci1_micro" ]; then
    log_message "Generating SiGMA grids (a1g2_nci1_micro: NCI1, 4 configs, GIN+SAGE)"
elif [ "${MODE}" = "anchor_refine" ]; then
    log_message "Generating SiGMA grids (anchor_refine: PROTEINS, dropout×pool)"
elif [ "${MODE}" = "nci1_refine" ]; then
    log_message "Generating SiGMA grids (nci1_refine: NCI1, 2 configs, drop0.5×pool)"
elif [ "${MODE}" = "native_fair" ]; then
    log_message "Generating SiGMA grids (native_fair: a0g2+a1g2 P/NCI1/REDDIT, 480 select)"
elif [ "${MODE}" = "native_fair_v2" ]; then
    log_message "Generating SiGMA grids (native_fair_v2: UniGCN mixes P/NCI1/REDDIT, 180 select)"
elif [ "${MODE}" = "native_fair_v2_l4" ]; then
    log_message "Generating SiGMA grids (native_fair_v2_l4: L=4 add-on, 180 select)"
elif [ "${MODE}" = "a0g_pnr" ]; then
    log_message "Generating SiGMA grids (a0g_pnr: MP-only P/NCI1/REDDIT, 1440 select)"
elif [ "${MODE}" = "tiny_pnr" ]; then
    log_message "Generating SiGMA grids (tiny_pnr: sensible a2g4 P/NCI1/REDDIT, 120 select)"
elif [ "${MODE}" = "specialist_tiny" ]; then
    log_message "Generating SiGMA grids (specialist_tiny: GCN/SAGE a0g1+a1g1 × lr, 120 select)"
else
    log_message "Generating SiGMA grids (fixed8 SIGMA_GRID, no param ceiling)"
fi

python scripts/tu_errica/generate_sigma_errica_grids.py "$@"

if [ "${MODE}" = "full64" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_full64/manifest.json"
    next_phase="sigma_grid_select_full64"
elif [ "${MODE}" = "anchor_boost" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_anchor_boost/manifest.json"
    next_phase="sigma_grid_select_anchor_boost"
elif [ "${MODE}" = "a1g2_micro" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_a1g2_micro/manifest.json"
    next_phase="sigma_grid_select_a1g2_micro"
elif [ "${MODE}" = "a1g2_nci1_micro" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_a1g2_nci1_micro/manifest.json"
    next_phase="sigma_grid_select_a1g2_nci1_micro"
elif [ "${MODE}" = "anchor_refine" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_anchor_refine/manifest.json"
    next_phase="sigma_grid_select_anchor_refine"
elif [ "${MODE}" = "nci1_refine" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_nci1_refine/manifest.json"
    next_phase="sigma_grid_select_nci1_refine"
elif [ "${MODE}" = "native_fair" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_native_fair/manifest.json"
    next_phase="sigma_grid_select_native_fair"
elif [ "${MODE}" = "native_fair_v2" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_native_fair_v2/manifest.json"
    next_phase="sigma_grid_select_native_fair_v2"
elif [ "${MODE}" = "native_fair_v2_l4" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_native_fair_v2_l4/manifest.json"
    next_phase="sigma_grid_select_native_fair_v2_l4"
elif [ "${MODE}" = "a0g_pnr" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_a0g_pnr/manifest.json"
    next_phase="sigma_grid_select_a0g_pnr"
elif [ "${MODE}" = "tiny_pnr" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_tiny_pnr/manifest.json"
    next_phase="sigma_grid_select_tiny_pnr"
elif [ "${MODE}" = "specialist_tiny" ]; then
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids_specialist_tiny/manifest.json"
    next_phase="sigma_grid_select_specialist_tiny"
else
    manifest="${REPO_ROOT}/configs/tu_errica/sigma_grids/manifest.json"
    next_phase="sigma_grid_select"
fi
if [ -f "${manifest}" ]; then
    num_tasks="$(python3 -c "import json; print(json.load(open('${manifest}'))['num_tasks'])")"
    log_message "Wrote ${manifest} (${num_tasks} sigma_grid_select tasks)"
    log_message "Next: bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh ${next_phase}"
fi
