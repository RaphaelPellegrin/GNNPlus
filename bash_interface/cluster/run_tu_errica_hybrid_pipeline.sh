#!/usr/bin/env bash
# Convenience launcher for Errica hybrid pipeline phases (Option 3).
#
# Phase 1 — classical model selection:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_gin
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_sage
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_gcn
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_gat
#
# Phase 2 — aggregate + build SiGMA grids:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_gin
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids
#
# Phase 3 — SiGMA select + eval:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval
#
# Phase 4 — classical eval (3 seeds):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gin
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_sage
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gcn
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gat

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

phase="${1:-}"
if [ -z "${phase}" ]; then
    echo "Usage: $0 <phase>"
    exit 1
fi

case "${phase}" in
    grid_select_gin)
        TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=gin \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    grid_select_sage)
        TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=graphsage \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    grid_select_gcn)
        TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=gcn \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    grid_select_gat)
        TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=gat \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    aggregate_gin)
        python scripts/tu_errica/aggregate_hp_selection.py --model gin
        ;;
    aggregate_sage)
        python scripts/tu_errica/aggregate_hp_selection.py --model graphsage
        ;;
    aggregate_gcn)
        python scripts/tu_errica/aggregate_hp_selection.py --model gcn
        ;;
    aggregate_gat)
        python scripts/tu_errica/aggregate_hp_selection.py --model gat
        ;;
    generate_sigma_grids)
        python scripts/tu_errica/generate_sigma_errica_grids.py
        ;;
    sigma_grid_select)
        TU_ERRICA_CAMPAIGN=sigma_grid_select TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    aggregate_sigma)
        python scripts/tu_errica/aggregate_sigma_hp_selection.py
        ;;
    grid_eval_gin)
        TU_ERRICA_CAMPAIGN=grid_eval TU_ERRICA_EVAL_MODEL=gin \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    grid_eval_sage)
        TU_ERRICA_CAMPAIGN=grid_eval TU_ERRICA_EVAL_MODEL=graphsage \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    grid_eval_gcn)
        TU_ERRICA_CAMPAIGN=grid_eval TU_ERRICA_EVAL_MODEL=gcn \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    grid_eval_gat)
        TU_ERRICA_CAMPAIGN=grid_eval TU_ERRICA_EVAL_MODEL=gat \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_eval)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    *)
        echo "Unknown phase: ${phase}"
        exit 1
        ;;
esac
