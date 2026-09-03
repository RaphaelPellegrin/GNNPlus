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
# Alternative — protocol-matched 64-config SiGMA (same search budget as GIN):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids_full64
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_full64
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_full64
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_full64
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gin
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_sage
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gcn
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gat

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

_run_python() {
    python "$@"
}

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
        _run_python scripts/tu_errica/aggregate_hp_selection.py --model gin
        ;;
    aggregate_sage)
        _run_python scripts/tu_errica/aggregate_hp_selection.py --model graphsage
        ;;
    aggregate_gcn)
        _run_python scripts/tu_errica/aggregate_hp_selection.py --model gcn
        ;;
    aggregate_gat)
        _run_python scripts/tu_errica/aggregate_hp_selection.py --model gat
        ;;
    generate_sigma_grids)
        bash bash_interface/cluster/run_generate_sigma_errica_grids.sh
        ;;
    generate_sigma_grids_full64)
        bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode full64
        ;;
    sigma_grid_select)
        # Prefer fixed8 campaign name so W&B groups do not collide with budget_bio.
        TU_ERRICA_CAMPAIGN=sigma_grid_select_fixed8 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_select_full64)
        TU_ERRICA_CAMPAIGN=sigma_grid_select_full64 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    aggregate_sigma)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_fixed8 \
            --out configs/tu_errica/selections/sigma_fixed8_per_fold.json
        ;;
    aggregate_sigma_full64)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_full64 \
            --manifest configs/tu_errica/sigma_grids_full64/manifest.json \
            --out configs/tu_errica/selections/sigma_full64_per_fold.json
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
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_fixed8 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_fixed8_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_eval_full64)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_full64 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_full64_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    *)
        echo "Unknown phase: ${phase}"
        exit 1
        ;;
esac
