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
#
# PROTEINS + REDDIT push (paper a2g4 anchor + LR/depth/d_h/batch variants):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids_anchor_boost
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_anchor_boost
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_anchor_boost
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_anchor_boost
#
# PROTEINS + REDDIT a1g2 micro (4-config bs×lr; GCN+GIN, no SAGE/GAT):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids_a1g2_micro
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_a1g2_micro
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_a1g2_micro
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_a1g2_micro
#
# NCI1 a1g2 micro (4-config bs×lr; GIN+SAGE):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids_a1g2_nci1_micro
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_a1g2_nci1_micro
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_a1g2_nci1_micro
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_a1g2_nci1_micro
#
# PROTEINS anchor_refine (4-config dropout×pool around a2g4 mode winner):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids_anchor_refine
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_anchor_refine
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_anchor_refine
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_anchor_refine
#
# SiGMA ungated (same fixed8 grid, gate=none):
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_fixed8_ungated
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_fixed8_ungated
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_fixed8_ungated
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
    generate_sigma_grids_anchor_boost)
        bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode anchor_boost
        ;;
    generate_sigma_grids_a1g2_micro)
        bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode a1g2_micro
        ;;
    generate_sigma_grids_a1g2_nci1_micro)
        bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode a1g2_nci1_micro
        ;;
    generate_sigma_grids_anchor_refine)
        bash bash_interface/cluster/run_generate_sigma_errica_grids.sh --mode anchor_refine
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
    sigma_grid_select_anchor_boost)
        TU_ERRICA_CAMPAIGN=sigma_grid_select_anchor_boost \
            TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
            TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_select_a1g2_micro)
        bash bash_interface/cluster/submit_tu_errica_a1g2_micro_select.sh
        ;;
    sigma_grid_select_a1g2_nci1_micro)
        bash bash_interface/cluster/submit_tu_errica_a1g2_nci1_micro_select.sh
        ;;
    sigma_grid_select_anchor_refine)
        bash bash_interface/cluster/submit_tu_errica_anchor_refine_select.sh
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
    aggregate_sigma_anchor_boost)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_anchor_boost \
            --manifest configs/tu_errica/sigma_grids_anchor_boost/manifest.json \
            --out configs/tu_errica/selections/sigma_anchor_boost_per_fold.json
        ;;
    aggregate_sigma_a1g2_micro)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_a1g2_micro \
            --model-tag SiGMA_a1g2 \
            --manifest configs/tu_errica/sigma_grids_a1g2_micro/manifest.json \
            --out configs/tu_errica/selections/sigma_a1g2_micro_per_fold.json
        ;;
    aggregate_sigma_a1g2_nci1_micro)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_a1g2_nci1_micro \
            --model-tag SiGMA_a1g2_ginsage \
            --manifest configs/tu_errica/sigma_grids_a1g2_nci1_micro/manifest.json \
            --out configs/tu_errica/selections/sigma_a1g2_nci1_micro_per_fold.json
        ;;
    aggregate_sigma_anchor_refine)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_anchor_refine \
            --manifest configs/tu_errica/sigma_grids_anchor_refine/manifest.json \
            --out configs/tu_errica/selections/sigma_anchor_refine_per_fold.json
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
    sigma_grid_eval_anchor_boost)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_anchor_boost \
            TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
            TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_anchor_boost_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_eval_a1g2_micro)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_a1g2_micro \
            TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
            TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_a1g2_micro_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_eval_a1g2_nci1_micro)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_a1g2_nci1_micro \
            TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
            TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_a1g2_nci1_micro_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_eval_anchor_refine)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_anchor_refine \
            TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
            TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_anchor_refine_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    sigma_grid_select_fixed8_ungated)
        bash bash_interface/cluster/submit_tu_errica_fixed8_ungated_select.sh
        ;;
    aggregate_sigma_fixed8_ungated)
        _run_python scripts/tu_errica/aggregate_sigma_hp_selection.py \
            --campaign sigma_grid_select_fixed8_ungated \
            --model-tag SiGMA_ungated \
            --manifest configs/tu_errica/sigma_grids/manifest.json \
            --out configs/tu_errica/selections/sigma_fixed8_ungated_per_fold.json
        ;;
    sigma_grid_eval_fixed8_ungated)
        TU_ERRICA_CAMPAIGN=sigma_grid_eval_fixed8_ungated \
            TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 TU_ERRICA_NICE=0 \
            TU_ERRICA_PARALLEL=20 TU_ERRICA_PARTITION=mweber_gpu \
            TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_fixed8_ungated_per_fold.json \
            bash bash_interface/cluster/submit_tu_errica_fair.sh
        ;;
    *)
        echo "Unknown phase: ${phase}"
        exit 1
        ;;
esac
