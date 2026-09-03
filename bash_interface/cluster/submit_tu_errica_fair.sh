#!/usr/bin/env bash
# Submit Errica-fair TU campaigns (canonical / grid_select / grid_eval / sigma_*).
#
# Examples:
#   TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=gin \
#     bash bash_interface/cluster/submit_tu_errica_fair.sh
#
#   TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=gcn \
#     bash bash_interface/cluster/submit_tu_errica_fair.sh
#
#   TU_ERRICA_CAMPAIGN=grid_eval TU_ERRICA_EVAL_MODEL=gcn \
#     bash bash_interface/cluster/submit_tu_errica_fair.sh
#
#   TU_ERRICA_CAMPAIGN=sigma_grid_select \
#     bash bash_interface/cluster/submit_tu_errica_fair.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

CAMPAIGN="${TU_ERRICA_CAMPAIGN:-canonical}"
NUM_DATASETS="${TU_ERRICA_NUM_DATASETS:-7}"
NUM_FOLDS="${TU_ERRICA_NUM_FOLDS:-10}"
NUM_SEEDS="${TU_ERRICA_NUM_SEEDS:-3}"

case "${CAMPAIGN}" in
    canonical)
        NUM_TASKS="${TU_ERRICA_NUM_TASKS:-$((NUM_DATASETS * 3 * NUM_FOLDS * NUM_SEEDS))}"
        JOB_SUFFIX="canonical"
        ;;
    grid_select)
        HP_MODEL="${TU_ERRICA_GRID_MODEL:-gin}"
        GRID_SIZE=$(python3 -c "import json; from pathlib import Path; p=Path('configs/tu_errica/${HP_MODEL}_hp_grid.json'); print(len(json.load(p.open())['grid']))")
        NUM_TASKS="${TU_ERRICA_NUM_TASKS:-$((NUM_DATASETS * GRID_SIZE * NUM_FOLDS))}"
        JOB_SUFFIX="grid_select_${HP_MODEL}"
        ;;
    grid_eval|sigma_grid_eval|sigma_grid_eval_fixed8)
        NUM_TASKS="${TU_ERRICA_NUM_TASKS:-$((NUM_DATASETS * NUM_FOLDS * NUM_SEEDS))}"
        if [ "${CAMPAIGN}" = "grid_eval" ]; then
            JOB_SUFFIX="grid_eval_${TU_ERRICA_EVAL_MODEL:-gin}"
        else
            JOB_SUFFIX="${CAMPAIGN}"
        fi
        ;;
    sigma_grid_select|sigma_grid_select_fixed8)
        NUM_TASKS=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids/manifest.json'))['num_tasks'])")
        JOB_SUFFIX="${CAMPAIGN}"
        ;;
    *)
        echo "Unknown TU_ERRICA_CAMPAIGN=${CAMPAIGN}"
        exit 1
        ;;
esac

ARRAY_SPEC="${TU_ERRICA_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_ERRICA_PARALLEL:-12}"
PARTITION="${TU_ERRICA_PARTITION:-mweber_gpu}"
NICE="${TU_ERRICA_NICE:-10000}"
MEM="${TU_ERRICA_MEM:-32GB}"
TIME="${TU_ERRICA_TIME:-48:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_errica_fair] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

export_vars="ALL,TU_ERRICA_CAMPAIGN=${CAMPAIGN},TU_ERRICA_NUM_FOLDS=${NUM_FOLDS},TU_ERRICA_NUM_SEEDS=${NUM_SEEDS}"
if [ -n "${TU_ERRICA_GRID_MODEL:-}" ]; then
    export_vars="${export_vars},TU_ERRICA_GRID_MODEL=${TU_ERRICA_GRID_MODEL}"
fi
if [ -n "${TU_ERRICA_EVAL_MODEL:-}" ]; then
    export_vars="${export_vars},TU_ERRICA_EVAL_MODEL=${TU_ERRICA_EVAL_MODEL}"
fi
if [ -n "${TU_ERRICA_SELECTION_FILE:-}" ]; then
    export_vars="${export_vars},TU_ERRICA_SELECTION_FILE=${TU_ERRICA_SELECTION_FILE}"
fi

JOBID=$(sbatch --parsable \
    --job-name="tu_errica_${JOB_SUFFIX}" \
    --array="${ARRAY_SPEC}%${PARALLEL}" \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --nice="${NICE}" \
    --export="${export_vars}" \
    "${SCRIPT_DIR}/run_tu_errica_fair.sh")

echo "Submitted ${CAMPAIGN} JOBID=${JOBID}  tasks=${ARRAY_SPEC}  (${NUM_TASKS} total)"
echo "Paste into Paper_tu_errica_fair_comparison.md"
