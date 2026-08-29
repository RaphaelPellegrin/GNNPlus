#!/usr/bin/env bash
# Submit Errica-fair TU comparison (canonical campaign default: 630 jobs).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   python scripts/tu_errica/vendor_errica_splits.py   # once after pull
#
# Smoke (1 dataset × 1 fold × 1 seed × GIN only = 1 task):
#   TU_ERRICA_CAMPAIGN=canonical TU_ERRICA_NUM_DATASETS=1 TU_ERRICA_NUM_FOLDS=1 \
#   TU_ERRICA_NUM_SEEDS=1 TU_ERRICA_ARRAY=1 TU_ERRICA_MODELS=gin \
#     bash bash_interface/cluster/submit_tu_errica_fair.sh
#
# Full canonical (630 jobs):
#   bash bash_interface/cluster/submit_tu_errica_fair.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

CAMPAIGN="${TU_ERRICA_CAMPAIGN:-canonical}"
NUM_DATASETS="${TU_ERRICA_NUM_DATASETS:-7}"
NUM_MODELS="${TU_ERRICA_NUM_MODELS:-3}"
NUM_FOLDS="${TU_ERRICA_NUM_FOLDS:-10}"
NUM_SEEDS="${TU_ERRICA_NUM_SEEDS:-3}"

case "${CAMPAIGN}" in
    canonical|grid_eval)
        NUM_TASKS="${TU_ERRICA_NUM_TASKS:-$((NUM_DATASETS * NUM_MODELS * NUM_FOLDS * NUM_SEEDS))}"
        ;;
    grid_select)
        HP_MODEL="${TU_ERRICA_GRID_MODEL:-gin}"
        GRID_SIZE=$(python3 -c "import json; from pathlib import Path; p=Path('configs/tu_errica/${HP_MODEL}_hp_grid.json'); print(len(json.load(p.open())['grid']))")
        NUM_TASKS="${TU_ERRICA_NUM_TASKS:-$((NUM_DATASETS * GRID_SIZE * NUM_FOLDS))}"
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

sbatch_args=(
    --parsable
    --job-name="tu_errica_${CAMPAIGN}"
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --nice="${NICE}"
    --export=ALL,TU_ERRICA_CAMPAIGN="${CAMPAIGN}",TU_ERRICA_NUM_FOLDS="${NUM_FOLDS}",TU_ERRICA_NUM_SEEDS="${NUM_SEEDS}"
)

JOBID=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_tu_errica_fair.sh")
echo "Submitted ${CAMPAIGN} array JOBID=${JOBID}  tasks=${ARRAY_SPEC}  (${NUM_TASKS} total)"
echo "Paste into Paper_tu_errica_fair_comparison.md"
