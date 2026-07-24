#!/usr/bin/env bash
# =============================================================================
# Last-layer activation plots on TU datasets (MUTAG / ENZYMES / PROTEINS).
#
# Array layout (3 datasets × optional seeds):
#   task_id = seed * 3 + dataset_slot
#     dataset_slot 1 = MUTAG    hetero sigma
#     dataset_slot 2 = ENZYMES  ogpkubk9 a4g4 plateau (paper-best cfg)
#     dataset_slot 3 = PROTEINS hetero sigma
#   seed = (task_id - 1) // 3
#
# Defaults:
#   ACT_ARRAY=1-3     → seed 0 only (activation figure + Acc)
#   ACT_ARRAY=1-15    → seeds 0–4 (appendix Acc mean±std + plots each)
#
# Submit:
#   bash bash_interface/cluster/submit_last_layer_activations_tu.sh
# =============================================================================

#SBATCH --job-name=tu_last_act
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_ds=3
seed=$(( (task_id - 1) / num_ds ))
ds_slot=$(( (task_id - 1) % num_ds + 1 ))

case "${ds_slot}" in
    1)
        ds=mutag
        tag=sigma
        cfg="configs/heterogeneity/mutag-sigma.yaml"
        ;;
    2)
        ds=enzymes
        tag=sigma_ogpkubk9
        cfg="configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml"
        ;;
    3)
        ds=proteins
        tag=sigma
        cfg="configs/heterogeneity/proteins-sigma.yaml"
        ;;
    *)
        log_message "ds_slot=${ds_slot} out of range (1..3)"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Missing config: ${cfg}"
    exit 1
fi

extra=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/activations/${ds}_${tag}_seed${seed}"
else
    out_dir="results/activations/${ds}_${tag}_seed${seed}"
fi
mkdir -p "${out_dir}"

log_message "last-layer act task ${task_id}: ds=${ds} tag=${tag} seed=${seed}"
log_message "cfg=${cfg} out=${out_dir}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python scripts/heterogeneity/run_last_layer_activations.py \
    --cfg "${cfg}" \
    --seed "${seed}" \
    --output_dir "${out_dir}" \
    --wandb \
    "${extra[@]}"
