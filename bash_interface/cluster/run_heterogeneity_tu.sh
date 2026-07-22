#!/usr/bin/env bash
# =============================================================================
# Heterogeneity profiles on TU datasets: MUTAG / ENZYMES / PROTEINS
# × {GCN, GIN, SiGMA}  (default 9 tasks).
#
# Protocol: 50/25/25 random splits, 300 epochs, val-best, ≥N test appearances.
#
# Submit:
#   bash bash_interface/cluster/submit_heterogeneity_tu.sh
# =============================================================================

#SBATCH --job-name=hetero_tu
#SBATCH --ntasks=1
#SBATCH --time=192:00:00
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
required="${HETERO_REQUIRED_TEST_APPEARANCES:-100}"
max_trials="${HETERO_MAX_TRIALS:-2000}"
seed0="${HETERO_SEED0:-0}"

# task layout: dataset_idx * 3 + model_idx + 1
datasets=(mutag enzymes proteins)
models=(gcn gin sigma)
num_models=${#models[@]}
num_datasets=${#datasets[@]}
num_tasks=$((num_datasets * num_models))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
model_idx=$((idx % num_models))
dataset_idx=$((idx / num_models))
ds="${datasets[$dataset_idx]}"
model="${models[$model_idx]}"
cfg="configs/heterogeneity/${ds}-${model}.yaml"

if [ ! -f "${cfg}" ]; then
    log_message "Missing config: ${cfg}"
    exit 1
fi

extra=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

out_dir_args=()
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}/heterogeneity"
    out_dir_args+=(--output_dir "${GNNPLUS_OUT_DIR}/heterogeneity/${ds}_${model}")
    log_message "output_dir: ${GNNPLUS_OUT_DIR}/heterogeneity/${ds}_${model}"
fi

log_message "Heterogeneity task ${task_id}/${num_tasks}: ds=${ds} model=${model} required=${required}"
log_message "cfg=${cfg}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

wandb_flag=(--wandb)
if [ "${HETERO_WANDB:-1}" = "0" ]; then
    wandb_flag=(--no-wandb)
fi

# Group: building_hetero_profile_<dataset>  (GCN/GIN/SiGMA share a group)
# Run name: <dataset>_<MODEL>
exec python scripts/heterogeneity/run_heterogeneity_profiles.py \
    --cfg "${cfg}" \
    --required_test_appearances "${required}" \
    --max_trials "${max_trials}" \
    --seed "${seed0}" \
    "${out_dir_args[@]}" \
    "${wandb_flag[@]}" \
    wandb.group "building_hetero_profile_${ds}" \
    wandb.name "${ds}_${model}" \
    "${extra[@]}"
