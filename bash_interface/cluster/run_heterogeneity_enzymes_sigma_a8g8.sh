#!/usr/bin/env bash
# =============================================================================
# ENZYMES SiGMA a8g8 L12 heterogeneity profile (MOE_6/7dsqq7z2 match).
#
# Distinct from the small a2g2 L4 enzymes-sigma.yaml used in the 9-job grid.
# W&B name: enzymes_sigma_a8g8  (group: building_hetero_profile_enzymes)
#
# Submit:
#   bash bash_interface/cluster/submit_heterogeneity_enzymes_sigma_a8g8.sh
# =============================================================================

#SBATCH --job-name=hetero_enz_a8g8
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

required="${HETERO_REQUIRED_TEST_APPEARANCES:-100}"
max_trials="${HETERO_MAX_TRIALS:-2000}"
seed0="${HETERO_SEED0:-0}"
cfg="configs/heterogeneity/enzymes-sigma-a8g8.yaml"
ds="enzymes"
model="sigma_a8g8"

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

log_message "ENZYMES SiGMA a8g8 hetero: required=${required} cfg=${cfg}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

wandb_flag=(--wandb)
if [ "${HETERO_WANDB:-1}" = "0" ]; then
    wandb_flag=(--no-wandb)
fi

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
