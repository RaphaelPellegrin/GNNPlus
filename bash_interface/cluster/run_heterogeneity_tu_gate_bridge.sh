#!/usr/bin/env bash
# =============================================================================
# Heterogeneity profiles for TU gate–operator bridge (Tables 1–2 prep).
#
# MUTAG / ENZYMES × {GCN, GIN, SAGE, GatedGCN} = 8 tasks.
# Protocol: random 50/25/25, ≥N test appearances per graph (default 100).
#
# Pairs with SiGMA hetero gate dumps under tu_sigma_homo_hetero/ and plots in
# results/gate_viz/tu_hh_hetero/ (Appendix F).
#
# Submit:
#   bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
# =============================================================================

#SBATCH --job-name=hetero_gate_bridge
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

# Override with comma lists, e.g. HETERO_DATASETS=mutag HETERO_MODELS=gcn,gin
IFS=',' read -r -a datasets <<< "${HETERO_DATASETS:-mutag,enzymes}"
IFS=',' read -r -a models <<< "${HETERO_MODELS:-gcn,gin,sage,gatedgcn}"

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
cfg="configs/heterogeneity/powerful_gnns/${ds}-${model}.yaml"

if [ ! -f "${cfg}" ]; then
    log_message "Missing config: ${cfg}"
    exit 1
fi

extra=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

out_subdir="${HETERO_OUT_SUBDIR:-heterogeneity/powerful_gnns/tu_gate_bridge}"
out_dir_args=()
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}/${out_subdir}"
    out_dir_args+=(--output_dir "${GNNPLUS_OUT_DIR}/${out_subdir}/${ds}_${model}")
    log_message "output_dir: ${GNNPLUS_OUT_DIR}/${out_subdir}/${ds}_${model}"
fi

log_message "gate-bridge hetero task ${task_id}/${num_tasks}: ds=${ds} model=${model} required=${required}"
log_message "cfg=${cfg}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

wandb_flag=(--wandb)
if [ "${HETERO_WANDB:-1}" = "0" ]; then
    wandb_flag=(--no-wandb)
fi

wandb_group="${HETERO_WANDB_GROUP:-building_hetero_profile_${ds}_tu_gate_bridge}"

exec python scripts/heterogeneity/run_heterogeneity_profiles.py \
    --cfg "${cfg}" \
    --required_test_appearances "${required}" \
    --max_trials "${max_trials}" \
    --seed "${seed0}" \
    "${out_dir_args[@]}" \
    "${wandb_flag[@]}" \
    wandb.group "${wandb_group}" \
    wandb.name "${ds}_${model}_tu_gate_bridge" \
    "${extra[@]}"
