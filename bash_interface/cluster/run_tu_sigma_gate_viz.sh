#!/usr/bin/env bash
# =============================================================================
# Train SiGMA (Xu-recipe a2g2 GIN,GIN) with checkpoints for gate dumps.
# One task per TU dataset (6): MUTAG … TRIANGLES.
#
# After training, dump with:
#   bash bash_interface/cluster/submit_dump_tu_sigma_gates.sh
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_gate_viz.sh
# =============================================================================

#SBATCH --job-name=tu_sigma_gate
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
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
seed="${GATE_VIZ_SEED:-2}"
ckpt_period="${GATE_VIZ_CKPT_PERIOD:-50}"

datasets=(mutag enzymes proteins dd nci1 triangles)
num_tasks=${#datasets[@]}

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

ds="${datasets[$((task_id - 1))]}"
cfg="configs/heterogeneity/powerful_gnns/${ds}-sigma.yaml"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/gate_viz_${ds}_sigma_powerful_seed${seed}"
else
    out_dir="results/gate_viz_${ds}_sigma_powerful_seed${seed}"
fi
mkdir -p "${out_dir}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "TU SiGMA gate-viz: ds=${ds} seed=${seed} out_dir=${out_dir}"

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    out_dir "${out_dir}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "tu_sigma_gate_viz_powerful" \
    wandb.name "${ds}_sigma_gate_viz_seed${seed}" \
    train.enable_ckpt True \
    train.ckpt_clean False \
    train.ckpt_period "${ckpt_period}" \
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
