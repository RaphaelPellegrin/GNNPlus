#!/usr/bin/env bash
# =============================================================================
# MalNet-Tiny hybrid gate-viz run: train one seed with frequent checkpoints.
#
# Dedicated out_dir (default results/gate_viz_malnet_9h3jqzkm_seed2) so ckpt is
# not overwritten by other jobs. Uses gnnplus conda env via common_env.sh.
#
# Submit:
#   bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_gate_viz.sh
# =============================================================================

#SBATCH --job-name=malnet_gate_viz
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/%x_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

seed="${GATE_VIZ_SEED:-2}"
max_epoch="${GATE_VIZ_MAX_EPOCH:-250}"
min_lr="${GATE_VIZ_MIN_LR:-1e-6}"
ckpt_period="${GATE_VIZ_CKPT_PERIOD:-50}"
out_dir="${GATE_VIZ_OUT_DIR:-results/gate_viz_malnet_9h3jqzkm_seed${seed}}"
wandb_name="${GATE_VIZ_WANDB_NAME:-malnet_gate_viz_seed${seed}}"
cfg="configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml"

mkdir -p "${out_dir}"

log_message "MalNet gate-viz: seed=${seed} max_epoch=${max_epoch} out_dir=${out_dir}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    out_dir "${out_dir}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.name "${wandb_name}" \
    train.enable_ckpt True \
    train.ckpt_clean False \
    train.ckpt_period "${ckpt_period}" \
    optim.max_epoch "${max_epoch}" \
    optim.min_lr "${min_lr}" \
    "${extra_args[@]}"
