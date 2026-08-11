#!/usr/bin/env bash
# =============================================================================
# SiGMA a2g4 + Transolver++ Physics-Attention on Transolver PDE suite.
#
# Tasks 0..7 (or 1..8 with 1-indexed arrays — this script uses 0-based SLURM ids):
#   0 elasticity
#   1 plasticity
#   2 airfoil
#   3 pipe
#   4 darcy
#   5 navier_stokes
#   6 airfrans
#   7 shapenet_car
#
# Submit (max 2 GPUs):
#   bash bash_interface/cluster/submit_sigma_pde_physics.sh
#   PDE_ARRAY=0-5%2 bash ...   # standard-6 only until industrial data ready
# =============================================================================

#SBATCH --job-name=sigma_pde_phys
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
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

task_id=${SLURM_ARRAY_TASK_ID:-0}
seed="${PDE_SEED:-0}"

cfgs=(
  configs/gated_hybrid/pde_physics/elasticity-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/plasticity-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/airfoil-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/pipe-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/darcy-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/navier_stokes-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/airfrans-a2g4-physics.yaml
  configs/gated_hybrid/pde_physics/shapenet_car-a2g4-physics.yaml
)
tags=(
  elasticity
  plasticity
  airfoil
  pipe
  darcy
  navier_stokes
  airfrans
  shapenet_car
)

n=${#cfgs[@]}
if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${n}" ]; then
  log_message "task_id=${task_id} out of range (0..$((n - 1)))"
  exit 1
fi

cfg_path="${cfgs[$task_id]}"
tag="${tags[$task_id]}"
out_root="${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/sigma_pde_physics"
run_dir="${out_root}/${tag}_a2g4_physics_seed${seed}"
mkdir -p "${run_dir}"

log_message "SiGMA Physics-Attn PDE task=${task_id} tag=${tag} cfg=${cfg_path}"
log_message "out=${run_dir}"

extra_args=()
if [ -n "${PDE_MAX_EPOCH:-}" ]; then
  extra_args+=(optim.max_epoch "${PDE_MAX_EPOCH}")
fi
if [ -n "${PDE_BATCH:-}" ]; then
  extra_args+=(train.batch_size "${PDE_BATCH}")
fi

python main.py \
  --cfg "${cfg_path}" \
  out_dir "${run_dir}" \
  seed "${seed}" \
  wandb.use True \
  name_tag "a2g4_physics_seed${seed}" \
  "${extra_args[@]}"

log_message "DONE task=${task_id} tag=${tag}"
