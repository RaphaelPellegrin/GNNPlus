#!/usr/bin/env bash
# SiGMA (gated attention + message passing) on ZINC with the Table 11 settings.
#
# Usage (locally from anywhere; with `sbatch`, submit from the repo root):
#   bash bash_interface/run_sigma_zinc.sh
#   SEED=1 WANDB=True bash bash_interface/run_sigma_zinc.sh
#   bash bash_interface/run_sigma_zinc.sh optim.max_epoch 5   # extra YACS overrides
#
#SBATCH --job-name=sigma_zinc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets}"
WANDB="${WANDB:-False}"

python main.py --cfg configs/sigma/zinc-sigma.yaml \
  seed "${SEED}" \
  dataset.dir "${DATASET_DIR}" \
  wandb.use "${WANDB}" \
  "$@"
