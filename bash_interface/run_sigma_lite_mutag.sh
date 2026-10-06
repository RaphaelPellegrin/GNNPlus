#!/usr/bin/env bash
# SiGMA-lite (ungated MMA heads + one gate on the mixed update) on MUTAG.
#
# Usage (locally from anywhere; with `sbatch`, submit from the repo root):
#   bash bash_interface/run_sigma_lite_mutag.sh
#   SEED=1 WANDB=True bash bash_interface/run_sigma_lite_mutag.sh
#   GATE_ACT=gelu bash bash_interface/run_sigma_lite_mutag.sh      # Gated-GPS gate
#   bash bash_interface/run_sigma_lite_mutag.sh optim.max_epoch 5   # extra YACS overrides
#
#SBATCH --job-name=sigma_lite_mutag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets}"
WANDB="${WANDB:-False}"
GATE_ACT="${GATE_ACT:-sigmoid}"

python main.py --cfg configs/sigma_lite/mutag-sigma-lite.yaml \
  seed "${SEED}" \
  dataset.dir "${DATASET_DIR}" \
  wandb.use "${WANDB}" \
  gnn.sigma_lite.gate_act "${GATE_ACT}" \
  "$@"
