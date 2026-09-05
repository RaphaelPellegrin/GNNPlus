#!/usr/bin/env bash
# Local smoke train for GIN depth-routing (2-layer SiGMA gated, 1 seed).
#
# Example:
#   bash bash_interface/local/run_gin_depth_routing_smoke.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

dataset_parent="${GIN_DEPTH_DATASET_DIR:-${REPO_ROOT}/results/gin_routing_depth/data}"
dataset_root="${dataset_parent}/GinDepthRouting"
cfg="${1:-configs/synthetic/gin_depth_routing_toy_l2_a0g1_gated.yaml}"
seed="${GIN_DEPTH_SMOKE_SEED:-0}"
lr="${GIN_DEPTH_SMOKE_LR:-0.001}"
max_epoch="${GIN_DEPTH_SMOKE_EPOCHS:-30}"

if [ ! -f "${dataset_root}/processed/train.pt" ]; then
  python scripts/synthetic/generate_gin_depth_routing_dataset.py \
    --root "${dataset_root}" \
    --train 2000 --val 400 --test 400
fi

run_dir="results/gin_routing_depth/runs/smoke/l2_a0g1_gated_lr001_seed${seed}"
mkdir -p "${run_dir}"

echo "[smoke] cfg=${cfg} seed=${seed} lr=${lr} epochs=${max_epoch}"
echo "[smoke] dataset=${dataset_root}"
echo "[smoke] out=${run_dir}"

python main.py \
  --cfg "${cfg}" \
  --repeat 1 \
  seed "${seed}" \
  wandb.use False \
  out_dir "${run_dir}" \
  optim.base_lr "${lr}" \
  optim.max_epoch "${max_epoch}" \
  dataset.dir "${dataset_parent}" \
  gnn.layers_mp 2 \
  gnn.hybrid.log_gate_stats True \
  train.enable_ckpt True \
  train.ckpt_best True \
  train.ckpt_clean True

echo "[smoke] done → ${run_dir}"
