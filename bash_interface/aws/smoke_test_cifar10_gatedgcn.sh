#!/usr/bin/env bash
# Smoke test: GatedGCN on CIFAR10 (5 epochs by default) with W&B.
#
# Docker:
#   docker run --gpus all --rm -v /data/gnnplus:/data \
#     -e WANDB_API_KEY=... gnnplus:gpu \
#     bash bash_interface/aws/smoke_test_cifar10_gatedgcn.sh
#
# Bare EC2 (repo cloned, env installed):
#   export WANDB_API_KEY=...
#   bash bash_interface/aws/smoke_test_cifar10_gatedgcn.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

REPEAT="${REPEAT:-1}"
SEED="${SEED:-0}"
MAX_EPOCH="${MAX_EPOCH:-5}"

log_message "Smoke test: gatedgcn / cifar10 / repeat=${REPEAT} / max_epoch=${MAX_EPOCH}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
extra_args+=(out_dir "${GNNPLUS_RESULTS_DIR}")

python main.py \
    --cfg configs/gatedgcn/cifar10.yaml \
    --repeat "${REPEAT}" \
    seed "${SEED}" \
    optim.max_epoch "${MAX_EPOCH}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "smoke_cifar10_gatedgcn_aws_ep${MAX_EPOCH}" \
    "${extra_args[@]}"

log_message "Smoke test finished OK"
