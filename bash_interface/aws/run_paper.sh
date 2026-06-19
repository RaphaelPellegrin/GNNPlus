#!/usr/bin/env bash
# Paper-default launcher (same as local run.sh) with W&B for AWS.
# Usage:
#   bash bash_interface/aws/run_paper.sh cifar10 2
#   DATASET=cluster REPEAT=2 bash bash_interface/aws/run_paper.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

DATASET="${1:-${DATASET:-cifar10}}"
REPEAT="${2:-${REPEAT:-2}}"

log_message "Paper run: dataset=${DATASET} repeat=${REPEAT} (gcn → gine → gatedgcn)"

for gnn in gcn gine gatedgcn; do
    extra_args=()
    if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
        extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
    fi
    extra_args+=(out_dir "${GNNPLUS_RESULTS_DIR}")

    python main.py \
        --cfg "configs/${gnn}/${DATASET}.yaml" \
        --repeat "${REPEAT}" \
        seed 0 \
        wandb.use True \
        wandb.entity "${WANDB_ENTITY}" \
        wandb.project "${WANDB_PROJECT}" \
        wandb.name "aws_${DATASET}_${gnn}" \
        "${extra_args[@]}"
done

log_message "Finished paper run for ${DATASET}"
