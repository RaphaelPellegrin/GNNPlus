#!/usr/bin/env bash
# Wait for an in-flight GNNPlus job, then run paper-default datasets sequentially.
#
# Usage:
#   # Queue MNIST + MUTAG after the current CIFAR10 shell (PID 9950) exits:
#   WAIT_PID=9950 DATASETS="mnist mutag" bash bash_interface/local/run_paper_queue.sh
#
#   # Run all three from scratch (no wait):
#   DATASETS="cifar10 mnist mutag" REPEAT=2 bash bash_interface/local/run_paper_queue.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

WAIT_PID="${WAIT_PID:-}"
DATASETS="${DATASETS:-mnist mutag}"
REPEAT="${REPEAT:-2}"
LOG_FILE="${LOG_FILE:-/tmp/gnnplus_paper_queue.log}"

export PYTHONNOUSERSITE=1
export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb_gnnplus_local}"
mkdir -p "${WANDB_DIR}"

if [ -n "${WAIT_PID}" ]; then
    if ! kill -0 "${WAIT_PID}" 2>/dev/null; then
        log_message "WAIT_PID=${WAIT_PID} is not running — starting queued datasets now."
    else
        log_message "Waiting for PID ${WAIT_PID} to finish before starting: ${DATASETS}"
        while kill -0 "${WAIT_PID}" 2>/dev/null; do
            sleep 60
        done
        log_message "PID ${WAIT_PID} finished."
    fi
fi

exec >>"${LOG_FILE}" 2>&1
log_message "Paper queue started: datasets=${DATASETS} repeat=${REPEAT}"

for dataset in ${DATASETS}; do
    log_message "=== dataset=${dataset} (gcn → gine → gatedgcn) ==="
    for gnn in gcn gine gatedgcn; do
        log_message "Starting ${dataset} / ${gnn}"
        python main.py \
            --cfg "configs/${gnn}/${dataset}.yaml" \
            --repeat "${REPEAT}" \
            seed 0 \
            wandb.use True \
            wandb.entity "${WANDB_ENTITY}" \
            wandb.project "${WANDB_PROJECT}" \
            wandb.name "local_${dataset}_${gnn}"
        log_message "Finished ${dataset} / ${gnn}"
    done
    log_message "=== done ${dataset} ==="
done

log_message "Paper queue complete."
