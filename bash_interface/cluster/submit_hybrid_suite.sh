#!/usr/bin/env bash
# =============================================================================
# Submit hybrid_gnn SLURM arrays (GNN+ repo, paper datasets, MOE_6 / tag gnnplus).
#
# Architecture: 2 attention heads + 2 MP heads (GCN+GIN or GCN+GINE).
# Outer hyperparams copied from GNN+ paper gcne configs per dataset.
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export ENV_NAME=gnnplus
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_hybrid_suite.sh
#   bash bash_interface/cluster/submit_hybrid_suite.sh mnist cifar10
#   bash bash_interface/cluster/submit_hybrid_suite.sh --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

export WANDB_PROJECT="${WANDB_PROJECT:-MOE_6}"

DEFAULT_ORDER=(mnist cifar10 peptides-func coco voc)

hybrid_num_seeds() {
    case "$1" in
        mnist|cifar10|coco|voc) echo 2 ;;
        peptides-func) echo 4 ;;
        *) return 1 ;;
    esac
}

hybrid_max_parallel() {
    case "$1" in
        mnist|cifar10|peptides-func) echo 4 ;;
        coco|voc) echo 2 ;;
        *) return 1 ;;
    esac
}

hybrid_slurm_time() {
    case "$1" in
        mnist|cifar10|voc) echo "48:00:00" ;;
        peptides-func|coco) echo "96:00:00" ;;
        *) return 1 ;;
    esac
}

hybrid_mem() {
    case "$1" in
        coco) echo "128GB" ;;
        *) echo "64GB" ;;
    esac
}

submit_dataset() {
    local dataset="$1"
    local num_seeds max_parallel time_limit mem job_name
    num_seeds="$(hybrid_num_seeds "$dataset")"
    max_parallel="$(hybrid_max_parallel "$dataset")"
    time_limit="$(hybrid_slurm_time "$dataset")"
    mem="$(hybrid_mem "$dataset")"
    job_name="gnnplus_hybrid_${dataset//-/_}"

    if [ ! -f "configs/gated_hybrid/${dataset}.yaml" ]; then
        echo "ERROR: no configs/gated_hybrid/${dataset}.yaml"
        return 1
    fi

    echo "→ hybrid ${dataset}: ${num_seeds} seeds, mem=${mem}, time=${time_limit}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] sbatch --job-name=${job_name} --array=1-${num_seeds}%${max_parallel} \\"
        echo "    --time=${time_limit} --mem=${mem} \\"
        echo "    --export=ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds},ENV_NAME=gnnplus,WANDB_PROJECT=${WANDB_PROJECT} \\"
        echo "    bash_interface/cluster/run_hybrid_array.sh"
    else
        sbatch \
            --job-name="${job_name}" \
            --array="1-${num_seeds}%${max_parallel}" \
            --time="${time_limit}" \
            --mem="${mem}" \
            --export="ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds},ENV_NAME=gnnplus,WANDB_PROJECT=${WANDB_PROJECT}" \
            bash_interface/cluster/run_hybrid_array.sh
    fi
}

if [ "$#" -gt 0 ]; then
    REQUESTED=("$@")
else
    REQUESTED=("${DEFAULT_ORDER[@]}")
fi

for name in "${REQUESTED[@]}"; do
    if ! hybrid_num_seeds "$name" >/dev/null 2>&1; then
        echo "ERROR: unknown dataset '${name}'."
        echo "Known: ${DEFAULT_ORDER[*]}"
        exit 1
    fi
    submit_dataset "$name"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs_gnnplus/hybrid_gnnplus_hybrid_<dataset>_<jobid>_<task>.log"
echo "W&B:     https://wandb.ai/${WANDB_ENTITY:-weber-geoml-harvard-university}/${WANDB_PROJECT}"
