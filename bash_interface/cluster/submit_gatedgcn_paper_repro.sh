#!/usr/bin/env bash
# Submit GatedGCN+ paper baselines only (custom_gnn + layer_type: gatedgcn).
#
# Paper targets (GNN+, arXiv:2502.09263):
#   CIFAR10         test/accuracy  ~0.7006 ± 0.0033  (2 seeds)
#   peptides-struct test/mae       ~0.2431 ± 0.0020  (4 seeds)
#   peptides-func   test/ap        ~0.4263 ± 0.0057  (4 seeds)
#   COCO-SP         test/f1        ~0.3802 ± 0.0015  (2 seeds, 128GB)
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_gatedgcn_paper_repro.sh
#   bash bash_interface/cluster/submit_gatedgcn_paper_repro.sh coco cifar10

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

DEFAULT_ORDER=(cifar10 peptides-func peptides-struct coco)

gatedgcn_num_seeds() {
    case "$1" in
        cifar10|mnist|coco|voc) echo 2 ;;
        peptides-func|peptides-struct) echo 4 ;;
        *) return 1 ;;
    esac
}

gatedgcn_mem() {
    case "$1" in
        coco|voc) echo "128GB" ;;
        *) echo "64GB" ;;
    esac
}

gatedgcn_time() {
    case "$1" in
        cifar10|mnist|voc) echo "48:00:00" ;;
        peptides-func|peptides-struct|coco) echo "96:00:00" ;;
        *) return 1 ;;
    esac
}

gatedgcn_parallel() {
    case "$1" in
        cifar10|mnist|peptides-func|peptides-struct) echo 4 ;;
        coco|voc) echo 2 ;;
        *) return 1 ;;
    esac
}

# run_paper_array.sh: tasks 1..N gcn, N+1..2N gine, 2N+1..3N gatedgcn
gatedgcn_task_range() {
    local dataset="$1"
    local num_seeds start end
    num_seeds="$(gatedgcn_num_seeds "$dataset")"
    start=$((2 * num_seeds + 1))
    end=$((3 * num_seeds))
    echo "${start}-${end}"
}

submit_dataset() {
    local dataset="$1"
    local num_seeds mem time_limit parallel task_range job_name
    num_seeds="$(gatedgcn_num_seeds "$dataset")"
    mem="$(gatedgcn_mem "$dataset")"
    time_limit="$(gatedgcn_time "$dataset")"
    parallel="$(gatedgcn_parallel "$dataset")"
    task_range="$(gatedgcn_task_range "$dataset")"
    job_name="gnnplus_${dataset//-/_}_gatedgcn"

    if [ ! -f "configs/gatedgcn/${dataset}.yaml" ]; then
        echo "ERROR: missing configs/gatedgcn/${dataset}.yaml"
        return 1
    fi

    echo "→ GatedGCN+ ${dataset}: array ${task_range} (${num_seeds} seeds), mem=${mem}, time=${time_limit}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] sbatch --job-name=${job_name} --array=${task_range}%${parallel} \\"
        echo "    --time=${time_limit} --mem=${mem} \\"
        echo "    --export=ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds},ENV_NAME=gnnplus \\"
        echo "    bash_interface/cluster/run_paper_array.sh"
    else
        sbatch \
            --job-name="${job_name}" \
            --array="${task_range}%${parallel}" \
            --time="${time_limit}" \
            --mem="${mem}" \
            --export="ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds},ENV_NAME=gnnplus" \
            bash_interface/cluster/run_paper_array.sh
    fi
}

if [ "$#" -gt 0 ]; then
    REQUESTED=("$@")
else
    REQUESTED=("${DEFAULT_ORDER[@]}")
fi

for name in "${REQUESTED[@]}"; do
    if ! gatedgcn_num_seeds "$name" >/dev/null 2>&1; then
        echo "ERROR: unknown dataset '${name}'. Known: ${DEFAULT_ORDER[*]}"
        exit 1
    fi
    submit_dataset "$name"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs_gnnplus/gnnplus_<dataset>_<jobid>_<task>.log"
echo "W&B:     filter wandb.name *_gatedgcn_seed*_cluster"
echo "Configs: configs/gatedgcn/<dataset>.yaml (custom_gnn, layer_type: gatedgcn)"
