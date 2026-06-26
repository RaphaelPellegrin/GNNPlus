#!/usr/bin/env bash
# =============================================================================
# Submit GNNPlus paper-default SLURM arrays (README / ICML 2025 repeat counts).
#
# Datasets auto-download on first run into GNNPLUS_DATASET_DIR:
#   - mnist, cifar10     — PyG GNNBenchmark (data.pyg.org)
#   - peptides-func, peptides-struct — OGB-style peptides (Dropbox via loader)
#   - coco, voc          — PyG superpixels (PASCAL VOC = voc.yaml)
#
# Usage (login node, after smoke test passes):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_paper_suite.sh
#   bash bash_interface/cluster/submit_paper_suite.sh mnist coco   # subset only
#   bash bash_interface/cluster/submit_paper_suite.sh --dry-run
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

DEFAULT_ORDER=(cifar10 mnist peptides-func peptides-struct coco voc)

paper_num_seeds() {
    case "$1" in
        cifar10|mnist|coco|voc) echo 2 ;;
        peptides-func|peptides-struct) echo 4 ;;
        *) return 1 ;;
    esac
}

paper_max_parallel() {
    case "$1" in
        cifar10|mnist|peptides-func|peptides-struct) echo 6 ;;
        coco|voc) echo 4 ;;
        *) return 1 ;;
    esac
}

paper_slurm_time() {
    case "$1" in
        cifar10|mnist|voc) echo "48:00:00" ;;
        peptides-func|peptides-struct|coco) echo "96:00:00" ;;
        *) return 1 ;;
    esac
}

paper_mem() {
    case "$1" in
        coco|voc) echo "128GB" ;;
        *) echo "64GB" ;;
    esac
}

submit_dataset() {
    local dataset="$1"
    local num_seeds max_parallel time_limit mem ntasks job_name
    num_seeds="$(paper_num_seeds "$dataset")"
    max_parallel="$(paper_max_parallel "$dataset")"
    time_limit="$(paper_slurm_time "$dataset")"
    mem="$(paper_mem "$dataset")"
    ntasks=$((3 * num_seeds))
    job_name="gnnplus_${dataset//-/_}"

    if [ ! -f "configs/gcn/${dataset}.yaml" ]; then
        echo "ERROR: no configs/gcn/${dataset}.yaml"
        return 1
    fi

    echo "→ ${dataset}: ${ntasks} tasks (${num_seeds} seeds × 3 models), time=${time_limit}, mem=${mem}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] sbatch --job-name=${job_name} --array=1-${ntasks}%${max_parallel} \\"
        echo "    --time=${time_limit} --mem=${mem} --export=ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds} \\"
        echo "    bash_interface/cluster/run_paper_array.sh"
    else
        sbatch \
            --job-name="${job_name}" \
            --array="1-${ntasks}%${max_parallel}" \
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
    if ! paper_num_seeds "$name" >/dev/null 2>&1; then
        echo "ERROR: unknown dataset '${name}'."
        echo "Known: ${DEFAULT_ORDER[*]}"
        exit 1
    fi
    submit_dataset "$name"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs_gnnplus/gnnplus_<dataset>_<jobid>_<task>.log"
echo "W&B:     https://wandb.ai/weber-geoml-harvard-university/GNNPlus"
