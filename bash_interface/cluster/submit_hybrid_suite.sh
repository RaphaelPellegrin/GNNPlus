#!/usr/bin/env bash
# =============================================================================
# Submit hybrid_gnn SLURM arrays (GNN+ repo, MOE_6 / tag gnnplus).
#
# Architecture: 2 attn + 2 MP heads; outer hyperparams from configs/gcn/*.yaml.
#
# Priority tiers (submit in order, or pass tier name):
#   tier1 — MNIST, CIFAR10
#   tier2 — COCO-SP, Pascal VOC
#   tier3 — peptides-func, peptides-struct
#   tier4 — ENZYMES (TUDataset; new gcn/enzymes.yaml baseline)
#   tier5 — hiv, ppa, zinc, mutag, mal, pcba, code2, cluster, pattern
#   all   — tier1 → tier5 (default)
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export ENV_NAME=gnnplus
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_hybrid_suite.sh tier1
#   bash bash_interface/cluster/submit_hybrid_suite.sh tier2 tier3
#   bash bash_interface/cluster/submit_hybrid_suite.sh mnist cifar10
#   bash bash_interface/cluster/submit_hybrid_suite.sh --dry-run all
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

TIER1=(mnist cifar10)
TIER2=(coco voc)
TIER3=(peptides-func peptides-struct)
TIER4=(enzymes)
TIER5=(hiv ppa zinc mutag mal pcba code2 cluster pattern)
ALL_ORDER=("${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}" "${TIER4[@]}" "${TIER5[@]}")

KNOWN_DATASETS=()
_add_known() {
    local d
    for d in "$@"; do
        KNOWN_DATASETS+=("$d")
    done
}
_add_known "${ALL_ORDER[@]}"

hybrid_num_seeds() {
    case "$1" in
        mnist|cifar10|coco|voc|hiv|ppa|zinc|mutag|enzymes|cluster|pattern) echo 2 ;;
        peptides-func|peptides-struct) echo 4 ;;
        mal) echo 5 ;;
        code2) echo 1 ;;
        pcba) echo 2 ;;
        *) return 1 ;;
    esac
}

hybrid_max_parallel() {
    case "$1" in
        mnist|cifar10|peptides-func|peptides-struct|enzymes|mutag|hiv|code2|mal) echo 4 ;;
        coco|voc|cluster|pattern|ppa|pcba|zinc) echo 2 ;;
        *) return 1 ;;
    esac
}

hybrid_slurm_time() {
    case "$1" in
        mnist|cifar10|voc|mutag|enzymes|hiv|code2|mal) echo "48:00:00" ;;
        cifar10) echo "48:00:00" ;;
        peptides-func|peptides-struct|coco|ppa|cluster|pattern|pcba) echo "96:00:00" ;;
        zinc) echo "192:00:00" ;;
        *) return 1 ;;
    esac
}

hybrid_mem() {
    case "$1" in
        coco|cluster|pattern|pcba) echo "128GB" ;;
        ppa) echo "96GB" ;;
        *) echo "64GB" ;;
    esac
}

resolve_tier() {
    case "$1" in
        tier1|priority1|p1) printf '%s\n' "${TIER1[@]}" ;;
        tier2|priority2|p2) printf '%s\n' "${TIER2[@]}" ;;
        tier3|priority3|p3) printf '%s\n' "${TIER3[@]}" ;;
        tier4|priority4|p4) printf '%s\n' "${TIER4[@]}" ;;
        tier5|priority5|p5|other|others) printf '%s\n' "${TIER5[@]}" ;;
        all) printf '%s\n' "${ALL_ORDER[@]}" ;;
        *) return 1 ;;
    esac
}

is_known_dataset() {
    local name="$1" d
    for d in "${KNOWN_DATASETS[@]}"; do
        if [ "$d" = "$name" ]; then
            return 0
        fi
    done
    return 1
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

REQUESTED=()
if [ "$#" -eq 0 ]; then
    REQUESTED=("${ALL_ORDER[@]}")
else
    for arg in "$@"; do
        if resolved="$(resolve_tier "$arg" 2>/dev/null)"; then
            while IFS= read -r name; do
                REQUESTED+=("$name")
            done <<< "$resolved"
        elif is_known_dataset "$arg"; then
            REQUESTED+=("$arg")
        else
            echo "ERROR: unknown argument '${arg}'."
            echo "Tiers: tier1 tier2 tier3 tier4 tier5 all"
            echo "Datasets: ${ALL_ORDER[*]}"
            exit 1
        fi
    done
fi

for name in "${REQUESTED[@]}"; do
    if ! hybrid_num_seeds "$name" >/dev/null 2>&1; then
        echo "ERROR: dataset '${name}' missing scheduler metadata."
        exit 1
    fi
    submit_dataset "$name"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs_gnnplus/hybrid_gnnplus_hybrid_<dataset>_<jobid>_<task>.log"
echo "W&B:     https://wandb.ai/${WANDB_ENTITY:-weber-geoml-harvard-university}/${WANDB_PROJECT}  (tag: gnnplus)"
