#!/usr/bin/env bash
# Submit fair repro arrays: standard GNN+ vs hybrid (+1 attn), all paper seeds.
#
# Usage:
#   bash bash_interface/cluster/submit_fair_repro_suite.sh
#   bash bash_interface/cluster/submit_fair_repro_suite.sh pattern cluster
#   bash bash_interface/cluster/submit_fair_repro_suite.sh --dry-run

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

DEFAULT_ORDER=(pattern cluster mal)

fair_repro_num_seeds() {
    case "$1" in
        pattern) echo 4 ;;
        cluster) echo 2 ;;
        mal) echo 5 ;;
        *) return 1 ;;
    esac
}

fair_repro_mem() {
    case "$1" in
        pattern|cluster) echo "128GB" ;;
        mal) echo "64GB" ;;
        *) return 1 ;;
    esac
}

submit_dataset() {
    local dataset="$1"
    local num_seeds mem ntasks
    num_seeds="$(fair_repro_num_seeds "$dataset")"
    mem="$(fair_repro_mem "$dataset")"
    ntasks=$((2 * num_seeds))
    local job_name="fair_repro_${dataset//-/_}"

    echo "→ ${dataset}: ${ntasks} tasks (${num_seeds} seeds × baseline+hybrid), mem=${mem}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] sbatch --job-name=${job_name} --array=1-${ntasks}%4 \\"
        echo "    --mem=${mem} --time=120:00:00 \\"
        echo "    --export=ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds},ENV_NAME=gnnplus \\"
        echo "    bash_interface/cluster/run_fair_repro_array.sh"
    else
        sbatch \
            --job-name="${job_name}" \
            --array="1-${ntasks}%4" \
            --mem="${mem}" \
            --time=120:00:00 \
            --export="ALL,DATASET=${dataset},NUM_SEEDS=${num_seeds},ENV_NAME=gnnplus" \
            bash_interface/cluster/run_fair_repro_array.sh
    fi
}

if [ "$#" -gt 0 ]; then
    REQUESTED=("$@")
else
    REQUESTED=("${DEFAULT_ORDER[@]}")
fi

for name in "${REQUESTED[@]}"; do
    if ! fair_repro_num_seeds "$name" >/dev/null 2>&1; then
        echo "ERROR: unknown dataset '${name}'. Known: ${DEFAULT_ORDER[*]}"
        exit 1
    fi
    submit_dataset "$name"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs_gnnplus/fair_repro_<dataset>_<jobid>_<task>.log"
