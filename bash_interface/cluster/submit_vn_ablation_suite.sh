#!/usr/bin/env bash
# Launch hybrid virtual-node ablations (baseline + vn 1/2/4) for all anchored datasets.
#
# Requires harvard_cluster commit with per-node label padding (96e35a8+).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull origin harvard_cluster
#   bash bash_interface/cluster/submit_vn_ablation_suite.sh          # all
#   bash bash_interface/cluster/submit_vn_ablation_suite.sh voc      # subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

submit_one() {
    local slug="$1"
    local script="${SCRIPT_DIR}/submit_${slug}_hybrid_vn_ablation.sh"
    if [ ! -f "${script}" ]; then
        echo "Skip ${slug}: missing ${script}" >&2
        return 1
    fi
    echo "======== ${slug} ========"
    bash "${script}"
    echo ""
}

DEFAULT_ORDER=(cifar10 cluster pattern voc)
if [ "$#" -eq 0 ]; then
    DATASETS=("${DEFAULT_ORDER[@]}")
else
    DATASETS=("$@")
fi

for ds in "${DATASETS[@]}"; do
    submit_one "${ds}"
done

echo "Done. Monitor: squeue -u \"\$USER\" -n cifar10_hybrid_vn,cluster_hybrid_vn,pattern_hybrid_vn,voc_hybrid_vn"
