#!/usr/bin/env bash
# =============================================================================
# Pre-download GNNPlus datasets on the login node (single-threaded).
#
# Avoids SLURM array races that corrupt MNIST/CIFAR10 zips (BadZipFile).
#
# Usage:
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   conda activate .../conda/envs/gnnplus
#   cd /n/holylabs/.../GNNPlus
#   bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10
#   bash bash_interface/cluster/prep_gnnplus_datasets.sh all
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}"
export PYTHONNOUSERSITE=1

# PyG GNNBenchmarkDataset root used by load_dataset_master (PyG-GNNBenchmarkDataset):
GNN_BENCH_ROOT="${GNNPLUS_DATASET_DIR}/GNNBenchmarkDataset"

prep_mnist() {
    echo "=== MNIST -> ${GNN_BENCH_ROOT}/MNIST ==="
    python - <<PY
import os
from GNNPlus.loader.master_loader import preformat_GNNBenchmarkDataset
root = "${GNN_BENCH_ROOT}"
preformat_GNNBenchmarkDataset(root, "MNIST")
print("MNIST OK")
PY
    unzip -t "${GNN_BENCH_ROOT}/MNIST/raw/MNIST_v2.zip" | tail -1
}

prep_cifar10() {
    echo "=== CIFAR10 -> ${GNN_BENCH_ROOT}/CIFAR10 ==="
    python - <<PY
import os
from GNNPlus.loader.master_loader import preformat_GNNBenchmarkDataset
root = "${GNN_BENCH_ROOT}"
preformat_GNNBenchmarkDataset(root, "CIFAR10")
print("CIFAR10 OK")
PY
    unzip -t "${GNN_BENCH_ROOT}/CIFAR10/raw/CIFAR10_v2.zip" | tail -1
}

fix_misplaced_cifar10() {
    # Older prep used gnnplus_datasets/CIFAR10 instead of .../GNNBenchmarkDataset/CIFAR10
    local wrong="${GNNPLUS_DATASET_DIR}/CIFAR10"
    local right="${GNN_BENCH_ROOT}/CIFAR10"
    if [ -d "${wrong}" ] && [ ! -d "${right}/processed" ]; then
        echo "Moving misplaced CIFAR10 cache -> ${right}"
        mkdir -p "${GNN_BENCH_ROOT}"
        mv "${wrong}" "${right}"
    fi
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {mnist|cifar10|all|fix-cifar10} ..."
    exit 1
fi

fix_misplaced_cifar10

for arg in "$@"; do
    case "${arg}" in
        mnist) prep_mnist ;;
        cifar10) prep_cifar10 ;;
        all)
            prep_mnist
            prep_cifar10
            ;;
        fix-cifar10) fix_misplaced_cifar10 ;;
        *)
            echo "Unknown: ${arg}"
            exit 1
            ;;
    esac
done

echo ""
echo "Dataset root: ${GNNPLUS_DATASET_DIR}"
echo "GNNBenchmark: ${GNN_BENCH_ROOT}"
ls -lh "${GNN_BENCH_ROOT}/MNIST/raw/"*.zip 2>/dev/null || true
ls -lh "${GNN_BENCH_ROOT}/CIFAR10/raw/"*.zip 2>/dev/null || true
