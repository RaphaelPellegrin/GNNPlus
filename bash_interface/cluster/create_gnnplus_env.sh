#!/usr/bin/env bash
# =============================================================================
# One-time GNNPlus conda env on Harvard FASRC (mweber lab storage).
#
# Run from an *interactive* session (not inside an activated env):
#   salloc --partition test --nodes=1 --cpus-per-task=4 --mem=16GB --time=0-04:00:00
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/create_gnnplus_env.sh
#
# Uses README deps (PyTorch 2.2 + PyG 2.3.1) with cu121 wheels for cuda/12.9.
# =============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-gnnplus}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs}"
ENV_PREFIX="${CONDA_ENVS_PATH}/${ENV_NAME}"

echo "Creating GNNPlus env at: ${ENV_PREFIX}"

if command -v module &> /dev/null; then
    module load python/3.10.12-fasrc01 2>/dev/null || true
fi

if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
fi

if [ -d "${ENV_PREFIX}" ]; then
    echo "Env already exists: ${ENV_PREFIX}"
    echo "Delete it first or set ENV_NAME to a different name."
    exit 1
fi

if command -v mamba &> /dev/null; then
    mamba create -y -p "${ENV_PREFIX}" python=3.10 pip wheel
else
    conda create -y -p "${ENV_PREFIX}" python=3.10 pip wheel
fi

# shellcheck source=/dev/null
source activate "${ENV_PREFIX}"

pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install -r requirements-cluster.txt

PROJECT_ROOT="${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
cd "${PROJECT_ROOT}"
pip install -e . --no-deps

python - <<'PY'
import torch
import torch_geometric
import wandb
import GNNPlus
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("pyg", torch_geometric.__version__)
print("wandb", wandb.__version__)
print("GNNPlus OK")
PY

echo ""
echo "Done. Use ENV_NAME=${ENV_NAME} in SLURM scripts (default in common_env.sh)."
