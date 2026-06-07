#!/usr/bin/env bash
# =============================================================================
# One-time GNNPlus conda env on Harvard FASRC (mweber lab storage).
#
# Run from an *interactive* session (not inside an activated env):
#   salloc --partition test --nodes=1 --cpus-per-task=4 --mem=16GB --time=0-04:00:00
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/clean_gnnplus_env.sh    # if reinstalling
#   bash bash_interface/cluster/create_gnnplus_env.sh
#
# Env + package cache on holylabs (NOT $HOME):
#   env:  .../conda/envs/gnnplus
#   pkgs: .../conda/pkgs
#   pip/tmp caches: .../conda/cache/
# =============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-gnnplus}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs}"
ENV_PREFIX="${CONDA_ENVS_PATH}/${ENV_NAME}"
SITE_PACKAGES="${ENV_PREFIX}/lib/python3.10/site-packages"

# Keep pip/conda temp and caches off $HOME (FASRC home quota is often small/full).
LAB_CACHE_ROOT="${GNNPLUS_CACHE_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/cache}"
export TMPDIR="${TMPDIR:-${LAB_CACHE_ROOT}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${LAB_CACHE_ROOT}/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LAB_CACHE_ROOT}/xdg}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

# Never install into ~/.local during this script.
export PYTHONNOUSERSITE=1
export PIP_USER=0
unset PIP_TARGET

if df -h "${HOME}" 2>/dev/null | tail -1 | grep -qE ' 100%| 9[0-9]%'; then
    echo "WARNING: \$HOME is nearly full ($(df -h "${HOME}" | tail -1))"
    echo "         Run: bash bash_interface/cluster/clean_gnnplus_env.sh"
fi

echo "Creating GNNPlus env at: ${ENV_PREFIX}"
echo "TMPDIR=${TMPDIR}  PIP_CACHE_DIR=${PIP_CACHE_DIR}"

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

# Leave any activated conda env before creating gnnplus.
while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do
    conda deactivate 2>/dev/null || break
done

if [ -d "${ENV_PREFIX}" ]; then
    echo "Env already exists: ${ENV_PREFIX}"
    echo "Run: bash bash_interface/cluster/clean_gnnplus_env.sh"
    exit 1
fi

if command -v mamba &> /dev/null; then
    mamba create -y -p "${ENV_PREFIX}" python=3.10 pip wheel
else
    conda create -y -p "${ENV_PREFIX}" python=3.10 pip wheel
fi

chmod -R u+w "${ENV_PREFIX}"

# shellcheck source=/dev/null
conda activate "${ENV_PREFIX}"

PYTHON="${ENV_PREFIX}/bin/python"
PIP=( "${PYTHON}" -m pip )

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: expected env python missing: ${PYTHON}"
    exit 1
fi

echo "Using python: $("${PYTHON}" -c "import sys; print(sys.executable)")"
echo "sys.prefix: $("${PYTHON}" -c "import sys; print(sys.prefix)")"

if ! "${PYTHON}" -c "import sys; assert sys.prefix == '${ENV_PREFIX}'"; then
    echo "ERROR: python prefix is not ${ENV_PREFIX}"
    exit 1
fi

mkdir -p "${SITE_PACKAGES}"
if ! touch "${SITE_PACKAGES}/.write_test" 2>/dev/null; then
    echo "ERROR: site-packages not writable: ${SITE_PACKAGES}"
    echo "       Try: chmod -R u+w ${ENV_PREFIX}"
    exit 1
fi
rm -f "${SITE_PACKAGES}/.write_test"

_pip_install() {
    echo "+ pip install $*"
    local log
    log="$(mktemp "${TMPDIR}/gnnplus_pip.XXXXXX")"
    if ! "${PIP[@]}" install "$@" 2>&1 | tee "${log}"; then
        rm -f "${log}"
        exit 1
    fi
    if grep -q "Defaulting to user installation" "${log}"; then
        echo "ERROR: pip tried to install to \$HOME/.local — aborting"
        rm -f "${log}"
        exit 1
    fi
    rm -f "${log}"
}

_verify_imports_in_env() {
    echo "Verifying imports resolve inside env (PYTHONNOUSERSITE=1)..."
    PYTHONNOUSERSITE=1 "${PYTHON}" - <<PY
import importlib.util
import sys

prefix = "${ENV_PREFIX}"
required = ["torch", "torch_geometric", "wandb"]
errors = []
for name in required:
    spec = importlib.util.find_spec(name)
    origin = spec.origin if spec and spec.origin else ""
    if not origin.startswith(prefix):
        errors.append(f"{name} -> {origin or 'NOT FOUND'}")
if errors:
    print("IMPORT CHECK FAILED (packages must live under env prefix):")
    for e in errors:
        print(" ", e)
    sys.exit(1)
print("Import paths OK under", prefix)
PY
}

"${PIP[@]}" install --upgrade pip
_pip_install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
_pip_install torch_geometric==2.3.1
_pip_install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
_pip_install -r requirements-cluster.txt

PROJECT_ROOT="${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
cd "${PROJECT_ROOT}"
_pip_install -e . --no-deps

_verify_imports_in_env

PYTHONNOUSERSITE=1 "${PYTHON}" - <<'PY'
import torch
import torch_geometric
import wandb
import GNNPlus
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("pyg", torch_geometric.__version__)
print("wandb", wandb.__version__)
print("torch file", torch.__file__)
print("GNNPlus OK")
PY

du -sh "${SITE_PACKAGES}"
echo ""
echo "Done. Use ENV_NAME=${ENV_NAME} in SLURM scripts (default in common_env.sh)."
