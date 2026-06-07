#!/usr/bin/env bash
# Remove broken gnnplus conda env and pip --user packages from failed installs.
#
# Usage:
#   bash bash_interface/cluster/clean_gnnplus_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-gnnplus}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs}"
ENV_PREFIX="${CONDA_ENVS_PATH}/${ENV_NAME}"
USER_SITE="${HOME}/.local/lib/python3.10/site-packages"

echo "Removing conda env: ${ENV_PREFIX}"
rm -rf "${ENV_PREFIX}"

if [ -d "${USER_SITE}" ]; then
    echo "Removing GNNPlus-related user-site packages under ${USER_SITE}"
    # Only packages installed by failed gnnplus pip --user attempts (not all of ~/.local).
    for pkg in \
        torch torchvision torchaudio torch_geometric pyg_lib torch_scatter \
        torch_sparse torch_cluster torch_spline_conv wandb ogb sklearn scikit_learn \
        pytorch_lightning torchmetrics tensorboardX yacs rdkit numpy pandas \
        pydantic pydantic_core aiohttp lightning_utilities protobuf \
        GNNPlus gnnplus; do
        rm -rf "${USER_SITE}/${pkg}" "${USER_SITE}/${pkg}"-*.dist-info 2>/dev/null || true
        rm -rf "${USER_SITE}/$(echo "${pkg}" | tr '-' '_')"*.dist-info 2>/dev/null || true
    done
    # Editable-install metadata
    rm -rf "${HOME}/.local/lib/python3.10/site-packages/__editable__"* 2>/dev/null || true
    rm -rf "${HOME}/.local/lib/python3.10/site-packages/gnnplus-"*.dist-info 2>/dev/null || true
fi

rm -rf "${HOME}/.local/bin/wandb" "${HOME}/.local/bin/wb" 2>/dev/null || true

echo "Clean done."
echo "  env removed: ${ENV_PREFIX}"
echo "  user-site trimmed (torch/pyg/wandb/...) under ${USER_SITE}"
du -sh "${USER_SITE}" 2>/dev/null || true
