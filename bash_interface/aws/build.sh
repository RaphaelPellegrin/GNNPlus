#!/usr/bin/env bash
# Build the GNNPlus GPU Docker image from the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-gnnplus:gpu}"

cd "${REPO_ROOT}"
docker build -f bash_interface/aws/Dockerfile -t "${IMAGE_TAG}" .
echo "Built ${IMAGE_TAG}"
echo "Run smoke test:"
echo "  docker run --gpus all --rm -v /data/gnnplus:/data -e WANDB_API_KEY=\$WANDB_API_KEY ${IMAGE_TAG}"
