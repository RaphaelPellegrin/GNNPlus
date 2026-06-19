# GNNPlus GPU training image (AWS EC2 / any NVIDIA host).
# Build from repo root:
#   docker build -f bash_interface/aws/Dockerfile -t gnnplus:gpu .
#
# Run (mount persistent data + results):
#   docker run --gpus all --rm \
#     -v /data/gnnplus:/data \
#     -e WANDB_API_KEY=... \
#     gnnplus:gpu bash bash_interface/aws/smoke_test_cifar10_gatedgcn.sh

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    GNNPLUS_PROJECT_ROOT=/workspace \
    GNNPLUS_DATASET_DIR=/data/datasets \
    GNNPLUS_RESULTS_DIR=/data/results \
    WANDB_DIR=/data/wandb

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-cluster.txt setup.py README.md main.py ./
COPY GNNPlus ./GNNPlus
COPY configs ./configs

RUN pip install --upgrade pip wheel \
    && pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install torch_geometric==2.3.1 \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html \
    && pip install -r requirements-cluster.txt \
    && pip install -e . --no-deps

COPY bash_interface/aws ./bash_interface/aws

RUN mkdir -p /data/datasets /data/results /data/wandb \
    && python -c "import torch; import GNNPlus; print('cuda', torch.cuda.is_available())"

VOLUME ["/data"]

ENTRYPOINT ["bash", "bash_interface/aws/entrypoint.sh"]
CMD ["bash", "bash_interface/aws/smoke_test_cifar10_gatedgcn.sh"]
