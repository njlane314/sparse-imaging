ARG TORCH_VERSION=2.1.2
ARG MINKOWSKIENGINE_TAG=v0.5.4
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=4

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel ninja \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION} \
    && python3 -m pip install numpy uproot \
    && git clone --depth 1 --branch ${MINKOWSKIENGINE_TAG} https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine \
    && cd /tmp/MinkowskiEngine \
    && python3 setup.py install --blas=openblas --force_cpu \
    && rm -rf /tmp/MinkowskiEngine

WORKDIR /workspace

COPY . /workspace

RUN python3 check_local.py

ENV PYTHONPATH=/workspace \
    ROOT_FILE=/workspace/events.root \
    TREE=events \
    SHARDS_DIR=/workspace/shards \
    CHECKPOINT_PATH=/workspace/checkpoints/checkpoint.pt \
    LOSS_LOG_PATH=/workspace/loss.tsv

CMD ["bash"]
