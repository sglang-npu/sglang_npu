#!/bin/bash
set -euo pipefail

CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"


# Update apt & pip sources
sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
pip config set global.index-url http://${CACHING_URL}/pypi/simple
pip config set global.trusted-host ${CACHING_URL}


# Install the required dependencies in CI.
apt update -y && apt install -y \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    ca-certificates
update-ca-certificates
python3 -m pip install --upgrade pip --no-cache-dir

### Install MemFabric
MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/sglang/${MF_WHL_NAME}"
wget "${MEMFABRIC_URL}" && pip install "./${MF_WHL_NAME}"


### Install PyTorch and PTA
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
PTA_VERSION=2.6.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install torch_npu==$PTA_VERSION --no-cache-dir


### Install Triton-Ascend
TRITON_ASCEND_VERSION=3.2.0rc2
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 --no-cache-dir
pip install triton-ascend==$TRITON_ASCEND_VERSION --no-cache-dir

pip install httpx openai einops --no-cache-dir
pip install -e "python[srt_npu]" --no-cache-dir


### Modify PyTorch TODO: to be removed later
TORCH_LOCATION=$(pip show torch | grep Location | awk -F' ' '{print $2}')
sed -i 's/from triton.runtime.autotuner import OutOfResources/from triton.runtime.errors import OutOfResources/' "${TORCH_LOCATION}/torch/_inductor/runtime/triton_heuristics.py"

transformers env
