#!/bin/bash
# Install the required dependencies in CI.
sed -Ei 's@(ports|archive).ubuntu.com@cache-service.nginx-pypi-cache.svc.cluster.local:8081@g' /etc/apt/sources.list
apt update -y
apt install -y build-essential cmake python3-pip python3-dev wget net-tools zlib1g-dev lld clang software-properties-common


pip config set global.index-url http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple
pip config set global.trusted-host cache-service.nginx-pypi-cache.svc.cluster.local

python3 -m pip install --upgrade pip --no-cache-dir
pip uninstall sgl-kernel -y || true


### Download MemFabricV2
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/sglang/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
wget "${MEMFABRIC_URL}" && pip install ./mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl


### Install vLLM
VLLM_TAG=v0.8.5
git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG && \
    cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v -e . && cd ..


### Install PyTorch and PTA
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
PTA_VERSION=2.6.0rc1
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install torch_npu==$PTA_VERSION --no-cache-dir


### Install Triton-Ascend
TRITON_ASCEND_VERSION=3.2.0rc2
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 --no-cache-dir
pip install triton-ascend==$TRITON_ASCEND_VERSION --no-cache-dir


pip install -e "python[srt_npu]" --no-cache-dir


### Modify PyTorch TODO: to be removed later
TORCH_LOCATION=$(pip show torch | grep Location | awk -F' ' '{print $2}')
sed -i 's/from triton.runtime.autotuner import OutOfResources/from triton.runtime.errors import OutOfResources/' "${TORCH_LOCATION}/torch/_inductor/runtime/triton_heuristics.py"
