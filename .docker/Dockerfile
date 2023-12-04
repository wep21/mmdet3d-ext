FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 python3-dev python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mmdet3d_ext
COPY README.md .
COPY pyproject.toml .

RUN pip install --no-cache-dir -U pip==23.2.1
RUN pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN pip install --no-cache-dir -e . && rm -rf README.md pyproject.toml

COPY src .

RUN python3 projects/BEVFusion/setup.py develop
