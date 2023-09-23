FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mmdet3d_ext
COPY README.md .
COPY pyproject.toml .

RUN pip install --no-cache-dir -U pip==23.2.1
RUN pip install --no-cache-dir mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
    && pip install --no-cache-dir -e . && rm -rf README.md pyproject.toml
