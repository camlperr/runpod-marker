FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y rsync && rm -rf /var/lib/apt/lists/*

# Force pip to prioritize hardware-accelerated PyTorch wheels
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

# Pin the PyTorch ecosystem to exact versions matching the base image CUDA toolkit
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1

# Install Marker, allowing it to resolve against the pre-installed CUDA wheels
RUN pip install --no-cache-dir marker-pdf
