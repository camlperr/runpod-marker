FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y rsync && rm -rf /var/lib/apt/lists/*

# Force pip to prioritize hardware-accelerated PyTorch wheels
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

# Resolve dependency conflicts
RUN pip install --no-cache-dir torch torchvision torchaudio marker-pdf --extra-index-url https://download.pytorch.org/whl/cu121
