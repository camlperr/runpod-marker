FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RAY_TMPDIR=/tmp
ENV HF_HOME=/workspace/huggingface

RUN apt-get update && apt-get install -y \
    wget build-essential gcc g++ gawk autoconf automake \
    python3-pip python3-cmarkgfm libssl-dev libxxhash-dev \
    libzstd-dev liblz4-dev libmagic1 rsync \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "torch>=2.6" torchvision torchaudio "ray<=2.47.0" python-dotenv marker-pdf --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY marker_wrapper.py /app/marker_wrapper.py

CMD ["sleep", "infinity"]
