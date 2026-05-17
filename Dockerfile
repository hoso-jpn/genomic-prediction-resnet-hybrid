FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python 3.12 (Ubuntu 24.04 標準) + R + ビルドに必要なシステムライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# R: utils.py の GBLUP 計算 (sommer::mmer) で使用
RUN Rscript -e "install.packages('sommer', repos='https://cloud.r-project.org', quiet=TRUE)"

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ソースコードのみイメージに含める（データはボリュームでマウント）
COPY *.py .
