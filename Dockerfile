FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# R + ビルドに必要なシステムライブラリ
# ※ PyTorch は pip wheel に CUDA ライブラリを同梱しているため CUDA ベースイメージ不要
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# R: utils.py の GBLUP 計算 (sommer::mmer) で使用
RUN Rscript -e "install.packages('sommer', repos='https://cloud.r-project.org', quiet=TRUE)"

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードのみイメージに含める（データはボリュームでマウント）
COPY *.py .
