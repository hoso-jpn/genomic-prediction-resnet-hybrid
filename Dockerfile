FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# R + ビルドに必要なシステムライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# R: gblup_baseline.py の GBLUP 計算 (sommer::mmer) で使用
RUN Rscript -e "install.packages('sommer', repos='https://cloud.r-project.org', quiet=TRUE)"

WORKDIR /workspace

COPY requirements.txt .
# PyTorch Geometric のインストール
# PyTorch のバージョンに合わせた URL を指定する
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
RUN pip install torch_geometric
