FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    GPRH_ENVIRONMENT=cpu-torch-2.2.1

# uv: バージョンを固定して導入する（CIの astral-sh/setup-uv と同一バージョン）
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/

WORKDIR /workspace

# 依存関係の定義だけを先にCOPYし、ソース変更時もこのレイヤーをキャッシュさせる
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra gblup --dev --no-cache

# 検証済みソースコードとテストを含むリポジトリ全体をイメージへ配置する
# （.dockerignoreにより実データ・.venv・キャッシュ等は除外される）
COPY . .

CMD ["bash"]
