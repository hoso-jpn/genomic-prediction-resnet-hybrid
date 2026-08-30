# GPU実行環境の検証記録

Issue #13で整備したCUDA実行環境の状態を記録します。**実機での確認が済んでいない項目は「未実施」と明記し、実施済みとして扱いません。**

## 1. 現在の状態（2026-08-30時点）

| 項目 | 状態 |
|---|---|
| CUDA用の依存固定（`cuda/pyproject.toml`・`cuda/uv.lock`） | 作成済み・CPU lockとは独立 |
| CUDAイメージ定義（`Dockerfile.cuda`） | 作成済み |
| GPU起動経路（`docker compose --profile gpu`） | 定義済み・`docker compose config`で検証 |
| syntheticデータのGPU smoke（`tests/test_gpu_smoke.py`） | 実装済み・**GPU実機では未実施**（CPU環境ではskip） |
| CUDA要求時の明確な失敗（`--device cuda`） | CPU環境で実行し確認済み |
| 既定のCPU unit/smoke・Docker経路 | 変更なし・成功を確認 |
| 実験対象GPUのarchitecture / OS / driverの実機確認 | **未実施**（GPUを利用できる環境が未確保） |
| GPU上でのforward/backward・LOFO完走 | **未実施** |
| 実データでの本実験（#6） | **未実施**（本Issueの対象外） |

CI（GitHub Actions）にGPU runnerはありません。CIの成功はCPU経路の確認であり、GPU経路の確認ではありません。

## 2. 固定した依存関係

CPU既定環境（ルートの`pyproject.toml` / `uv.lock`）は変更していません。CUDA環境は`cuda/`配下で独立に固定します。

| 項目 | 値 |
|---|---|
| Python | 3.11 |
| PyTorch | `2.2.1+cu121`（`https://download.pytorch.org/whl/cu121`） |
| CUDA runtime / cuDNN | `cuda/uv.lock`が固定する`nvidia-*` wheel（CUDA 12.1系、cuDNN 8） |
| PyTorch Geometric | 2.7.0（純Python、CPU/CUDA共通） |
| numpy / pandas / scikit-learn / scipy | ルートと同一のピン留め |
| uv | 0.12.3（CPUイメージ・CIと同一） |

ホスト側の要件（**公式ドキュメントに基づく想定値であり、実機未確認**）:

- NVIDIA driver: CUDA 12.xのminor version compatibilityにより`>= 525.60.13`
- NVIDIA Container Toolkit（Docker経由でGPUを渡す場合）
- compute capability: PyTorch 2.2.1のcu121ビルドが対応する範囲（sm_50〜sm_90）

実機で確認したら、下記の§4へ実際の値を追記してください。手動の`pip install`で上書きした環境は再現手順として扱いません。

## 3. 手順

### 3.1 Dockerを使う場合

```bash
# イメージのbuild（CPUイメージとは別）
docker compose --profile gpu build gpu-smoke

# syntheticな3家系でのGPU smoke
# GPUが見えない場合はskipではなく失敗する（GPRH_REQUIRE_CUDA=1）
docker compose --profile gpu run --rm gpu-smoke

# 実データのResNet（CUDA）
docker compose --profile gpu run --rm resnet-gpu

# 比較用のGBLUP（同一イメージでのCPU実行）
docker compose --profile gpu run --rm gblup-cuda-env
```

### 3.2 Dockerを使わない場合

```bash
uv sync --frozen --dev --project cuda
GPRH_ENVIRONMENT=cuda-12.1-torch-2.2.1 \
  uv run --frozen --project cuda \
  python resnet_baseline.py \
  --data-dir data --output-dir resnet_results --device cuda
```

`GPRH_ENVIRONMENT`は`metadata.json`の`environment_label`へ記録され、どの固定環境で実行したかを後から識別できます（Dockerイメージでは自動的に設定されます）。

### 3.3 記録される情報

`resnet_results/artifacts/<run_id>/metadata.json`に次が入ります。

- `device_requested` / `device_resolved`（要求と実際に使われたdevice）
- `cuda_version` / `cudnn_version`
- `gpu_name` / `gpu_compute_capability`
- `cuda_driver_api_version`（driverが対応するCUDA API版）/ `nvidia_driver_version`（`nvidia-smi`から取得、取得できない場合は`null`）
- `environment_label`（`GPRH_ENVIRONMENT`）
- `library_versions`（numpy / pandas / scikit-learn / torch / torch-geometric）

## 4. 実機検証ログ

実施した場合のみ追記します。未実施の項目を埋めないでください。

| 日時(UTC) | 実行コマンド | GPU / driver / CUDA | イメージ・lock | 結果 | run_id |
|---|---|---|---|---|---|
| （未実施） | | | | | |

記入例（実施後に置き換える）:

```text
2026-09-01T00:00:00Z | docker compose --profile gpu run --rm gpu-smoke |
  <GPU名> / <driver> / <CUDA> | Dockerfile.cuda + cuda/uv.lock | 3/3 fold完走 | <run_id>
```

## 5. 既知の未解決事項

- 対象GPUのarchitecture・OS・driverの組合せが未確定のため、`cuda/uv.lock`のCUDA版（12.1）が実機に適合するかは未確認です。適合しない場合は`cuda/pyproject.toml`のindexを対応する版へ変更し、`uv lock --project cuda`で固定し直してください（ルートのlockは変更しない）。
- `Dockerfile.cuda`はCPUイメージと同じ`python:3.11-slim`をベースに、CUDA runtimeをwheelから取得します。ホストのdriverが古くwheel側のCUDA 12.1を満たさない場合は、対応する`nvidia/cuda`ベースイメージへの変更が必要になる可能性があります。
- GPU実行時の数値はCPU実行と完全一致しません（cuDNNのアルゴリズム選択・非決定的なreduction）。#6の比較では、同一splitと同一の評価尺度を使ったうえで、この差を制約として明記してください。
