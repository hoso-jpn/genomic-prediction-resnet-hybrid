# GPU実行環境の検証記録

Issue #13で整備したCUDA実行環境の状態を記録します。**実機での確認が済んでいない項目は「未実施」と明記し、実施済みとして扱いません。**

## 0. 現在の状態（2026-09-15時点、GPU実機検証後）

2026-09-15に対象GPUホスト（seedcore-01）でCUDAイメージのbuildとsynthetic GPU smokeを実行し、成功しました。実測値・コマンド・結果は§7に記録しています。§1〜§6の2026-08-31時点の記録は、当時の状態を示す履歴として変更せずに残しています。

| 項目 | 状態（2026-09-15） | 2026-08-31からの変化 |
|---|---|---|
| 対象GPUホストの構成確認（読み取りのみ） | **実施済み**（§7.1） | 再計測。containerdの保存先と`/`の空き容量が変化 |
| CUDAイメージ定義（`Dockerfile.cuda`）のbuild | **実施済み・成功**（§7.3） | 未実施 → 成功 |
| GPU起動経路（`docker compose --profile gpu`） | **起動済み**（`config --quiet`・`build gpu-smoke`・`run --rm gpu-smoke`） | 未実施 → 成功 |
| syntheticデータのGPU smoke（`tests/test_gpu_smoke.py`） | **GPU実機で成功**（`test_resnet_cuda_smoke` PASSED、skipではない） | 未実施 → 成功 |
| GPU上でのforward/backward・3家系LOFO完走・OOF/run artifacts保存 | **実施済み**（§7.4、§7.5） | 未実施 → 成功 |
| 実際に選択されたdeviceの記録 | **実機で確認**（`device_requested=cuda` / `device_resolved=cuda`） | 未実施 → 確認 |
| CUDA要求時の明確な失敗（GPU不在） | CPU環境でのみ確認済み（GPUホストでは対象テストが仕様どおりskip） | 変化なし |
| GPUと互換性の無いビルドでの失敗（実機） | **未実施** | 変化なし |
| ホスト上の非Docker経路（`uv run --project cuda`） | **未実施**（ホストにuv未導入） | 変化なし |
| 実データでの本実験（#6） | **未実施**（本Issueの対象外） | 変化なし |

CI（GitHub Actions）にGPU runnerはありません。§7の成功はGPUホストでの手動実行によるもので、CIの結果ではありません。

## 1. 過去の状態（2026-08-31時点、履歴として保持）

| 項目 | 状態 |
|---|---|
| 対象GPUホストの構成確認（読み取りのみ） | **実施済み**（§2の実測値） |
| CUDA用の依存固定（`cuda/pyproject.toml`・`cuda/uv.lock`） | 作成済み・CPU lockとは独立。対象GPUのcompute capabilityに合わせて選定（§3） |
| CUDA wheelがsm_120を含むことの確認 | **実施済み**（CPU側で`torch._C._cuda_getArchFlags()`を確認、§5.1） |
| CUDA環境でのテストスイート（CPU実行） | **実施済み**（112 passed / 1 skipped、§5.1） |
| CUDAイメージ定義（`Dockerfile.cuda`） | 作成済み・**イメージbuildは未実施** |
| GPU起動経路（`docker compose --profile gpu`） | 定義済み・`docker compose config`で検証・**起動は未実施** |
| syntheticデータのGPU smoke（`tests/test_gpu_smoke.py`） | 実装済み・**GPU実機では未実施**（CPU環境ではskip） |
| CUDA要求時の明確な失敗（`--device cuda`） | CPU環境で実行し確認済み |
| 既定のCPU unit/smoke・Docker経路 | 変更なし・成功を確認 |
| GPU上でのforward/backward・LOFO完走 | **未実施** |
| 実データでの本実験（#6） | **未実施**（本Issueの対象外） |

CI（GitHub Actions）にGPU runnerはありません。CIの成功はCPU経路の確認であり、GPU経路の確認ではありません。

## 2. 対象GPUホストの実測値（seedcore-01、2026-08-31、読み取りのみ）

> 2026-08-31時点の履歴です。2026-09-15の再計測値と差分は§7.1・§7.2を参照してください（containerdの保存先は`/data/containerd`に変わり、`/`の空きは130Gです）。本節の「実機実行の前に解決が必要な点」は2026-09-15時点では解消しています。

過去の構成情報ではなく、当日にホストへ接続して読み取った値です。設定変更・サービス停止・GPUを使用する実行は行っていません。

| 項目 | 実測値 |
|---|---|
| OS | Ubuntu 24.04.4 LTS（kernel 6.8.0-136-generic） |
| GPU | NVIDIA GeForce RTX 5090（1基） |
| compute capability | 12.0（sm_120） |
| NVIDIA driver | 595.84 |
| `nvidia-smi`のCUDA表示 | 13.2（**driverが対応する最大のCUDA版**を示す表示。ホストにCUDA toolkitが導入されていることや、実行時に使われるCUDA runtimeの版を意味しない。実際のruntimeは`cuda/uv.lock`が固定する`nvidia-*` wheelが提供し、`torch.version.cuda`で確認できる） |
| VRAM | 32,607 MiB 中 6,579 MiB 使用・**25,527 MiB 空き**（GPU使用率4%） |
| GPU上で稼働中のプロセス | `llama-server`（6,412 MiB を使用中） |
| Docker | 29.7.2（`nvidia` runtime登録済み、default runtimeは`runc`） |
| NVIDIA Container Toolkit | 1.20.0-1 |
| ホストのPython / uv | Python 3.12.3 / uv 未導入 |

### ディスクの実測と内訳（2026-08-31、読み取りのみ）

| マウント | デバイス | 容量 | 使用 | 空き | 使用率 |
|---|---|---:|---:|---:|---:|
| `/` | `/dev/nvme0n1p2` (ext4) | 183 GiB | 173 GiB | **238 MiB** | **100%** |
| `/home` | `/dev/nvme0n1p3` (ext4) | 3.5 TiB | 880 GiB | 2.4 TiB | 27% |
| `/data` | `/dev/nvme1n1p1` (ext4) | 3.6 TiB | 115 GiB | 3.3 TiB | 4% |

inodeは`/`で12%使用（枯渇していません）。

**Dockerの保存先は1か所ではありません。**

| 項目 | 実測値 | 置き場所 |
|---|---|---|
| `Docker Root Dir` | `/data/docker` | `/data`（空き3.3 TiB） |
| Storage Driver | `overlayfs`（`driver-type: io.containerd.snapshotter.v1`） | — |
| containerdの起動引数 | `/usr/bin/containerd`（`--root`指定なし） | 既定の`/var/lib/containerd`＝**`/`（満杯のFS）** |
| `/etc/containerd/config.toml` | 存在するが`root` / `state`の指定行なし | 既定値が有効 |
| `dockerd`の起動引数 | `-H fd:// --containerd=/run/containerd/containerd.sock` | 外部containerdを使用 |

`docker system df`（論理サイズ。containerd image store使用時も同じ値を返します）:

| 種別 | 総数 | 使用中 | サイズ | 解放可能（docker表示） |
|---|---:|---:|---:|---:|
| Images | 30 | 16 | 115.5 GB | 21.5 GB |
| Containers | 23 | 0 | 1.65 GB | 1.65 GB |
| Local Volumes | 3 | 3 | 1.121 GB | 0 B |
| Build Cache | 34 | 0 | 35.01 GB | 215.3 MB |

大きいイメージ上位: `vllm/vllm-openai:latest` 29.9 GB、`vllm-gemma4-base:2026-06-29` 29.9 GB、`vllm/vllm-openai:qwen38-x86_64-cu130` 23 GB、`local/flash-next:pilot-v1-*` 15.9 GB、`nvidia/cuda:12.8.1-devel-ubuntu24.04` 14.6 GB。

**`/`の内訳（一般ユーザー権限で読める範囲）**: `/usr` 22 GiB、`/var` 5.0 GiB（うち`/var/log` 1.2 GiB、`journald` 1,021 MiB）、`/tmp` 2.3 GiB（内訳: `/tmp/issue30_hc_benchmark` 2.3 GiB、所有者は作業ユーザー、更新 2026-08-24）、`/opt` 679 MiB、`/boot` 391 MiB。読める範囲の合計は約46 GiBで、**使用中173 GiBとの差およそ127 GiBは、権限がなく`du`で読めないディレクトリにあります**（`/var/lib/containerd`は`root:root` 0700、`/var/lib/docker`は0710）。

**推定（未確定）**: Dockerは containerd snapshotter を使用しており、イメージ層は`Docker Root Dir`ではなく**containerdのroot（既定の`/var/lib/containerd`＝`/`）**に置かれます。読めない約127 GiBは、`docker system df`が示すイメージ115.5 GBとおおむね整合します。**これは権限の制約による推定であり、確定していません。** 確定させるには、ホスト側で次を実行してください（読み取りのみ）。

```bash
sudo du -sh /var/lib/containerd /var/lib/docker /data/docker
sudo du -xh --max-depth=1 /var/lib | sort -h | tail
```

### 容量に関する候補の整理（**本Issueでは一切実行していません**）

以下は「候補の一覧」であり、**解放見込み量の見積もりではありません**。`docker system df`が示す数値は種別ごとの論理サイズで、層の共有分を含みます。また、この数値は**保存先のファイルシステムを区別しません**。ルートFS（`/`）で実際に回復する容量は、共有層の重複排除・containerdの実際の保存先・削除対象の実体位置に依存するため、**現時点では未確認です**。

| 対象 | dockerが表示する容量 | dockerが表示する解放可能 | 保存先FS | 用途・状態 | 再生成 | 留意点 |
|---|---:|---:|---|---|---|---|
| 停止（exited）コンテナ 23個 | 1.65 GB | 1.65 GB | 未確認（containerd store と推定） | すべて`exited`。**停止中であることは不要であることを意味しません**。保持の要否は所有者の判断 | 不可（ログ・停止時の状態は失われる） | 判断前に用途の確認が必要 |
| Build Cache 34件 | 35.01 GB | 215.3 MB | 未確認（同上） | 使用中0件と表示 | 再ビルドで再生成（時間コスト） | 表示上の解放可能は215.3 MBのみ。`docker builder prune -a`は表示上の全キャッシュを対象にしますが、**ルートFSで回復する実容量は共有層次第で未確認** |
| 未使用イメージ（30個中14個） | 21.5 GB（解放可能表示） | 21.5 GB | 未確認（同上） | 未使用と表示 | pull / build で再取得可 | vllm等の大型イメージは再取得コスト大。tagの消失に注意 |
| `/tmp/issue30_hc_benchmark` | 2.3 GB（`du`実測） | — | **`/`（実測）** | 過去のベンチマーク成果物（作業ユーザー所有、2026-08-24） | 再実行で再生成 | 保持要否は所有者の判断 |
| journald | 1,021 MiB（`journalctl --disk-usage`） | — | **`/`（実測）** | システムログ（使用中） | 不可 | 保持期間の縮小になる |

**これらを合算して「約○○ GB解放できる」と扱わないでください。** 種別ごとの表示容量・共有分・表示上の解放可能量・保存先FSはそれぞれ別物で、ルートFSで実際に空く量は測定していません。

### 別作業（containerd移設）との関係

containerdの保存先を空きのある`/data`へ移す作業は、**このIssue／PRの範囲外で、別に進行しています**（ホスト上に `~/seedcore-containerd-copy.sh`、18,614 bytes、更新 2026-08-31 09:41 が存在）。本記録の時点で読み取り確認できた状態は次のとおりです。

| 確認項目（2026-08-31、読み取りのみ） | 結果 |
|---|---|
| `/` の空き | 238 MiB（100%使用）。**変化なし** |
| containerdの起動引数 / `config.toml` | `--root`指定なし、`root`/`state`行なし（ファイル更新は 6月19日のまま）→ 有効な保存先は既定の`/var/lib/containerd`（＝`/`） |
| 移設先候補ディレクトリ | `/data/containerd`・`/home/containerd`等は**存在しない** |
| コピー等の実行中プロセス | 検出なし |
| `docker` / `containerd` サービス | いずれも active / enabled |

すなわち、**移設は完了していない（少なくとも有効化されていない）**と読み取れます。コピーが行われたとしても、コピー完了は移設完了でも空き容量の回復でもありません（設定の切り替えと、旧領域の扱いの決定が別途必要です）。

このスレッドでは、削除・prune・再コピー・設定変更・サービスの起動停止を**一切行いません**。GPUのイメージbuildとsmokeは、**移設・復旧の状態と`/`の空き容量が確認できた後の別段階**とします。

### 実機実行の前に解決が必要な点

- **`/`が100%使用（空き238 MiB）です。** CUDAイメージのbuildはcontainerdのstore（推定`/var/lib/containerd`＝`/`）へ数GBを書き込むため、この状態では完了しません。空き容量の確保は、上記の別作業（containerd移設）または候補の整理を踏まえた**所有者の判断**によります。本PRでは実行も見積もりも行いません。
- GPUは共有中です（`llama-server`が6.4 GiBを使用）。本Issueの範囲では既存サービスを停止しません。GPU smokeの所要VRAMは小規模（synthetic 3家系・1 epoch）ですが、本実験（#6）を行う場合は、空きVRAMと既存サービスへの影響を事前に確認してください。
- ホストにuvが無いため、非Docker経路を使う場合は導入が必要です（Docker経路であれば不要）。

## 3. 依存関係の選定と根拠

CPU既定環境（ルートの`pyproject.toml` / `uv.lock`、torch 2.2.1 CPU build）は変更していません。CUDA環境は`cuda/`配下で独立に固定します。

| 項目 | 値 |
|---|---|
| Python | 3.11（リポジトリのピン留めどおり。cu130 indexにcp311 wheelが存在することを確認） |
| PyTorch | `2.12.1+cu130`（`https://download.pytorch.org/whl/cu130`） |
| CUDA / cuDNN | wheel同梱（`torch.version.cuda` = 13.0、cuDNN 9.20.0） |
| PyTorch Geometric | 2.7.0（純Python。CPU環境と同一） |
| numpy / pandas / scikit-learn / scipy | ルートと同一のピン留め（1.26.4 / 3.0.3 / 1.8.0 / 1.17.1） |
| uv | 0.12.3（CPUイメージ・CIと同一） |

選定根拠:

1. **GPU architecture（必須条件）**: RTX 5090のcompute capabilityは12.0（NVIDIAの"CUDA GPUs"一覧でRTX 50シリーズは12.0）。sm_120に対応するPyTorchは2.7以降で、PyTorch 2.7のリリースノートに "support for the NVIDIA Blackwell GPU architecture and pre-built wheels for CUDA 12.8" と明記されています。**CPU環境と同じ2.2.1はsm_120を含まないため、CUDA wheelへ差し替えるだけでは使えません**（当初案のtorch 2.2.1+cu121はこの理由で不採用）。
2. **Python wheelの存在（必須条件）**: 本リポジトリのピン留めであるPython 3.11のwheel（cp311）がcu130 indexに存在することを確認しています。
3. **依存整合性（必須条件）**: numpy / pandas / scikit-learn / scipy をルートと同一のピン留めにしたまま解決できることを`uv lock`で確認しています（77 packages）。
4. **実証（選定の裏付け）**: この組合せを実際に導入し、wheelのarch flagsにsm_120が含まれること、およびリポジトリのテストスイートがこの環境（CPU実行）で通ることを確認しました（§5.1）。**GPU実機での動作は未検証です。**
5. **driver要件**: cu130はCUDA 13.0 GA相当で、Linux x86_64のdriver `>= 580.65.06` を要求します。実機の595.84はこれを満たします。

**driverの互換性について（重要な訂正）**: 以前の版では「driverと同じCUDAメジャー系列でなければならない」としていましたが、これは誤りです。NVIDIAの[minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)によれば、CUDA 12.xでビルドしたアプリケーションはr580以降のdriverでも**バイナリ後方互換により動作**します（要件表の`< 580`は、minor version compatibilityが適用される範囲を示すもので、580以降でCUDA 12.xが使えないという意味ではありません）。したがってcu128系（torch 2.7〜2.11、CUDA 12.8 GAはdriver `>= 570.26`）も対象ホストで動作しうる選択肢です。cu130を採用したのは「メジャー系列を揃える必要があるから」ではなく、上記1〜4を満たす組合せとして**実際に検証したのがこれだから**です。

**バージョンの選び方（方針であり、安定性の実証ではありません）**: cu130 indexにcp311 wheelが存在するのは2.9.0以降です。その中で、最新の2.13.0（`.0`リリース）ではなく、後続のpatch releaseが出ている系列の最新である2.12.1を選びました。これは変更を小さく保つための方針であり、「patch版だから安定している」という実証ではありません。

トレードオフと制約:

- cu130は**driver >= 580.65.06**を要求します。それより古いdriverのホストでは動きません。その場合はsm_120対応を維持したままcu128系（torch 2.7〜2.11、driver `>= 570.26`）へ切り替えられます。`cuda/pyproject.toml`のindexとtorchのピンを変更して`uv lock --project cuda`を再実行します（ルートの`uv.lock`は変更しません）。ただし、その組合せは本記録の§5.1のような導入・テストによる裏付けが別途必要です。
- **torch自体はCPU既定環境（2.2.1）とCUDA環境（2.12.1）で異なります。** CPU実行とGPU実行を直接比較する場合は、`resnet-cpu-cuda-env`サービス（CUDAイメージでのCPU実行）を使い、torchを揃えてください。GBLUPも同じイメージで実行する`gblup-cuda-env`があります。既定のCPUイメージ（torch 2.2.1）の結果とGPU結果を突き合わせる場合は、この差を比較の制約として明記してください。
- ベースイメージの選択は**ホストのdriver要件を変えません**（§6を参照）。

## 4. 手順

### 4.1 Dockerを使う場合

```bash
# イメージのbuild（CPUイメージとは別。数GB規模のダウンロードが発生する）
docker compose --profile gpu build gpu-smoke

# syntheticな3家系でのGPU smoke
# GPUが見えない場合はskipではなく失敗する（GPRH_REQUIRE_CUDA=1）
docker compose --profile gpu run --rm gpu-smoke

# 実データのResNet（CUDA）
docker compose --profile gpu run --rm resnet-gpu

# 比較用: 同一イメージ・同一torchでのCPU実行
docker compose --profile gpu run --rm resnet-cpu-cuda-env
docker compose --profile gpu run --rm gblup-cuda-env
```

### 4.2 Dockerを使わない場合

```bash
uv sync --frozen --dev --project cuda
GPRH_ENVIRONMENT=cuda-13.0-torch-2.12.1 \
  uv run --frozen --project cuda \
  python resnet_baseline.py \
  --data-dir data --output-dir resnet_results --device cuda
```

`GPRH_ENVIRONMENT`は`metadata.json`の`environment_label`へ記録され、どの固定環境で実行したかを後から識別できます（Dockerイメージでは自動的に設定されます）。

### 4.3 記録される情報

`resnet_results/artifacts/<run_id>/metadata.json`に次が入ります。

- `device_requested` / `device_resolved`（要求と実際に使われたdevice）
- `cuda_version` / `cudnn_version`
- `gpu_name` / `gpu_compute_capability`
- `cuda_driver_api_version`（driverが対応するCUDA API版）/ `nvidia_driver_version`（`nvidia-smi`から取得、取得できない場合は`null`）
- `environment_label`（`GPRH_ENVIRONMENT`）
- `library_versions`（numpy / pandas / scikit-learn / torch / torch-geometric）

## 5. 検証ログ

### 5.1 実施済み（CPU側で確認できる範囲）

| 日時(UTC) | 内容 | 結果 |
|---|---|---|
| 2026-08-31T00:27Z | `uv sync --frozen --dev --project cuda`（CPUマシン上でCUDA lockを導入） | 成功 |
| 2026-08-31T00:27Z | 導入されたtorchとCUDA runtimeの確認（ホストのdriver表示とは別物） | `torch 2.12.1+cu130` / `torch.version.cuda = 13.0`（wheel同梱のCUDA runtime）/ cuDNN 9.20.0 / PyG 2.7.0 / numpy 1.26.4 |
| 2026-08-31T00:27Z | wheelが含むGPU architectureの確認（`torch._C._cuda_getArchFlags()`） | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` → **sm_120（RTX 5090）を含む** |
| 2026-08-31T00:28Z | CUDA環境でのテストスイート（CPU実行、`uv run --project cuda pytest -q`） | **112 passed, 1 skipped**（skipはGPU実機を要するCUDA smoke） |
| 2026-08-31 | `docker compose config --quiet` / `--profile gpu config` | 成功（gpu profileの4サービスを解決） |

### 5.2 未実施（GPU実機）

> 2026-08-31時点の表です。build・smoke・forward/backward・LOFO完走・run artifacts保存は2026-09-15に実施し成功しました（§7）。

| 項目 | 状態 |
|---|---|
| `docker compose --profile gpu build gpu-smoke` | 未実施（`/`が100%使用・空き238 MiB。containerd移設という別作業の状態確認と空き容量の確保が先。§2参照） |
| `docker compose --profile gpu run --rm gpu-smoke` | 未実施 |
| CUDA上でのforward/backward・LOFO完走・run artifacts保存 | 未実施 |
| GPU要求時の不在・不一致の挙動（実機） | 未実施（GPUの無い環境での失敗のみ確認済み） |
| 実データのGPU本実験・精度・計算コスト比較 | 未実施（#6） |

実施した場合は、次の形式でこの表の下へ追記してください。

```text
<UTC日時> | <実行コマンド> | <GPU名 / driver / CUDA> | <イメージ・lockの識別> | <結果> | <run_id>
```

## 6. 既知の未解決事項

- synthetic GPU smokeは2026-09-15に実機で成功しました（§7）。実データのGPU本実験は未実施です（#6）。GPUと互換性の無いビルドでの実機失敗、ホスト上の非Docker経路は未実施です（§7.7）。
- コンテナ内で作成された`metadata.json`の`git_commit`は`null`です（`.dockerignore`が`.git/`を除外するため）。GPU実行の証跡では、build元checkoutのcommit（§7.3）を別途記録してください。
- `metadata.json`の`cuda_driver_api_version`は、2026-09-15のGPU実機実行でも`null`でした（`nvidia_driver_version`は`595.84`を記録）。同じイメージ内で確認したところ、`resnet_baseline.py`が参照する`torch._C._cuda_getDriverVersion`はtorch 2.12.1+cu130に存在せず（`getattr`の結果が`None`）、実装どおり`null`が記録されています。driver版の証跡は`nvidia_driver_version`と§7.1の`nvidia-smi`を使ってください。
- `nvidia-smi`が表示するCUDA版（実機では13.2）は、driverが対応する最大のCUDA版です。コンテナ／venv内で実際に使われるCUDA runtimeの版（`torch.version.cuda` = 13.0）とは別物であり、両者を同一視しないでください。
- **ベースイメージの変更はホスト側の制約を解消しません。** `Dockerfile.cuda`が`python:3.11-slim`を使うのは、CPUイメージと基盤を揃えて依存差を減らすためです。CUDA runtime / cuDNNは`cuda/uv.lock`が固定するwheelが提供します。ホストのdriverが要件（cu130なら`>= 580.65.06`）未満の場合や、GPUのarchitectureが採用したPyTorchビルドの対象外（例: sm_120非対応のビルド）である場合は、`nvidia/cuda`系のベースイメージへ変更しても解決しません。必要なのはdriverの更新、またはGPUに対応したPyTorch/CUDAの組合せへの変更です。
- GPU実行時の数値はCPU実行と完全一致しません（cuDNNのアルゴリズム選択・非決定的なreduction）。#6の比較では、同一splitと同一の評価尺度を使ったうえで、この差を制約として明記してください。
- 既定のCPUイメージ（torch 2.2.1）とCUDA環境（torch 2.12.1）ではtorchが異なります。CPU/GPU比較を行う場合は§3のトレードオフに従い、同一イメージでの実行を使うか、差を明記してください。

## 7. GPU実機検証（seedcore-01、2026-09-15）

作業端末（a5-pro、NVIDIA GPUなし）からSSHで対象GPUホストへ接続し、ホスト上で実行しました。事前計測は読み取りのみです。削除・prune・containerd移設・Docker設定変更・サービス停止・driver更新は行っていません。GPU上の既存プロセスは停止していません（計測時点でcompute processは存在せず、§7.1）。使用したのはsyntheticデータのみで、実データ・非公開データは使っていません。

### 7.1 事前計測（2026-09-15T14:01:28Z〜14:02:42Z、読み取りのみ）

| 項目 | 実測値（2026-09-15） | 2026-08-31（§2） |
|---|---|---|
| hostname | `seedcore-01` | seedcore-01 |
| OS / kernel | Ubuntu 24.04.5 LTS / `6.8.0-139-generic` | 24.04.4 LTS / 6.8.0-136-generic |
| GPU / compute capability | NVIDIA GeForce RTX 5090（1基）/ 12.0 | 同じ |
| NVIDIA driver / `nvidia-smi`のCUDA表示 | 595.84 / 13.2 | 同じ |
| VRAM（`nvidia-smi --query-gpu`） | 32,607 MiB 中 **80 MiB 使用・32,026 MiB 空き**、GPU使用率 0% | 6,579 MiB 使用・25,527 MiB 空き、4% |
| GPU上のプロセス | `Xorg` 44 MiB・`gnome-shell` 11 MiB（いずれもType G）。compute processなし。`pgrep llama-server`は該当なし | `llama-server` 6,412 MiB |
| Docker（client / server） | 29.8.0 / 29.8.0、Compose v5.5.1 | 29.7.2（Composeは未記録） |
| Runtimes / Default Runtime | `runc io.containerd.runc.v2 nvidia` / `runc` | `nvidia`登録済み / `runc` |
| NVIDIA Container Toolkit | `nvidia-ctk` 1.20.0、`nvidia-container-toolkit` 1.20.0-1 | 1.20.0-1 |
| Storage Driver | `overlayfs`（`driver-type: io.containerd.snapshotter.v1`） | 同じ |
| `Docker Root Dir` | `/data/docker` | 同じ |
| containerdの起動引数 | `/usr/bin/containerd`（`--root`指定なし）、プロセス開始 2026-09-12 09:29:44（ホスト時刻） | `--root`指定なし |
| `/etc/containerd/config.toml` | 1行目に`root = "/data/containerd"`あり（ファイルのmtimeは2026-06-19 19:57:29 +0900と表示） | `root` / `state`の指定行なし |
| `containerd config dump` | `root = '/data/containerd'`、`state = '/run/containerd'` | 未取得 |
| containerdのsystemd drop-in | `/etc/systemd/system/containerd.service.d/90-seedcore-data-mount.conf`が存在（内容は権限不足で**未確認**） | 未記録 |
| containerdの実効root | **`/data/containerd`**（`/dev/nvme1n1p1`、`/data`にマウント）。`/var/lib/containerd`は存在しない | 既定の`/var/lib/containerd`（`/`）と判断 |
| `docker` / `containerd`サービス | active / active | active / active |
| ホストのPython / uv / git | 3.12.3 / uvなし / 2.43.0 | 3.12.3 / uvなし / 未記録 |

`df -h / /home /data`:

| マウント | デバイス | 容量 | 使用 | 空き | 使用率 | 2026-08-31 |
|---|---|---:|---:|---:|---:|---|
| `/` | `/dev/nvme0n1p2` | 183G | 44G | **130G** | 25% | 使用173G・空き238M・100% |
| `/home` | `/dev/nvme0n1p3` | 3.5T | 883G | 2.4T | 27% | 使用880G・空き2.4T・27% |
| `/data` | `/dev/nvme1n1p1` | 3.6T | 1.7T | 1.8T | 48% | 使用115G・空き3.3T・4% |

`df -i`: `/` 3%（2026-08-31は12%）、`/home` 1%、`/data` 1%。

`docker system df`（build前）:

| 種別 | 総数 | 使用中 | サイズ | 解放可能（docker表示） | 2026-08-31 |
|---|---:|---:|---:|---:|---|
| Images | 28 | 15 | 109.2GB | 25.71GB (23%) | 30 / 16 / 115.5 GB / 21.5 GB |
| Containers | 21 | 13 | 1.429GB | 608.8MB (42%) | 23 / 0 / 1.65 GB / 1.65 GB |
| Local Volumes | 3 | 3 | 1.118GB | 0B | 3 / 3 / 1.121 GB / 0 B |
| Build Cache | 34 | 0 | 35.01GB | 1.48GB | 34 / 0 / 35.01 GB / 215.3 MB |

**安全条件の判断**: GPUが見え（VRAM空き32,026 MiB）、containerdの実効rootが`/data/containerd`（空き1.8T）で、`/`にも130Gの空きがあるため、buildを実施しました。containerdの保存先が`/data`へ変わった経緯（2026-08-31時点で別作業として進行中だった移設との関係）は、本作業では確認していません。

### 7.2 2026-08-31からの主な差分

- **containerdの実効root**: 既定の`/var/lib/containerd`（`/`）→ `/data/containerd`（`/data`）。`/var/lib/containerd`は存在しなくなりました。
- **`/`の空き容量**: 238 MiB（100%）→ 130G（25%）。`/data`の使用は115G → 1.7Tに増加。
- **GPUの使用状況**: `llama-server`（6,412 MiB）が稼働 → compute processなし（80 MiB使用）。
- **ソフトウェア**: OS 24.04.4 → 24.04.5、kernel 6.8.0-136 → 6.8.0-139、Docker 29.7.2 → 29.8.0。driver（595.84）・NVIDIA Container Toolkit（1.20.0-1）は変化なし。

### 7.3 実行コマンドと結果

ホスト上に新規ディレクトリ`~/gprh-issue13-verify/`を作成し、公開リポジトリをcloneしてmainの`f732e4badaed4bfb63c2df217ebe3f4519d93b08`をcheckoutしました（`git status --short`は0行）。`cuda/uv.lock`のsha256は`ae3efbecd0517e1528dd85cfbadd1f30ae8ec881da6f8a9570fcf59c9c110de1`です。

| 日時(UTC) | コマンド（checkout直下） | 結果 |
|---|---|---|
| 2026-09-15T14:02:42Z | `docker compose --profile gpu config --quiet` | 成功（exit 0） |
| 2026-09-15T14:02:53Z〜14:04:50Z | `docker compose --profile gpu build --progress plain gpu-smoke` | 成功（exit 0）。全stepが`CACHED`なしで実行、`uv sync --frozen`で74 packagesを導入 |
| 2026-09-15T14:04:59Z〜14:05:08Z | `docker compose --profile gpu run --rm gpu-smoke` | **成功（exit 0）: `1 passed, 1 skipped in 7.27s`** |

build結果:

| 項目 | 値 |
|---|---|
| イメージ | `genomic-prediction-resnet-hybrid-gpu-smoke:latest` |
| image ID | `sha256:6cbde95a85adb5973c7fec253e544baa439f0c9104e10b3ffa60660595101c36`（8.38GB） |
| ベースイメージ | `python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534` |
| uv | `ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc` |
| build元commit | `f732e4badaed4bfb63c2df217ebe3f4519d93b08`（コンテナ内には`.git`が無く、`metadata.json`の`git_commit`は`null`） |

### 7.4 コンテナ内の環境（2026-09-15T14:05:26Z）

同じイメージで`docker compose --profile gpu run --rm gpu-smoke python -c ...`を実行して取得しました。

| 項目 | 値 |
|---|---|
| torch | `2.12.1+cu130` |
| `torch.version.cuda` | `13.0` |
| cuDNN（`torch.backends.cudnn.version()`） | `92000`（9.20.0）、`is_available()` = True |
| `torch.cuda.is_available()` / device数 | True / 1 |
| GPU名 / compute capability | `NVIDIA GeForce RTX 5090` / `(12, 0)` |
| `torch.cuda.get_arch_list()` | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` |
| PyG / numpy / Python | 2.7.0 / 1.26.4 / 3.11.16 |
| `GPRH_ENVIRONMENT` / `GPRH_REQUIRE_CUDA` | `cuda-13.0-torch-2.12.1` / `1` |
| 最小のautograd確認（CUDA tensorの行列積→`backward()`） | 勾配のdevice = `cuda:0` |

### 7.5 smokeの内訳とrun artifacts

`gpu-smoke`の既定コマンド（`pytest -q`）はテストごとの内訳を表示せず、OOF・artifactsはコンテナ内の一時ディレクトリに作られて`--rm`で消えます。そのため、同じイメージで詳細表示と成果物の保持を追加して再実行しました。

| 日時(UTC) | コマンド | 結果 |
|---|---|---|
| 2026-09-15T14:05:30Z | `docker compose --profile gpu run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/gprh-issue13-verify/smoke-out:/out gpu-smoke pytest -v -rA -p no:cacheprovider tests/test_gpu_smoke.py --basetemp=/out/pytest` | **失敗（1 failed, 1 skipped）**。原因は検証用に追加した`--user`指定: コンテナにUID 1000の`passwd`エントリが無く、torch import時の`getpass.getuser()`が`KeyError: 'getpwuid(): uid not found: 1000'`で終了。文書化された`gpu-smoke`経路（root実行）では発生しない |
| 2026-09-15T14:05:49Z〜14:05:57Z | 上記に`-e USER=gprh -e LOGNAME=gprh`を追加し、`--basetemp=/out/pytest-run2` | **成功（exit 0）: `1 passed, 1 skipped in 6.72s`** |

テストの内訳（2回目の詳細実行）:

- `tests/test_gpu_smoke.py::test_resnet_cuda_smoke` **PASSED**
- `tests/test_gpu_smoke.py::test_cuda_request_fails_when_cuda_is_unavailable` SKIPPED（`requires a machine without CUDA`。GPUが見えるホストでは仕様どおりskip）

`test_resnet_cuda_smoke`は、synthetic 3家系を`resnet_baseline.py --device cuda --max-epochs 1 --batch-size 4`で学習し（学習ループは`resnet_baseline.py`の`loss.backward()` / `optimizer.step()`）、OOFの列・行数・家系・有限値、run artifactsの必須ファイル、`metadata.json`のdevice記録、3 foldの`metrics.json`を検査します。保持した成果物を直接確認した結果:

| 項目 | 値 |
|---|---|
| run_id | `20260915T140556Z-947db6c5` |
| `oof_predictions.csv` | 18行（`Founder1_NAM01` / `Founder2_NAM02` / `Founder3_NAM03` 各6行）、列 `family_id,sample_name,observed_yield_kg_ha,predicted_yield_kg_ha` |
| run artifacts | `metadata.json` / `metrics.json` / `predictions.csv` / `preprocessing.json` / `preprocessing_arrays.npz` / `split.json` |
| `metrics.json`のfold | 3 fold（held-out: Founder1_NAM01, Founder2_NAM02, Founder3_NAM03）。synthetic 1 epochの値であり、精度の評価には使えません |
| `device_requested` / `device_resolved` | `cuda` / `cuda` |
| `cuda_version` / `cudnn_version` | `13.0` / `92000` |
| `gpu_name` / `gpu_compute_capability` | `NVIDIA GeForce RTX 5090` / `12.0` |
| `nvidia_driver_version` / `cuda_driver_api_version` | `595.84` / `null`（§6） |
| `environment_label` | `cuda-13.0-torch-2.12.1` |
| `library_versions` | numpy 1.26.4 / pandas 3.0.3 / scikit-learn 1.8.0 / torch 2.12.1+cu130 / torch-geometric 2.7.0 |
| `measurements`（PyTorch allocatorのみ） | `cuda_peak_allocated_bytes` 70,627,840 / `cuda_peak_reserved_bytes` 71,303,168 |

§5.2の形式での記録:

```text
2026-09-15T14:04:59Z | docker compose --profile gpu run --rm gpu-smoke | NVIDIA GeForce RTX 5090 (12.0) / 595.84 / torch.version.cuda 13.0, cuDNN 9.20.0 | image sha256:6cbde95a85ad…, commit f732e4b, cuda/uv.lock sha256 ae3efbec… | 1 passed, 1 skipped (exit 0) | run_idは一時ディレクトリで未保持
2026-09-15T14:05:49Z | 同上 + pytest -v -rA --basetemp（成果物保持） | 同上 | 同上 | 1 passed, 1 skipped (exit 0) | 20260915T140556Z-947db6c5
```

### 7.6 実行後の状態（2026-09-15T14:06:18Z、読み取りのみ）

| 項目 | 値 |
|---|---|
| VRAM / GPU使用率 | 80 MiB / 0%（実行前と同じ） |
| `df -h /` / `/data` | 空き130G（25%） / 空き1.8T（48%）（`df -h`の表示上は実行前と同じ） |
| 稼働中コンテナ数 | 13（実行前と同じ。`gpu-smoke`のコンテナは`--rm`で残っていない） |
| `docker system df` | Images 29 / 15 / 117.6GB / 34.09GB、Build Cache 48 / 0 / 43.47GB / 1.557GB |

ホスト上に残したもの（削除していません）: イメージ`genomic-prediction-resnet-hybrid-gpu-smoke:latest`、そのbuild cache、network `genomic-prediction-resnet-hybrid_default`（`compose run --rm`はnetworkを削除しない）、`~/gprh-issue13-verify/`（clone・`build.log`・syntheticの成果物、計7.4M。`smoke-out/`内のファイルはUID 1000所有）。

### 7.7 未実施・未確認

- GPUと互換性の無いビルド（例: sm_120非対応のtorch）での実機の失敗挙動: 未実施。
- GPU不在時の明確な失敗: CPU環境でのみ確認（§1）。GPUホストでは対象テストがskipのため再確認していません。
- ホスト上の非Docker経路（`uv sync --project cuda`）: 未実施（ホストにuv未導入）。
- containerd drop-in（`90-seedcore-data-mount.conf`）の内容: 権限不足で未確認。
- 実データのGPU本実験・精度・計算コスト比較: 未実施（#6、本Issueの対象外）。
