# Genomic-Prediction-ResNet-Hybrid

[![CI](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/workflows/ci.yml)

SoyNAM（Soybean Nested Association Mapping）の遺伝型データから収量を予測し、未知の家系への外挿性能を評価する研究用リポジトリです。

現在の再現性検証済み経路は、family単位のLeave-One-Family-Out cross-validation（LOFO-CV）を行うGBLUPとResNetの2つのベースラインです。旧来のGNN、W&B Sweepはexperimentalとして分離しています。Docker / Docker Composeは、単体テストとCPU smoke（GBLUP・ResNet）について検証済みです。

## 実装状況

| 機能 | 状態 | 実装 |
|---|---|---|
| SoyNAM raw data loader | 検証済み | `soynam_data.py` |
| GBLUP LOFO baseline | 検証済み | `gblup_baseline.py` |
| ResNet LOFO baseline | 検証済み | `resnet_baseline.py` |
| 単体テスト・synthetic CPU smoke（GBLUP・ResNet） | CI実行 | `tests/`, `.github/workflows/ci.yml` |
| Docker / Docker Compose（unit-test・cpu-smoke） | 検証済み | `Dockerfile`, `docker-compose.yml` |
| Docker / Docker Compose（gblup・resnet、実データ） | 手動実行経路（CI未実行） | `docker-compose.yml` |
| CUDA実行環境（GPU smoke・resnet GPU経路） | 準備済み・**GPU実機未検証** | `Dockerfile.cuda`, `cuda/`, `docs/gpu-verification.md` |
| 旧ResNet学習・W&B Sweep | experimental（`--allow-legacy`必須） | `main.py`, `sweep_config.yaml` |
| 旧前処理 | experimental（`--allow-legacy`必須） | `preprocess.py` |
| GNN | experimental（`--allow-legacy`必須） | `train_gnn.py` |

「検証済み」は、入力整合性・split・前処理・出力契約とCPU上の実行経路を自動テストまたはスモークテストで確認したことを意味します。予測精度の優位性や大規模GPU実験の再現を保証するものではありません。

## 評価設計

両ベースラインは、16家系のうち1家系をouter testとして保持するLOFO-CVを使用します。すべての個体は、held-out familyの予測として1回だけOOF（out-of-fold）出力へ現れます。

### GBLUP

`gblup_baseline.py`はNumPy/SciPyで実装した1-kernel GBLUPです。

- training familyだけでmarker欠損率、MAF、平均imputation値を推定
- training dataからVanRaden relationship matrixを構築
- profile REMLで遺伝分散と残差分散の比を推定
- held-out familyはtest-training relationshipから予測

Rおよび`sommer`は、この検証済みGBLUP経路では使用しません。

### ResNet

`resnet_baseline.py`は、PCAを入力する線形パスと1D CNN residual pathをゲートで統合した`GatedGenomicResNet`を使用します。

- marker filtering、平均imputation、標準化、PCAは学習partitionだけでfit
- outer testとは別のvalidation familyでepochを選択
- 選択後、outer training families全体で前処理とモデルを再fit
- seedをPython、NumPy、PyTorchへ設定

線形パスはAdamWで学習する正則化線形予測器です。relationship matrixやmixed model equationsを使うRR-BLUP / GBLUPと同一ではありません。比較対象のGBLUPは独立した`gblup_baseline.py`です。

## 必要環境とセットアップ

- Python `3.11.x`
- [uv](https://docs.astral.sh/uv/)（CI検証バージョン: `0.12.3`）
- CPU実行を標準経路とするPyTorch `2.2.1`（GPU実行は`cuda/`の独立した固定環境を使用、[GPU実行環境](#gpu実行環境)参照）

```bash
git clone https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid.git
cd genomic-prediction-resnet-hybrid

uv python install
uv sync --frozen --extra gblup --dev
```

`uv.lock`を使用するため、依存関係を更新せず再現する場合は`--frozen`を付けます。

## 入力データ

既定の入力先は`data/`です。各familyについて、次のgzip圧縮TSVを1組配置します。

```text
<family_id>_phenotype_data.tsv.gz
<family_id>_4312_SNP_genotype_Wm82.a1.tsv.gz
```

phenotype/genotypeファイルはファイル名から抽出したfamily IDで対応付けます。同じfamily IDへ複数のphenotypeファイル、または複数のgenotypeファイルが対応する場合はエラーとし、phenotype側とgenotype側のfamily ID集合が一致しない場合もエラーとします。

表現型ファイルの必須列は次のとおりです。

| 列 | 内容 |
|---|---|
| `Corrected Strain` | sample ID |
| `Yld (kg/ha)` | 収量（kg/ha） |

遺伝型ファイルは、先頭列をmarker ID、残りの列をsample IDとして読み込みます。対応する符号は次のとおりです。

| 入力 | 数値表現 |
|---|---:|
| `A`, `A/A` | -1 |
| `H`, `A/B` | 0 |
| `B`, `B/B` | 1 |
| `-`, empty | missing (`NaN`) |

未知の記号が含まれる場合はエラーとします。

loaderは読み込み時に次を検証します。

- sample ID（phenotypeの`Corrected Strain`列、genotypeの先頭行の列名）とmarker ID（genotypeの先頭列）は、前後の空白を除去したうえで、欠損・空文字・重複を許容しません。違反した場合はfamily ID、対象ファイル名、対象IDを含むエラーになります。genotypeファイルはpandasによる重複ヘッダーの自動リネームを避けるため、先頭行を直接読み取って検証してから本体を読み込みます。
- founder parent（family IDを`_NAM`で分割した前半部分）はphenotype/genotype双方のsample集合比較から除外します。片側のファイルにだけfounder parentの列・行が存在していても、RIL sample照合には影響しません。
- founder parent除外後のRIL sample集合は、phenotypeとgenotypeで完全に一致する必要があります。一致しない場合、`phenotype_only`・`genotype_only`としてどちらか一方にしか存在しないsample IDを列挙したエラーになります（共通部分だけを採用する暗黙の整列は行いません）。
- 全familyのmarker集合と順序は、最初に読み込んだfamilyを基準に一致している必要があります。marker集合そのものが異なる場合と、集合は同じだが順序だけが異なる場合を区別してエラーにします。
- phenotype値の欠損・空文字は、sample ID照合が成功した後に判定します。該当するsampleは学習対象から除外されますが、それ以外の値は数値へ変換できない場合エラーになります。欠損個体を除外した結果、familyのRIL sampleが0件になった場合もエラーになります。

出力される配列のsample順序は、phenotypeファイル内の出現順を維持します。

## 実行方法

### GBLUP baseline

```bash
uv run --frozen --extra gblup \
  python gblup_baseline.py \
  --data-dir data \
  --output-dir gblup_results \
  --expected-families 16
```

主なCLI引数は`--data-dir`、`--output-dir`、`--expected-families`、`--wandb-mode`です。既定値は`data`、`gblup_results`、`16`、`disabled`で、引数を省略した場合の入力・出力・家系数はこれまでと同じです。外部サービスの認証情報は不要です。

`--expected-families`は家系数チェックを無効化するためのものではなく、期待する家系数の指定です。読み込んだデータの家系数が一致しない場合は実行前に失敗します（LOFO-CVの都合上、2未満は指定できません）。

出力:

```text
gblup_results/oof_predictions.csv
```

3 familyのsyntheticデータなど、16家系以外のデータで配線を確認する例:

```bash
uv run --frozen --extra gblup \
  python gblup_baseline.py \
  --data-dir /path/to/synthetic_data \
  --output-dir gblup_smoke_results \
  --expected-families 3
```

### ResNet baseline

```bash
uv run --frozen --extra gblup \
  python resnet_baseline.py \
  --data-dir data \
  --output-dir resnet_results \
  --device cpu \
  --seed 42
```

主なCLI引数は`--device`、`--seed`、`--max-epochs`、`--patience`、`--batch-size`、`--pca-components`です。

短時間の配線確認例:

```bash
uv run --frozen --extra gblup \
  python resnet_baseline.py \
  --data-dir data \
  --output-dir resnet_smoke_results \
  --device cpu \
  --max-epochs 1 \
  --patience 1 \
  --pca-components 8
```

1 epochの実行は配線と出力契約の確認用であり、予測精度の評価には使用しません。

## W&B（Weights & Biases）の扱い

GBLUPの外部ロギングは`--wandb-mode`だけで決まります。

| `--wandb-mode` | W&Bの初期化 | 外部送信 | ローカル出力 |
|---|---|---|---|
| `disabled`（既定） | 行わない（`import wandb`もしない） | なし | run成果物のみ |
| `offline` | 行う | なし | run成果物 + ローカルW&B run directory |
| `online` | 行う | あり（API key等の認証情報が必要） | run成果物 + W&Bサービス上のrun |

優先順位は次のとおりです。

- CLIの`--wandb-mode`が唯一の決定要素です。`WANDB_MODE`等の環境変数でmodeを変更することはできません。
- `offline`・`online`を選んだ場合、`wandb.init`の直前にプロセスの`WANDB_MODE`を選択値へ上書きし、同じ値を`wandb.init(mode=...)`にも渡します。周囲の環境変数が`offline`を`online`へ引き上げることはありません。
- 既定は`disabled`です。`WANDB_MODE=online`が設定された環境で引数なしに実行しても、W&Bは初期化されません。`online`にできるのは`--wandb-mode online`を明示した場合だけです。
- CLI引数の検証とデータ読み込み・家系数チェックは、W&Bの初期化より前に行います。引数や入力が不正な実行がW&B上にrunを作ることはありません。

ResNet（`resnet_baseline.py`）はW&Bを使用しません。実行の記録は両ベースラインとも[成果物](#成果物)節のrun artifactsが担います。

legacy/experimentalの`main.py`・`train_gnn.py`も同じ`--wandb-mode`を持ち、既定は`disabled`です。`--allow-legacy`（[legacy / experimental経路](#legacy--experimental経路)）は旧経路の実行許可であり、外部送信の許可ではありません。両者は別々に明示する必要があります。

コードの公開は、研究データやログを外部サービスへ送信する許可を意味しません。`online`は利用者自身が明示的に選ぶ操作です。

## GPU実行環境

既定のCPU環境（ルートの`pyproject.toml` / `uv.lock`、PyTorch 2.2.1 CPU build）はCIとDockerの既定経路で使用し、変更していません。GPU比較実験用のCUDA環境は`cuda/pyproject.toml`と`cuda/uv.lock`で独立に固定します。

対象GPUは**NVIDIA GeForce RTX 5090（compute capability 12.0 / sm_120）**で、採用した組合せは**PyTorch 2.12.1 + CUDA 13.0 wheel（cu130）**です。sm_120対応はPyTorch 2.7以降であり、CPU側と同じ2.2.1をCUDA wheelへ置き換えるだけでは使えません。この組合せは`cu130`のwheelを使うため、ホストには**NVIDIA driver >= 580.65.06**とNVIDIA Container Toolkitが必要です（cu128系のwheelなら`>= 570.26`）。選定根拠・トレードオフ・実測したホスト構成は[docs/gpu-verification.md](docs/gpu-verification.md)にまとめています。GPU実機での実行は未検証です。

```bash
# synthetic 3家系でのGPU smoke（GPUが見えない場合はskipではなく失敗する）
docker compose --profile gpu build gpu-smoke
docker compose --profile gpu run --rm gpu-smoke

# 実データのResNet（CUDA）
docker compose --profile gpu run --rm resnet-gpu

# 比較用: 同一イメージ・同一torchでのCPU実行
docker compose --profile gpu run --rm resnet-cpu-cuda-env
docker compose --profile gpu run --rm gblup-cuda-env
```

- `resnet_baseline.py --device cuda`はCUDAが使えない場合に明確に失敗し、CPUへ黙って切り替わりません。実際に使われたdeviceは`metadata.json`の`device_resolved`で確認できます。
- `metadata.json`にはGPU名、compute capability、CUDA/cuDNN、driver（`nvidia-smi`から取得できた場合）、および`GPRH_ENVIRONMENT`に基づく`environment_label`（どの固定環境で実行したか）を記録します。
- ホストにはNVIDIA driverとNVIDIA Container Toolkitが必要です。バージョン対応と手順、**未検証事項**は[docs/gpu-verification.md](docs/gpu-verification.md)にまとめています。

検証の区別:

| 区分 | 状態 |
|---|---|
| CPU unit test・synthetic CPU smoke | CIで実行・成功 |
| CUDA要求時の明確な失敗（GPU不在時） | CPU環境で確認済み |
| CUDA環境の導入とテストスイート（CPU実行） | 確認済み（torch 2.12.1+cu130、112 passed / 1 skipped。wheelが`sm_120`を含むことも確認） |
| synthetic GPU smoke（`tests/test_gpu_smoke.py`） | **GPU実機未検証**（CPU環境ではskip、CIにGPU runnerなし） |
| 実データのGPU本実験・精度比較 | **未実施**（Issue #6） |

## 成果物

GBLUP・ResNetは、実行ごとに再現性の追跡・監査に必要な成果物を`<output-dir>/artifacts/<run_id>/`へ保存します（`gblup_results/artifacts/...`・`resnet_results/artifacts/...`）。ルート直下の共通`artifacts/`は使用せず、`docker-compose.yml`の既存bind mount（`./gblup_results`・`./resnet_results`）だけで書き込み先を確保できます。

```text
gblup_results/
  oof_predictions.csv        # 既存の互換出力（4列、パス・列名は不変）
  artifacts/
    <run_id>/
      metadata.json          # run_id、git commit、依存バージョン、入力ファイル情報など
                             #   GBLUPは hyperparameters.expected_family_count と
                             #   external_logging.mode（W&Bのmode）も記録する
      split.json             # outer（LOFO）split。GBLUPは inner: null
      preprocessing.json     # 欠損率・MAF等の設定値とfold単位の要約統計
      preprocessing_arrays.npz  # fold単位の実数値配列（marker mask、imputation mean等）
      metrics.json           # fold単位・summary単位の指標
      predictions.csv        # oof_predictions.csvと同一内容（run単位で自己完結させるための複製）
```

`resnet_results/artifacts/<run_id>/`も同じ6ファイル構成です。`split.json`の`inner`にはResNet固有のvalidation family選択（`validation_family_selection`）が入り、`preprocessing.json`のfoldエントリには`selection_transform`（epoch選択用）と`final_transform`（best epoch決定後の再fit用）の2種類が別々に記録されます。

各JSONファイルは`schema_version`を持ち、将来のフィールド追加・変更を安全に検出できるようにしています（現在は`1`）。

### split.jsonとouter_split_hash

`split.json`の`outer.outer_split_hash`は、`ordered_samples`（sample ID・family IDの対応、順序を保持）と`folds`（LOFOの各foldでのtrain/test family）だけから計算したSHA-256です。GBLUPとResNetが同じ`data/`を読んだ場合、この値は完全に一致します。異なればsplitが揃っていないことを意味します。ResNetのinner split（validation family選択のseed等）はこのhashの計算に含まれません。

### 前処理値の実体

閾値や件数などのスカラー値は`preprocessing.json`に、fold単位で実際にfitされた配列（marker mask、imputation mean、標準化のmean/scale、PCAのmean/components/explained varianceなど）は`preprocessing_arrays.npz`に保存します。`preprocessing.json`の各fold・各transformエントリの`arrays`オブジェクトが、対応するNPZ内の配列名（`*_ref`）を指します。入力genotype/phenotypeデータそのものは複製しません。

### run_idと上書き防止

`run_id`は`<UTC時刻>-<UUID4先頭8文字>`形式です（例: `20260812T123456Z-a1b2c3d4`）。既に存在する`run_id`のディレクトリへは書き込めません。書き込みは一時ディレクトリで行い、全ファイルの書き込みが成功した後にのみ最終ディレクトリへ切り替えるため、途中で失敗した場合に不完全なrunディレクトリが残ることはありません。既存の`oof_predictions.csv`は、run成果物の確定後にのみ置き換えます。

### git commitの取得

`metadata.json`の`git_commit`はベストエフォートです。Dockerイメージは`.dockerignore`で`.git/`をビルドコンテキストから除外しているため、コンテナ内で実行した場合は取得できず`null`になります。明示的に記録したい場合は、実行前に環境変数`GIT_COMMIT_SHA`を設定してください（`git`コマンドより優先されます）。

### 未対応（Issue #6予定）

`split.json`を読み込んで実行を固定する機能（同一splitの強制再利用）やCLIオプションは、本Issue #5では追加していません。Issue #6（GPU本実験・GBLUP/ResNet比較）で対応予定です。

## Docker / Docker Compose

`Dockerfile`（CPU既定イメージ）は`pyproject.toml`・`uv.lock`に基づき、`uv sync --frozen --extra gblup --dev`でイメージを構築します。GPU用の`Dockerfile.cuda`は`cuda/`配下の別lockを使う独立したイメージです（[GPU実行環境](#gpu実行環境)）。R・`rpy2`・`sommer`および`requirements.txt`には依存しません。ソースコードと`tests/`はイメージへ`COPY`されており、bind mountなしでコンテナ内に存在します。

`docker-compose.yml`のサービスは次の3系統に分かれます。

| 系統 | profile | 起動対象になる条件 |
|---|---|---|
| 検証済み（`unit-test`, `cpu-smoke`） | なし | `docker compose up`／`docker compose run <service>`で常に対象 |
| 実データ（`gblup`, `resnet`） | `real-data` | `--profile real-data`を明示した場合のみ |
| GPU（`gpu-smoke`, `resnet-gpu`, `resnet-cpu-cuda-env`, `gblup-cuda-env`） | `gpu` | `--profile gpu`を明示した場合のみ（CUDAイメージ） |
| legacy/experimental（`preprocess`, `train`, `train-gpu`, `sweep-init`, `sweep-agent`, `gblup-baseline`, `dev`, `create-weights`, `create-graph-data`, `train-gnn`） | `legacy` | `--profile legacy`を明示した場合のみ |

`docker compose up`をprofile指定なしで実行した場合、起動対象は`unit-test`・`cpu-smoke`だけです。`real-data`・`gpu`・`legacy`のサービスは既定では起動しません。

### 単体テスト（unit-test）

外部データ・GPU・`.env`・W&B API keyは不要です。bind mountも使用せず、イメージ内のソースだけで実行します。

```bash
docker compose build unit-test
docker compose run --rm unit-test
```

### CPU smoke test（cpu-smoke）

3 familyのsynthetic dataのみを使い、`gblup_baseline.py`・`resnet_baseline.py`のCLIと4列のOOF出力契約、run成果物6ファイルを検証します。外部データ・GPU・`.env`・W&B API keyは不要です。GBLUP側は、`WANDB_MODE=online`が設定された環境でも既定のままではW&Bを初期化しないことも確認します。

```bash
docker compose build cpu-smoke
docker compose run --rm cpu-smoke
```

### GBLUP・ResNet（実データ、手動実行）

`gblup`・`resnet`サービスは、実データを保有する利用者が手動で起動する経路です。`profiles: ["real-data"]`が付いており、CIでは実行しません。実行前に、[入力データ](#入力データ)節と同じ形式のSoyNAMデータを`./data`に配置してください。`data/`はいずれもread-onlyでmountし、結果ディレクトリのみ書き込み可能にしています。

```bash
# GBLUP（--data-dir data --output-dir gblup_results --expected-families 16）
docker compose --profile real-data run --rm gblup

# ResNet
docker compose --profile real-data run --rm resnet
```

`gblup`サービスはW&Bを既定の`disabled`で実行するため、W&B API keyも`WANDB_MODE`の設定も不要です。家系数が16以外のデータを使う場合や、W&Bのmodeを変える場合は、サービス定義の`command`を上書きします。

```bash
docker compose --profile real-data run --rm gblup \
  python gblup_baseline.py \
  --data-dir data \
  --output-dir gblup_results \
  --expected-families 8 \
  --wandb-mode offline
```

実データが無い状態でこれらのサービスを実行した場合の挙動（`FileNotFoundError`等）はCIでの検証対象にしていません。

### legacy / experimental

`preprocess`・`train`・`train-gpu`・`sweep-init`・`sweep-agent`・`gblup-baseline`・`dev`・`create-weights`・`create-graph-data`・`train-gnn`は`profiles: ["legacy"]`で分離されています。これらは検証済みベースライン経路ではなく、`--profile legacy`を明示しない限り起動しません。

さらに、`preprocess.py`・`main.py`・`train_gnn.py`を起動するサービスは、サービス定義に`--allow-legacy`を含めていないため、profileを指定して起動しても終了コード2で停止します（[legacy / experimental経路](#legacy--experimental経路)）。意図的に実行する場合はコマンドを上書きします。

```bash
docker compose --profile legacy run --rm train \
  python3 main.py --allow-legacy
```

実データはリポジトリにもDockerビルドコンテキストにも含めません（`.gitignore`・`.dockerignore`でそれぞれ除外済み）。`data/`はbind mountでのみコンテナへ渡します。

## legacy / experimental経路

`preprocess.py`・`main.py`・`train_gnn.py`は検証済みベースラインではありません。誤って本実験や精度比較に使わないよう、これらは`--allow-legacy`が無い限り、**入力の読み込み・ファイル生成・W&Bの初期化より前に**終了コード2で停止し、代替コマンドと既知の問題を表示します。

```bash
# 既定（何もせず終了し、検証済みコマンドと既知の問題を表示）
uv run --frozen --extra gblup python main.py

# legacy利用を承知したうえで実行する場合
uv run --frozen --extra gblup python main.py --allow-legacy
```

- `--allow-legacy`はコマンドライン引数です。Docker Composeの`--profile legacy`指定や、W&B sweep agentの起動だけではこの確認を満たしません。Composeでlegacyを実行する場合は`docker compose --profile legacy run --rm train python3 main.py --allow-legacy`のようにコマンドを明示的に上書きし、sweepの場合は`sweep_config.yaml`の`command`へ自分で追記します。
- legacy許可と外部ロギング許可は分離しています。`--allow-legacy`を付けてもW&Bは初期化されず、`--wandb-mode offline` / `online`を別途明示した場合だけ有効になります。
- `main.py`は`family_id`列を必須にします。欠落時にrandom CVへ暗黙に切り替えることはせず、明確に失敗します。
- 実行ログの冒頭・末尾と、`preprocess.py`が生成する`processed_data_hy/EXPERIMENTAL.txt`にexperimentalである旨を出力します。
- legacyの出力は家系内で標準化した表現型に対する指標であり、検証済み経路のraw kg/haのOOF性能とは比較できません。W&B Sweepの探索目標はouter LOFOの集計値であるため、探索後の最高値も独立した汎化性能の証跡にはなりません。これらを検証済み性能として引用しないでください。

### legacyを検証済み扱いへ移行するための条件

継続利用する場合、次をすべて満たすまではexperimentalのままとします。

1. 入力は共通ローダー（`soynam_data.py`）を経由し、family ID照合・founder除外・marker ID検証・未知記号の拒否を通ること。
2. sample IDとmarker IDを処理の全段階で保持し、位置インデックスだけに依存しないこと。
3. 欠損補完・分散/MAFフィルター・標準化・PCAをfold内のtrain partitionだけでfitすること。
4. inner selection（epoch・ハイパーパラメータ選択）とouter testを分離し、選択後の最高値をouter性能として報告しないこと。
5. 検証済み経路と同一のsplit（`outer_split_hash`が一致）と同一の尺度（raw kg/ha）で評価すること。
6. `run_manifest.py`の既存manifest契約（metadata/split/preprocessing/metrics/predictions、`schema_version`、原子的な確定）に沿った成果物を出力すること。

GNNについては、SNP→gene対応をfoldのmarker maskへIDベースで反映する必要があり、ローダーの差し替えだけでは条件を満たしません。

## OOF出力契約

GBLUPとResNetは同じ4列のCSVを出力します。

| 列 | 内容 |
|---|---|
| `family_id` | family ID |
| `sample_name` | sample ID |
| `observed_yield_kg_ha` | 観測収量 |
| `predicted_yield_kg_ha` | held-out familyに対するOOF予測 |

モデル比較では同一sample・同一splitのOOF予測を使用し、fold単位のPearson相関、macro family相関、pooled OOF相関、RMSEなどを目的に応じて明示します。この4列は`<output-dir>/oof_predictions.csv`と、[成果物](#成果物)節で説明する`<output-dir>/artifacts/<run_id>/predictions.csv`の両方で同一です。

## テストとCI

```bash
uv run --frozen --extra gblup \
  ruff format --check \
  gblup_baseline.py resnet_baseline.py soynam_data.py run_manifest.py \
  external_logging.py legacy_guard.py tests

uv run --frozen --extra gblup \
  ruff check \
  gblup_baseline.py resnet_baseline.py soynam_data.py run_manifest.py \
  external_logging.py legacy_guard.py tests

uv run --frozen --extra gblup pytest -q
```

GitHub Actionsでは、対象コードのformat/lint、単体テストスイート（legacy経路の`--allow-legacy`確認を含む）、3 familyのsynthetic dataを使うGBLUP・ResNetのCPU smoke testを実行します。加えて、別ジョブでDocker Composeの設定検証、イメージbuild、`unit-test`・`cpu-smoke`サービスの実行、bind mountなしでのソース配置確認、rpy2非依存の確認を行います。実データ・GPU・W&B API keyはCIへ含めません。`gblup`・`resnet`（実データ）と`legacy`profileのサービスはCIで実行しません。

## 既知の制約

- `split.json`を読み込んで実行を固定する機能（同一splitの強制再利用）は未実装です（Issue #6予定）。
- Docker Composeの`gblup`・`resnet`サービスは実データを用いた手動実行経路であり、CIでは実行していません。
- GPUでの本実験、精度比較、統計的不確実性の評価は未実施です（Issue #6）。
- CUDA実行環境（`Dockerfile.cuda` / `cuda/uv.lock` / `--profile gpu`）は対象GPU（RTX 5090）に合わせて選定済みで、CPU側で導入・テスト・sm_120対応まで確認していますが、**GPU実機での実行とイメージbuildは未実施**です。CIにGPU runnerは無く、CIの成功はGPU経路の検証にはなりません（[docs/gpu-verification.md](docs/gpu-verification.md)）。
- GPU実行の数値はCPU実行と完全には一致しません（cuDNNのアルゴリズム選択等）。比較時は同一splitと同一尺度を使い、この差を制約として明記してください。
- 既定のCPU環境（torch 2.2.1）とCUDA環境（torch 2.12.1）ではtorchのバージョンが異なります。CPU/GPUを直接比較する場合は、CUDAイメージでCPU実行する`resnet-cpu-cuda-env`・`gblup-cuda-env`を使ってtorchを揃えてください。
- `preprocess.py`、`main.py`、`train_gnn.py`、dummy graph、W&B Sweepはlegacy/experimentalであり、検証済みベースライン経路には含まれません。`--allow-legacy`は誤用防止のための確認であり、上記スクリプトの前処理・評価上の問題を解消するものではありません。
- CIのRuff対象は新しいベースライン実装と`tests/`に限定され、legacy scripts全体の整形は保証しません。

## データ引用

本解析にはSoyNAMプロジェクトの公開データセットを使用します。

- [SoyNAM project - SoyBase](https://www.soybase.org/projects/SoyNAM/)

データの利用条件と引用方法は配布元の案内に従ってください。

## ライセンス

[MIT License](LICENSE)
