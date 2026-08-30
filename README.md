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
| 旧ResNet学習・W&B Sweep | experimental | `main.py`, `sweep_config.yaml` |
| GNN | experimental | `train_gnn.py` |

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
- CPU実行を標準経路とするPyTorch `2.2.1`

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

コードの公開は、研究データやログを外部サービスへ送信する許可を意味しません。`online`は利用者自身が明示的に選ぶ操作です。

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

`Dockerfile`は`pyproject.toml`・`uv.lock`に基づき、`uv sync --frozen --extra gblup --dev`でイメージを構築します。R・`rpy2`・`sommer`および`requirements.txt`には依存しません。ソースコードと`tests/`はイメージへ`COPY`されており、bind mountなしでコンテナ内に存在します。

`docker-compose.yml`のサービスは次の3系統に分かれます。

| 系統 | profile | 起動対象になる条件 |
|---|---|---|
| 検証済み（`unit-test`, `cpu-smoke`） | なし | `docker compose up`／`docker compose run <service>`で常に対象 |
| 実データ（`gblup`, `resnet`） | `real-data` | `--profile real-data`を明示した場合のみ |
| legacy/experimental（`preprocess`, `train`, `train-gpu`, `sweep-init`, `sweep-agent`, `gblup-baseline`, `dev`, `create-weights`, `create-graph-data`, `train-gnn`） | `legacy` | `--profile legacy`を明示した場合のみ |

`docker compose up`をprofile指定なしで実行した場合、起動対象は`unit-test`・`cpu-smoke`だけです。`real-data`・`legacy`のサービスは、検証済みベースライン経路ではないため既定では起動しません。

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

実データはリポジトリにもDockerビルドコンテキストにも含めません（`.gitignore`・`.dockerignore`でそれぞれ除外済み）。`data/`はbind mountでのみコンテナへ渡します。

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
  gblup_baseline.py resnet_baseline.py soynam_data.py run_manifest.py tests

uv run --frozen --extra gblup \
  ruff check \
  gblup_baseline.py resnet_baseline.py soynam_data.py run_manifest.py tests

uv run --frozen --extra gblup pytest -q
```

GitHub Actionsでは、対象コードのformat/lint、単体テストスイート、3 familyのsynthetic dataを使うGBLUP・ResNetのCPU smoke testを実行します。加えて、別ジョブでDocker Composeの設定検証、イメージbuild、`unit-test`・`cpu-smoke`サービスの実行、bind mountなしでのソース配置確認、rpy2非依存の確認を行います。実データ・GPU・W&B API keyはCIへ含めません。`gblup`・`resnet`（実データ）と`legacy`profileのサービスはCIで実行しません。

## 既知の制約

- `split.json`を読み込んで実行を固定する機能（同一splitの強制再利用）は未実装です（Issue #6予定）。
- Docker Composeの`gblup`・`resnet`サービスは実データを用いた手動実行経路であり、CIでは実行していません。
- GPUでの本実験、精度比較、統計的不確実性の評価は未実施です。
- `main.py`、`train_gnn.py`、dummy graph、W&B Sweepはlegacy/experimentalであり、検証済みベースライン経路には含まれません。
- CIのRuff対象は新しいベースライン実装と`tests/`に限定され、legacy scripts全体の整形は保証しません。

## データ引用

本解析にはSoyNAMプロジェクトの公開データセットを使用します。

- [SoyNAM project - SoyBase](https://www.soybase.org/projects/SoyNAM/)

データの利用条件と引用方法は配布元の案内に従ってください。

## ライセンス

[MIT License](LICENSE)
