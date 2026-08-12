# Genomic-Prediction-ResNet-Hybrid

[![CI](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/workflows/ci.yml)

SoyNAM（Soybean Nested Association Mapping）の遺伝型データから収量を予測し、未知の家系への外挿性能を評価する研究用リポジトリです。

現在の再現性検証済み経路は、family単位のLeave-One-Family-Out cross-validation（LOFO-CV）を行うGBLUPとResNetの2つのベースラインです。旧来のGNN、W&B Sweepはexperimentalとして分離しています。Docker / Docker Composeは、単体テストとResNet CPU smokeについて検証済みです。

## 実装状況

| 機能 | 状態 | 実装 |
|---|---|---|
| SoyNAM raw data loader | 検証済み | `soynam_data.py` |
| GBLUP LOFO baseline | 検証済み | `gblup_baseline.py` |
| ResNet LOFO baseline | 検証済み | `resnet_baseline.py` |
| 単体テスト・synthetic CPU smoke | CI実行 | `tests/`, `.github/workflows/ci.yml` |
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

現在のGBLUP CLIは`data/`、16 family、`gblup_results/`を前提とします。W&Bへ送信せずローカルで再現する場合はoffline modeを使います。

```bash
WANDB_MODE=offline \
  uv run --frozen --extra gblup \
  python gblup_baseline.py
```

出力:

```text
gblup_results/oof_predictions.csv
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

3 familyのsynthetic dataのみを使い、`resnet_baseline.py`のCLIと4列のOOF出力契約を検証します。外部データ・GPU・`.env`・W&B API keyは不要です。

```bash
docker compose build cpu-smoke
docker compose run --rm cpu-smoke
```

### GBLUP・ResNet（実データ、手動実行）

`gblup`・`resnet`サービスは、実データを保有する利用者が手動で起動する経路です。`profiles: ["real-data"]`が付いており、CIでは実行しません。実行前に、[入力データ](#入力データ)節と同じ形式のSoyNAMデータを`./data`に配置してください。`data/`はいずれもread-onlyでmountし、結果ディレクトリのみ書き込み可能にしています。

```bash
# GBLUP（現時点のCLIはdata/、16 family、gblup_results/を前提とするため、
# 家系数が異なるデータでは完走しません）
docker compose --profile real-data run --rm gblup

# ResNet
docker compose --profile real-data run --rm resnet
```

`gblup`サービスは`WANDB_MODE=offline`をCompose側で設定しているため、W&B API keyは不要です。実データが無い状態でこれらのサービスを実行した場合の挙動（`FileNotFoundError`等）はCIでの検証対象にしていません。

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

モデル比較では同一sample・同一splitのOOF予測を使用し、fold単位のPearson相関、macro family相関、pooled OOF相関、RMSEなどを目的に応じて明示します。

## テストとCI

```bash
uv run --frozen --extra gblup \
  ruff format --check \
  gblup_baseline.py resnet_baseline.py soynam_data.py tests

uv run --frozen --extra gblup \
  ruff check \
  gblup_baseline.py resnet_baseline.py soynam_data.py tests

uv run --frozen --extra gblup pytest -q
```

GitHub Actionsでは、対象コードのformat/lint、単体テストスイート、3 familyのsynthetic dataを使うResNet CPU smoke testを実行します。加えて、別ジョブでDocker Composeの設定検証、イメージbuild、`unit-test`・`cpu-smoke`サービスの実行、bind mountなしでのソース配置確認、rpy2非依存の確認を行います。実データ・GPU・W&B API keyはCIへ含めません。`gblup`・`resnet`（実データ）と`legacy`profileのサービスはCIで実行しません。

## 既知の制約

- GBLUPはdata directory、出力先、16 familyをCLIで変更できません（Docker Composeの`gblup`サービスも同じ制約を継承します）。
- split、marker filter、imputation、PCA、選択epochを機械可読な成果物として保存していません。
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
