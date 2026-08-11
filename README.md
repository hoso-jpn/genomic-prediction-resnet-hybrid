# Genomic-Prediction-ResNet-Hybrid

[![CI](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/hoso-jpn/genomic-prediction-resnet-hybrid/actions/workflows/ci.yml)

SoyNAM（Soybean Nested Association Mapping）の遺伝型データから収量を予測し、未知の家系への外挿性能を評価する研究用リポジトリです。

現在の再現性検証済み経路は、family単位のLeave-One-Family-Out cross-validation（LOFO-CV）を行うGBLUPとResNetの2つのベースラインです。旧来のGNN、W&B Sweep、Docker経路はexperimentalまたは未検証として分離しています。

## 実装状況

| 機能 | 状態 | 実装 |
|---|---|---|
| SoyNAM raw data loader | 検証済み | `soynam_data.py` |
| GBLUP LOFO baseline | 検証済み | `gblup_baseline.py` |
| ResNet LOFO baseline | 検証済み | `resnet_baseline.py` |
| 単体テスト・synthetic CPU smoke | CI実行 | `tests/`, `.github/workflows/ci.yml` |
| 旧ResNet学習・W&B Sweep | experimental | `main.py`, `sweep_config.yaml` |
| GNN | experimental | `train_gnn.py` |
| Docker / Docker Compose | 未検証 | `Dockerfile`, `docker-compose.yml` |

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

loaderはfamily単位のファイル対応、必須列、全family間のmarker集合・順序、未知の遺伝型記号、最終的な配列次元を検証します。founder parentはfamily IDから判定して除外します。

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

GitHub Actionsでは、対象コードのformat/lint、13件の単体テスト、3 familyのsynthetic dataを使うResNet CPU smoke testを実行します。実データはCIへ含めません。

## 既知の制約

- GBLUPはdata directory、出力先、16 familyをCLIで変更できません。
- split、marker filter、imputation、PCA、選択epochを機械可読な成果物として保存していません。
- loaderはsample IDの共通部分を整列しますが、片側だけに存在するsampleや重複IDを厳格なエラーとして扱う対応は未完了です。
- Docker / Docker Compose経路は、現在の`uv`ベースラインに対して未検証です。
- GPUでの本実験、精度比較、統計的不確実性の評価は未実施です。
- `main.py`、`train_gnn.py`、dummy graph、W&B Sweepはlegacy/experimentalであり、検証済みベースライン経路には含まれません。
- CIのRuff対象は新しいベースライン実装と`tests/`に限定され、legacy scripts全体の整形は保証しません。

## データ引用

本解析にはSoyNAMプロジェクトの公開データセットを使用します。

- [SoyNAM project - SoyBase](https://www.soybase.org/projects/SoyNAM/)

データの利用条件と引用方法は配布元の案内に従ってください。

## ライセンス

[MIT License](LICENSE)
