# GBLUP / ResNet比較実験（Issue #6）

実データでの本実験は2026-09-18にRTX 5090実機で完走しました。実測値は[実測レポート](experiments/issue-6-soynam-1.6.2.md)にまとめています（集計JSONは`experiments/issue-6-soynam-1.6.2-results.json`）。
本書は手順・事前定義・計測範囲の規定を扱い、実測値そのものは扱いません。GPUの環境構築とsmokeは#13で完了し、データ源は下記のとおりCRAN SoyNAM 1.6.2で確定しています。

CPU側のsynthetic検証（3 family、ResNet 2 seeds、各1 epoch）は動作確認用として引き続き有効で、精度の測定結果ではありません。

## canonicalなデータ源

入力は`soynam_cran.py`が生成するCRAN SoyNAM 1.6.2由来のcanonical datasetです（生成手順はREADME、出典と基準hashは[data-provenance.md](data-provenance.md)）。

| 項目 | 値 |
|---|---|
| source | CRAN SoyNAM 1.6.2（SHA-256 `0bd87f7b101456006a42a11679809349f7545b95dee0b43d954aa5b642995aaa`、GPL-3） |
| 表現型 | `lmer(Y ~ (1 | environ) + (1 | strain))`、REML、`use.check=FALSE`、欠測yieldの354行のみ除外 |
| genotype | `gen.qa`を0/1/2/NAのまま使用（`A`/`H`/`B`へ呼び替えない） |
| 規模 | 5,142 sample / 39 family / 4,312 marker（family 46は`gen.qa`に不在） |
| 実データ実行例 | `--dataset-id cran-soynam-1.6.2-yield-blup-false`、`--data-source "CRAN SoyNAM 1.6.2"`、`--data-kind real`、GBLUP CLIは`--expected-families 39` |

`plan`は、data directoryに`soynam-cran-*-manifest.json`があればそのfilenameとSHA-256を`experiment.json`へ記録し、`run`・`report`で同一であることを検証します。manifestが無いsynthetic経路は従来どおり動作します。

**`use.check=TRUE`を採用しなかった経緯**: raw `data.line$spot`が全lineで単一set`2A`に退化しており、TRUEでは10,355観測が意図の説明なく除外され、残る観測も本来のsetではないcheck値を受けます。詳細と実測値は[data-provenance.md](data-provenance.md)を参照してください。

## 対象と事前定義

`compare_baselines.py`は検証済みSoyNAM形式と`Yld (kg/ha)`専用です。小豆GSパネル単体には表現型がないため、このCLIへ直接入力しません。最低3 familyが必要です。データ版・取得元・real/synthetic区分は明示してください。

`plan`は新しいディレクトリだけを受け付け、次を学習前に保存します。

- `split-plan.json`：入力checksum、sample QC閾値、個体順、全familyのtrain/testとhash。
- `experiment.json`：データ識別、形質、seed一覧、device、学習予算、全ResNetハイパーパラメータの候補範囲、コードchecksum、Python/ライブラリ版、bootstrap条件とhash。

初期の探索範囲は各パラメータ1候補です。外側test結果で探索せず、既存の内側family検証でepochだけを選択します。探索範囲を増やす実験は、内側検証による候補選択を別途実装してから事前定義します。seedは候補選択に使わず、指定した全seedを報告します。

両モデルのsample QCは共通ですが、既存pipelineには以下の差が残ります。モデル構造だけの比較とは解釈しません。

| 条件 | GBLUP | ResNet |
|---|---|---|
| 既定marker観測率 | > 0.1 | >= 0.9 |
| 既定MAF | 0.05 | 0.01 |
| 特徴変換 | 学習平均imputation、VanRaden-1 | 学習平均imputation、標準化。標準化済みSNPをCNN経路、そのPCA成分を線形経路へ入力（モデル全体をPCA次元へ圧縮するのではない） |
| 選択 | 学習foldのREML | 内側family検証でepoch選択、外側学習データで再fit |

閾値の詳細は[欠損率ポリシー](readiness-audit.md)を参照してください。

## 実行

#13のCUDA環境とGPU smokeを確認後、同じ固定環境でplanとrunを実行します。以下はその環境内でのコマンド例であり、GPU実行済みの証跡ではありません。`DATA_RELEASE`・`DATA_SOURCE`は実際の値へ置き換えます。データは`data/`、成果物は永続化した`comparison_results/`に配置してください。

```bash
python compare_baselines.py plan \
  --data-dir data --experiment-dir comparison_results/run-001 \
  --dataset-id DATA_RELEASE --data-source DATA_SOURCE --data-kind real \
  --device cuda --seeds 42 43 44 \
  --max-epochs 200 --patience 20 --batch-size 64 --pca-components 64 \
  --max-sample-missing-rate 1.0 \
  --gblup-marker-rate 0.1 --resnet-marker-rate 0.9 --threads 1

python compare_baselines.py run \
  --data-dir data --experiment-dir comparison_results/run-001

python compare_baselines.py report \
  --data-dir data --experiment-dir comparison_results/run-001
```

CPUの動作確認は`--device cpu --data-kind synthetic`を明示します。既定CPU環境では`uv run --frozen --extra gblup python ...`を使えます。CUDA指定でGPUが利用できなければ学習開始前に失敗します。GBLUPとResNetは同じPython executable・依存環境で逐次実行し、外部W&B送信は無効です。

データ・コード・環境・分割・設定がplanから変化した場合は新しい実験を作ってください。`runs/`の上書き・自動resumeは行いません。途中失敗時の既存ログと成果物は残し、不完全な結果からcompleteレポートは生成しません。

## 保存・集計・制約

`runs/gblup/`と各`runs/resnet-seed-N/`に既存の6ファイルrun artifactsを保存します。両CLIに`--split-file`を追加し、保存された全LOFO foldを検証して実際に消費します。任意のrandom splitを許可する機能ではありません。

集計は入力checksum・split hash・個体/家系ID・実測表現型・seed・device・学習予算・コード/環境を照合してから、OOFからPearson相関とRMSEを再計算します。family別、全体、全seedを`comparison.json`と`comparison.md`へ出力します。定数ベクトル等で相関が未定義の場合はnullとします。

不確実性はfamilyを単位とする2,000回のpaired bootstrap（seed 42）で、seed平均ResNet−GBLUPのfamily平均差と95%区間を記録します。RMSE差は負、相関差は正がResNet側の改善を表します。学習集合の重複、family数、seed間の依存があるため、独立な反復実験や確定的な優位性の証明とは扱いません。

実行計測はCLI検証後から入力読み込み・全fold評価終了までのwall秒、process CPU user/system秒、process lifetime peak RSS、CUDAのPyTorch allocator peak allocated/reserved bytesです。CUDAは計測前後に同期します。importとartifact直列化はwall計測範囲外です。Windowsのresource情報やCPU実行のCUDA情報はnullです。

canonical datasetの表現型は全familyの全環境を1つの混合モデルで解いた調整値です。held-out familyの観測も分散成分・環境効果の推定に寄与するため、**完全に独立した外部検証や未知環境への予測とは解釈しません**。比較できるのは、同じ調整済み表現型・同じsample集合・同じLOFO splitの上でのGBLUPとResNetの差だけです。marker QC条件の差（観測率 `> 0.1` 対 `>= 0.9`、MAF 0.05対0.01）が残るため、**モデル構造だけの比較にもなりません**。

**GPU使用率・電力・driverが使用するメモリ・他processの負荷はこの計測に含みません。** GPU本実験ではホストの`nvidia-smi`等による時系列記録、GPU名/driver、コンテナimage ID・lock識別、実行日時を併せて保存し、#13の証跡と対応付けてください。2026-09-18の本実験では5秒間隔の`nvidia-smi`時系列、実行前後の`nvidia-smi -q`、image ID、host情報を保存しています（[実測レポート](experiments/issue-6-soynam-1.6.2.md)）。CPU/GPUの異なるlockを混ぜた比較や、syntheticの数値を生物学的性能と扱う報告は行いません。
