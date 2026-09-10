# 再現性整備の受入状況（Issue #1）

親IssueはGPU本実験（#6）の完了までopenで維持します。

| 項目 | 実装・証跡 | 残る確認 |
|---|---|---|
| Python/Docker CPU経路 | #2 / #8、CPU CI・Docker smoke | 実データの性能保証は含まない |
| sample/family/marker照合 | #7、`soynam_data.py` | データ取得元・版・利用条件は利用者側で確定 |
| fold内前処理・split・実行記録 | #9、`run_manifest.py` | 同一splitの強制再利用と比較報告は#6 |
| 外部ロギング明示・legacy誤用防止 | #17 / #18 | legacyの正式サポート化は対象外 |
| CUDA環境 | #19、`docs/gpu-verification.md` | GPU上のbuild/smokeは#13で未完了 |
| 入力欠損率・閾値指定 | `input_qc.py`、両baselineのCLI、`tests/test_input_qc.py` | 実データの閾値は本実験前に確定 |
| GBLUP/ResNet本実験 | #6 | データ・形質・除外条件、GPU実行、精度/コスト/seed変動 |

## 欠損率ポリシー

両baselineに次の引数があります。閾値は結果を見る前に確定してください。

- `--max-sample-missing-rate`：0〜1。欠損率がこの値を超える個体を除外。既定1は全個体を維持します。各個体自身の遺伝型だけで判定し、家系が丸ごと消える場合は失敗します。
- `--min-marker-observed-rate`：学習fold内の観測率フィルター。GBLUPの既定0.1は厳密な`>`、ResNetの既定0.9は`>=`です。既存計算を維持しており、同じ数値を指定しても境界値の扱いは異なります。GBLUPでは0と1を引数検証で拒否します。

MAFの既定値もGBLUP 0.05 / ResNet 0.01、分散判定も異なります。同一splitだけで前処理が同一になったとは扱わず、#6の比較でそれぞれの条件を記録します。imputationは既存どおり学習foldの平均値です。

`metadata.json.input_qc`に対象個体の順序、閾値、除外前後の件数、配列参照を記録し、`preprocessing_arrays.npz`にsample欠損率・採用mask・marker欠損率を保存します。既存の6ファイル構成を維持します。ここでのmarker欠損率は欠損表現型・founder除外後、sample QC前の記述統計で、marker選択には使いません。marker選択・imputation・標準化・PCAは引き続き各学習fold内でfitします。

テストはsyntheticデータによる、閾値境界・個体と表現型の対応・家系消失の拒否・学習fold限定のmarker選択・両CLIの成果物保存の確認です。生物学的な閾値の妥当性を示すものではありません。
