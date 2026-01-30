# Genomic-ResNet-Prediction 

従来の統計モデル（GBLUP）と深層学習（ResNet）を統合し、ゲノムデータから収量予測を行うハイブリッド・フレームワークです。

## 概要
本プロジェクトは、線形モデルである MT-GBLUP (Multi-Trait Genomic Best Linear Unbiased Prediction) が捉えきれない、遺伝子間の非線形な相互作用（エピスタシス等）を ResNet (Residual Network) で抽出することを目指しています。

## 特徴
- **Hybrid Architecture**: 統計モデルによる線形推定と、DLによる残差補正を組み合わせた二段階予測。
- **Optimized ResNet**: 勾配消失を防ぐスキップ接続、GELU活性化関数、BatchNormを採用。
- **Rigorous Validation**: 10-fold Cross-Validationを50回反復し、統計的有意差（Paired t-test）を検証。
- **W&B Integration**: Weights & Biasesによるリアルタイムな学習ログの監視とハイパーパラメータ管理。

## プロジェクト構成
```text
genomic-resnet-prediction/
├── data/              # 表現型・遺伝型データ(CSV)
├── model.py           # GenomicResNetの定義
├── utils.py           # R(sommer)連携・データ前処理
├── main.py            # 50反復検証の実行スクリプト
├── requirements.txt   # 依存ライブラリ
└── LICENSE            # MIT License
## 実験結果と統計検証 (Benchmark Results)

モデルの堅牢性を検証するため、10-fold Cross-Validationを50回反復（計500回の試行）した結果は以下の通りです。

### 予測精度の比較 (Pearson's r)
| Model | Mean Accuracy | Std Dev | Max Accuracy |
| :--- | :---: | :---: | :---: |
| **ST-GBLUP** | 0.2730 | 0.0365 | 0.3573 |
| **MT-GBLUP** | **0.3102** | 0.0406 | 0.3811 |
| **Hybrid (ResNet)** | 0.3096 | 0.0405 | **0.3887** |

### 統計的考察
- **MT-GBLUPの優位性**: 単一形質モデル(ST)に対し、複数形質モデル(MT)は大幅な精度向上を達成しました。
- **Hybridモデルの現状**: 現時点のパラメータ設定では、HybridモデルはMT-GBLUPと同等の精度を維持していますが、統計的な有意差（p-value: 0.58）は認められませんでした。これは、線形効果が支配的なデータセットにおいてDLによる残差補正を最適化する難しさを示唆しています。
- **今後の展望**: W&B Sweepを用いたハイパーパラメータの自動最適化、およびアンサンブル学習を導入し、残差に含まれる非線形遺伝効果の抽出精度を向上させる予定です。
