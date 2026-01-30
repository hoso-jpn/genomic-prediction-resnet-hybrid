# genomic-resnet-prediction
## 概要
- 従来の統計モデル（GBLUP等）では捉えきれない遺伝子間の非線形な相互作用をResNetで抽出。
- 収量や成熟期などのマルチタスク学習を実装。

## 特徴
- 勾配消失を防ぐ残差ブロック構造
- 過学習を防ぐためのEarly Stoppingと反復検証スキーム
- 欠測値に対するデータ前処理パイプライン

## 使用技術
- Python 3.x
- PyTorch
- Scikit-learn
- W&B (Weights & Biases)
