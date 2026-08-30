"""R の SoyNAM パッケージから元データを書き出すオプションのツール。

検証済み経路（`soynam_data.py` → `gblup_baseline.py` / `resnet_baseline.py`）は
R も rpy2 も使わない。このスクリプトは、R の SoyNAM パッケージから
表現型・遺伝型を取り出して手元へ保存したい場合にだけ使う補助ツールで、
`uv sync --extra soynam`（rpy2）と R 側の SoyNAM パッケージが必要になる。

出力（`data/soynam_pheno.csv`・`data/soynam_geno.npy`）は、検証済み経路が
読み込む family 別の gzip TSV とは形式が異なる。そのまま
`gblup_baseline.py` / `resnet_baseline.py` の入力にはならない。

実行スクリプトであり、import すると R 側の処理がそのまま走る。
"""

import numpy as np
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

# RとPythonのデータ変換を有効化
pandas2ri.activate()

# Rのパッケージをロード (library(SoyNAM) と同じ)
soynam = importr("SoyNAM")

# データをロード (data(swat), data(G2f) と同じ)
r("data(swat)")
r("data(G2f)")

# RのオブジェクトをPandas/Numpyに変換
# swat: 表現型データ
df_pheno = r["swat"]
# G2f: ゲノムデータ (行列形式)
genotype_matrix = np.array(r["G2f"])

print(f"表現型データの形状: {df_pheno.shape}")
print(f"ゲノムデータの形状: {genotype_matrix.shape}")

# 保存先。検証済み経路の入力形式ではない点に注意（上のdocstring参照）。
df_pheno.to_csv("data/soynam_pheno.csv", index=False)
np.save("data/soynam_geno.npy", genotype_matrix)
