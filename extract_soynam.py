import pandas as pd
import numpy as np
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

# RとPythonのデータ変換を有効化
pandas2ri.activate()

# Rのパッケージをロード (library(SoyNAM) と同じ)
soynam = importr('SoyNAM')

# データをロード (data(swat), data(G2f) と同じ)
r('data(swat)')
r('data(G2f)')

# RのオブジェクトをPandas/Numpyに変換
# swat: 表現型データ
df_pheno = r['swat'] 
# G2f: ゲノムデータ (行列形式)
genotype_matrix = np.array(r['G2f'])

print(f"表現型データの形状: {df_pheno.shape}")
print(f"ゲノムデータの形状: {genotype_matrix.shape}")

# あとで train.py で使いやすいように保存しておく
df_pheno.to_csv("data/soynam_pheno.csv", index=False)
np.save("data/soynam_geno.npy", genotype_matrix)