import pandas as pd
import numpy as np
import os

# 設定：ファイルパスのリスト
families = [
    {
        "id": 3,
        "geno": "data/NAM24/4J105-3-4_NAM03_4312_SNP_genotype_Wm82.a1.tsv",
        "pheno": "data/NAM24/4J105-3-4_NAM03_phenotype_data.tsv"
    },
    {
        "id": 24,
        "geno": "data/NAM40/LG03-2979_NAM24_4312_SNP_genotype_Wm82.a1.tsv",
        "pheno": "data/NAM40/LG03-2979_NAM24_phenotype_data.tsv"
    },
    {
        "id": 40,
        "geno": "data/NAM03/PI_398881_NAM40_4312_SNP_genotype_Wm82.a1.tsv",
        "pheno": "data/NAM03/PI_398881_NAM40_phenotype_data.tsv"
    }
]

# SNPの置換マップ (A:0, H:1, B:2, 欠損:-1)
# int8を使うことでメモリを節約します
mapping = {'A': 0, 'H': 1, 'B': 2, '-': -1}

all_genotypes = []
all_phenotypes = []

print("データ統合プロセスを開始します...")

for fam in families:
    print(f"--- Processing NAM {fam['id']} ---")
    
    # 1. 表現型データの読み込み (収量 Yld を取得)
    df_p = pd.read_csv(fam['pheno'], sep='\t')
    # 必要な列だけ抽出し、家系IDを追加
    df_p = df_p[['Corrected Strain', 'Yld (kg/ha)']].dropna()
    df_p['Family_ID'] = fam['id']
    
    # 2. ゲノムデータの読み込み
    # SNPデータは「行=SNP, 列=個体」なので注意
    df_g = pd.read_csv(fam['geno'], sep='\t')
    
    # メタデータ列（dbSNP_ID等）を除外し、個体列のみにする
    # snippetによると最初の6列程度がメタデータ
    sample_cols = [c for c in df_g.columns if c.startswith('DS11') or c in ['Parent_IA3023', fam['geno'].split('_')[0]]]
    
    # 表現型データが存在する個体のみに絞り込む
    common_samples = list(set(sample_cols) & set(df_p['Corrected Strain']))
    df_p = df_p[df_p['Corrected Strain'].isin(common_samples)]
    
    # SNP行列を数値化して転置 (個体 x SNP)
    # 非常にメモリを食う作業なので注意
    geno_matrix = df_g.set_index('dbSNP_ID')[common_samples].replace(mapping).T
    geno_matrix = geno_matrix.astype(np.int8)
    
    all_genotypes.append(geno_matrix)
    all_phenotypes.append(df_p)

# 3. 全家系を縦方向に結合
print("全データをマージ中...")
final_geno = pd.concat(all_genotypes, axis=0).fillna(-1).astype(np.int8)
final_pheno = pd.concat(all_phenotypes, axis=0)

# 4. 保存
os.makedirs("processed_data", exist_ok=True)
final_geno.to_pickle("processed_data/X_genotype.pkl") # 高速読み込み用
final_pheno.to_csv("processed_data/y_phenotype.csv", index=False)

print(f"完了！ 最終的な行列サイズ: {final_geno.shape}")
print(f"ResNet入力用のSNP数: {final_geno.shape[1]}")