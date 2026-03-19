import pandas as pd
import numpy as np
import glob
import os
import gc

def preprocess_to_numpy():
    data_dir = './data'
    pheno_files = sorted(glob.glob(os.path.join(data_dir, '*_phenotype_data.tsv.gz')))
    geno_files = sorted(glob.glob(os.path.join(data_dir, '*_SNP_genotype_Wm82.a1.tsv.gz')))

    all_y_std = [] # 標準化後の表現型を格納
    all_X_list = []
    mapping = {'A': -1, 'B': 1, 'H': 0, 'A/A': -1, 'B/B': 1, 'A/B': 0}

    print(f"📊 {len(pheno_files)} 家族のデータを数値化・家族内標準化中...")

    for p_file, g_file in zip(pheno_files, geno_files):
        family_id = os.path.basename(p_file).split('_')[0]
        
        # 1. 表現型読み込み
        y_df = pd.read_table(p_file, compression='gzip')
        y_df['Yld (kg/ha)'] = pd.to_numeric(y_df['Yld (kg/ha)'], errors='coerce')
        y_df = y_df.dropna(subset=['Yld (kg/ha)', 'Corrected Strain'])
        y_df = y_df.drop_duplicates(subset='Corrected Strain')
        
        # 2. 遺伝型読み込み（転置）
        X_df_raw = pd.read_table(g_file, compression='gzip', index_col=0).T
        X_df_raw = X_df_raw[~X_df_raw.index.duplicated(keep='first')]
        
        # 3. 同期
        common_strains = y_df['Corrected Strain'].isin(X_df_raw.index)
        y_subset = y_df[common_strains].set_index('Corrected Strain')[['Yld (kg/ha)']]
        X_subset_raw = X_df_raw.loc[y_subset.index]
        
        # 【重要】家族内標準化 (Z-score)
        # 家族ごとの平均を0、分散を1に揃えることで環境ノイズを排除
        y_values = y_subset['Yld (kg/ha)'].values
        y_std = (y_values - np.mean(y_values)) / (np.std(y_values) + 1e-6)
        y_subset['Yld (kg/ha)'] = y_std
        
        # 4. 数値変換 (int8)
        X_numeric = np.zeros(X_subset_raw.shape, dtype=np.int8)
        for i, col in enumerate(X_subset_raw.columns):
            X_numeric[:, i] = X_subset_raw[col].map(mapping).fillna(0).values
        
        all_y_std.append(y_subset)
        all_X_list.append(X_numeric)
        
        print(f" ✅ {family_id}: {len(y_subset)} 個体完了 (Mean: {np.mean(y_values):.1f})")
        del X_df_raw, X_subset_raw
        gc.collect()

    # 統合
    final_y = pd.concat(all_y_std)
    final_X_array = np.vstack(all_X_list)

    # 5. 低分散SNPの除去（全員同じ値のSNPは予測に役立たないため除外）
    variances = np.var(final_X_array, axis=0)
    valid_snp_mask = variances > 1e-6
    final_X_array = final_X_array[:, valid_snp_mask]
    
    # 最終保存
    output_dir = 'processed_data_hy'
    os.makedirs(output_dir, exist_ok=True)
    final_y.to_csv(f'{output_dir}/y_phenotype_hy.csv')
    np.save(f'{output_dir}/X_genotype_int8.npy', final_X_array)
    
    print(f"\n✨ 前処理完了！")
    print(f"合計個体数: {final_X_array.shape[0]} | 残ったSNP数: {final_X_array.shape[1]}")
    
    # 相関チェック（全SNPの中から最大相関を探す）
    corrs = [np.corrcoef(final_X_array[:, i], final_y.iloc[:,0].values)[0,1] for i in range(min(500, final_X_array.shape[1]))]
    print(f"最大相関サンプル(先頭500中): {max(corrs, key=abs):.4f}")

if __name__ == "__main__":
    preprocess_to_numpy()