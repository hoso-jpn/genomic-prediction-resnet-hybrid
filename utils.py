#utils.py
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion # 追加

# --- ここから環境設定 ---
# activate() の代わりに、一括で変換ルールを登録します
my_converter = robjects.default_converter + pandas2ri.converter + numpy2ri.converter
conversion.set_conversion(my_converter)

sommer = importr('sommer')
# --- 環境設定ここまで ---

def load_genomic_data(pheno_path, geno_path):
    df_pheno = pd.read_csv(pheno_path)
    traits_map = {'Yld (bu/a)': 'Yield', 'Days to Mat': 'DaysToMat', 'Protein': 'Protein', '100 sdwt (g)': 'SeedWeight'}
    target_cols = list(traits_map.keys())
    
    for col in target_cols:
        df_pheno[col] = pd.to_numeric(df_pheno[col], errors='coerce')
    
    df_pheno_mean = df_pheno.groupby('Corrected Strain')[target_cols].mean().reset_index()
    df_pheno_mean = df_pheno_mean.dropna(subset=['Yld (bu/a)'])
    
    df_geno = pd.read_csv(geno_path)
    mapping = {'A': 0, 'H': 1, 'B': 2, '-': np.nan, 'N': np.nan}
    # T.copy() を入れて明示的にコピーを作成し、型エラーを防ぎます
    geno_data = df_geno.iloc[:, 5:].replace(mapping).infer_objects(copy=False).T.copy()
    geno_data.index.name = 'Corrected Strain'
    geno_data = geno_data.fillna(geno_data.mean())
    
    combined = pd.merge(df_pheno_mean, geno_data, on='Corrected Strain', how='inner')
    y_multi = combined[target_cols].astype(np.float64).copy()
    y_multi.columns = [traits_map[c] for c in y_multi.columns]
    X = combined.iloc[:, len(target_cols) + 1 :].values.astype(np.float32)
    return X, y_multi.fillna(y_multi.mean()), combined['Corrected Strain'].tolist()

def calculate_gblup_residuals(X, y_df, train_idx, test_idx, strain_ids):
    # numpyのデータをRに渡すために変換器を有効にした状態で処理
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    # G行列計算
    G = (np.dot(X_std, X_std.T) / X.shape[1]) + np.eye(len(X)) * 1e-5
    
    df_r = y_df.copy()
    df_r['id'] = strain_ids
    df_r.loc[test_idx, 'Yield'] = np.nan 

    # Rのグローバル環境へデータを転送
    robjects.globalenv['df_r'] = df_r
    robjects.globalenv['G_mat'] = G
    robjects.globalenv['strain_ids_r'] = robjects.StrVector(strain_ids)
    
    r_script = """
    library(sommer)
    rownames(G_mat) <- strain_ids_r; colnames(G_mat) <- strain_ids_r
    
    # sommerによる多変量GBLUP
    ans_mt <- mmer(fixed = cbind(Yield, DaysToMat, Protein, SeedWeight) ~ 1,
                   random = ~ vsr(id, Gu=G_mat, Gtc=unsm(4)), 
                   rcov = ~ vsr(units, Gtc=diag(4)), data = df_r, verbose = FALSE)
    
    u_mt_mat <- ans_mt$U[[1]]; if(is.list(u_mt_mat)) u_mt_mat <- do.call(cbind, u_mt_mat)
    u_mt_yield <- u_mt_mat[strain_ids_r, 1] 
    res_mat <- as.matrix(ans_mt$residuals)
    full_res <- rep(0, length(strain_ids_r))
    
    # 欠損値（test_idx）がある場合の残差補完
    existing_idx <- as.numeric(rownames(res_mat))
    full_res[existing_idx] <- as.numeric(res_mat[, 1])
    
    list(u_mt = u_mt_yield, res = full_res)
    """
    res = robjects.r(r_script)

    # res.names に () をつけて呼び出すように修正します
    names = list(res.names()) 
    u_mt_idx = names.index('u_mt')
    res_idx = names.index('res')

    return np.array(res[u_mt_idx]), np.array(res[res_idx])

print("関数の定義が完了しました。")
