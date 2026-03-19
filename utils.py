# utils.py
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion

# --- 環境設定 ---
my_converter = robjects.default_converter + pandas2ri.converter + numpy2ri.converter
conversion.set_conversion(my_converter)

# Rパッケージ sommer のロード（インストール済み前提）
sommer = importr('sommer')

def load_genomic_data(pheno_path, geno_path):
    # 1. 表現型データの読み込みと単位の修正
    df_pheno = pd.read_csv(pheno_path)
    
    # 昨日の preprocess.py に合わせ、'Yld (kg/ha)' を使用
    traits_map = {
        'Yld (kg/ha)': 'Yield', 
        'Days to Mat': 'DaysToMat', 
        'Protein': 'Protein', 
        '100 sdwt (g)': 'SeedWeight'
    }
    
    # データが存在する列のみに絞り込む（不足している形質があってもエラーにしない）
    available_traits = [c for c in traits_map.keys() if c in df_pheno.columns]
    
    for col in available_traits:
        df_pheno[col] = pd.to_numeric(df_pheno[col], errors='coerce')
    
    # 系統ごとの平均値を計算（個体名：Corrected Strain）
    df_pheno_mean = df_pheno.groupby('Corrected Strain')[available_traits].mean().reset_index()
    # 収量データがない行は削除
    df_pheno_mean = df_pheno_mean.dropna(subset=['Yld (kg/ha)'])
    
    # 2. 遺伝子データの読み込み
    # すでに preprocess.py で転置・数値化済みの .pkl を読み込む場合を想定
    if geno_path.endswith('.pkl'):
        geno_data = pd.read_pickle(geno_path)
    else:
        # .csv の場合は従来の処理
        df_geno = pd.read_csv(geno_path)
        mapping = {'A': 0, 'H': 1, 'B': 2, '-': np.nan, 'N': np.nan}
        geno_data = df_geno.iloc[:, 5:].replace(mapping).infer_objects(copy=False).T.copy()
    
    geno_data.index.name = 'Corrected Strain'
    # 欠損値を平均値で補完
    geno_data = geno_data.fillna(geno_data.mean())
    
    # 3. データの結合（Inner Join）
    combined = pd.merge(df_pheno_mean, geno_data, on='Corrected Strain', how='inner')
    
    # 戻り値の整理
    y_multi = combined[available_traits].astype(np.float64).copy()
    y_multi.columns = [traits_map[c] for c in y_multi.columns]
    
    # 遺伝子データ部分のみを抽出（形質列 + ID列 を飛ばす）
    X = combined.iloc[:, len(available_traits) + 1 :].values.astype(np.float32)
    
    return X, y_multi.fillna(y_multi.mean()), combined['Corrected Strain'].tolist()

def calculate_gblup_residuals(X, y_df, train_idx, test_idx, strain_ids):
    # データの標準化
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    # G行列（血縁行列）の計算
    G = (np.dot(X_std, X_std.T) / X.shape[1]) + np.eye(len(X)) * 1e-5
    
    df_r = y_df.copy()
    df_r['id'] = strain_ids
    # テストセットの収量を欠損（NaN）させて、GBLUPで予測させる
    df_r.loc[test_idx, 'Yield'] = np.nan 

    # Rのグローバル環境へデータを転送
    robjects.globalenv['df_r'] = df_r
    robjects.globalenv['G_mat'] = G
    robjects.globalenv['strain_ids_r'] = robjects.StrVector(strain_ids)
    
    # sommer による多変量 GBLUP 解析
    r_script = """
   library(sommer)
    rownames(G_mat) <- strain_ids_r; colnames(G_mat) <- strain_ids_r
    
    # データをRのデータフレームとして確実に認識させる
    dat <- as.data.frame(df_r)
    dat$id <- as.character(strain_ids_r)
    
    # 単変量 GBLUP 解析（Yieldのみに集中）
    # 固定効果：1（切片）、変量効果：id（G行列を使用）
    ans <- mmer(fixed = Yield ~ 1,
                random = ~ vsr(id, Gu = G_mat), 
                rcov = ~ vsr(units), 
                data = dat, verbose = FALSE)
    
    # 予測値（BLUP）の抽出
    u_blup <- ans$U[[1]]$Yield
    if(is.null(u_blup)) {
        # 構造が違う場合のフォールバック
        u_blup <- as.matrix(ans$U[[1]])[,1]
    }
    
    # 全個体分を strain_ids_r の順序で取得
    u_mt_yield <- u_blup[strain_ids_r]
    
    # 残差の取得
    res_val <- as.matrix(ans$residuals)
    full_res <- rep(0, length(strain_ids_r))
    
    # 残差が存在するインデックス（訓練データ）に値を代入
    existing_idx <- as.numeric(rownames(res_val))
    full_res[existing_idx] <- as.numeric(res_val[, 1])
    
    list(u_mt = u_mt_yield, res = full_res)
    """
    res = robjects.r(r_script)

    names = list(res.names()) 
    u_mt_idx = names.index('u_mt')
    res_idx = names.index('res')

    return np.array(res[u_mt_idx]), np.array(res[res_idx])

print("関数の定義が完了しました。")