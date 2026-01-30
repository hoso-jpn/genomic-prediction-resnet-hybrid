import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
numpy2ri.activate()
sommer = importr('sommer')

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
    geno_data = df_geno.iloc[:, 5:].replace(mapping).infer_objects(copy=False).T
    geno_data.index.name = 'Corrected Strain'
    geno_data = geno_data.fillna(geno_data.mean())
    
    combined = pd.merge(df_pheno_mean, geno_data, on='Corrected Strain', how='inner')
    y_multi = combined[target_cols].astype(np.float64).copy()
    y_multi.columns = [traits_map[c] for c in y_multi.columns]
    X = combined.iloc[:, len(target_cols) + 1 :].values.astype(np.float32)
    return X, y_multi.fillna(y_multi.mean())

def calculate_gblup_residuals(X, y_df, train_idx, test_idx, strain_ids):
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    G = (np.dot(X_std, X_std.T) / X.shape[1]) + np.eye(len(X)) * 1e-5
    
    df_r = y_df.copy()
    df_r['id'] = strain_ids
    df_r.loc[test_idx, 'Yield'] = np.nan 

    robjects.globalenv['df_r'] = df_r
    robjects.globalenv['G_mat'] = G
    robjects.globalenv['strain_ids_r'] = robjects.StrVector(strain_ids)
    
    r_script = """
    library(sommer)
    rownames(G_mat) <- strain_ids_r; colnames(G_mat) <- strain_ids_r
    
    ans_mt <- mmer(fixed = cbind(Yield, DaysToMat, Protein, SeedWeight) ~ 1,
                   random = ~ vsr(id, Gu=G_mat, Gtc=unsm(4)), 
                   rcov = ~ vsr(units, Gtc=diag(4)), data = df_r, verbose = FALSE)
    
    u_mt_mat <- ans_mt$U[[1]]; if(is.list(u_mt_mat)) u_mt_mat <- do.call(cbind, u_mt_mat)
    u_mt_yield <- u_mt_mat[strain_ids_r, 1] 
    res_mat <- as.matrix(ans_mt$residuals)
    full_res <- rep(0, length(strain_ids_r))
    full_res[as.numeric(rownames(res_mat))] <- as.numeric(res_mat[, 1])
    
    list(u_mt = u_mt_yield, res = full_res)
    """
    res = robjects.r(r_script)
    return np.array(res.rx2('u_mt')), np.array(res.rx2('res'))
