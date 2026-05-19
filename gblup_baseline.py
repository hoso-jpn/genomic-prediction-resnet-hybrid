"""
LOFO-CV GBLUP ベースライン (R/sommer)

実行方法:
  docker compose run --rm gblup-baseline

実行時間の目安:
  G行列転送: 数秒
  sommer::mmer × 16フォールド: 30〜120分（サンプル数に依存）

R/sommer の mmer は:
  - 訓練個体の表現型から分散成分 σ²_g, σ²_e を REML 推定
  - G行列の共分散構造を使ってテスト個体の BLUP を予測
これは農業分野の標準的なゲノムセレクション手法 (GBLUP) と等価。
"""
import numpy as np
import pandas as pd
import os
import wandb
from sklearn.model_selection import LeaveOneGroupOut

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri, conversion
from rpy2.robjects.packages import importr

_converter = robjects.default_converter + pandas2ri.converter + numpy2ri.converter
conversion.set_conversion(_converter)
importr('sommer')


def compute_G(X):
    X_f = X.astype(np.float64)
    X_std = (X_f - X_f.mean(axis=0)) / (X_f.std(axis=0) + 1e-6)
    G = (X_std @ X_std.T) / X_f.shape[1]
    np.fill_diagonal(G, np.diag(G) + 1e-4)
    return G


def setup_r_globals(G, n, ids):
    """G行列とIDをR環境に一度だけ転送し、dat_base データフレームを作成する"""
    ids_r = robjects.StrVector(ids)
    G_r = robjects.r['matrix'](
        robjects.FloatVector(G.flatten().tolist()),
        nrow=n, ncol=n
    )
    robjects.globalenv['G_mat'] = G_r
    robjects.globalenv['ids_r'] = ids_r
    robjects.r("""
    library(sommer)
    rownames(G_mat) <- ids_r
    colnames(G_mat) <- ids_r
    dat_base <- data.frame(
        id    = factor(ids_r, levels = ids_r),
        yield = rep(NA_real_, length(ids_r))
    )
    """)


def run_gblup_fold(fold_idx, test_idx, y_all, family_label):
    y_masked = y_all.flatten().astype(np.float64).copy()
    y_masked[test_idx] = np.nan
    robjects.globalenv['y_vec'] = robjects.FloatVector(y_masked.tolist())

    blup_vec = np.array(robjects.r("""
    dat_base$yield <- y_vec
    ans <- tryCatch(
        mmer(
            fixed  = yield ~ 1,
            random = ~ vsr(id, Gu = G_mat),
            rcov   = ~ vsr(units),
            data   = dat_base,
            verbose = FALSE
        ),
        error = function(e) { message("[sommer error] ", conditionMessage(e)); NULL }
    )
    if (is.null(ans)) {
        rep(NA_real_, length(ids_r))
    } else {
        as.numeric(ans$U[[1]]$yield[ids_r])
    }
    """))

    y_true = y_all.flatten()[test_idx]
    y_pred = blup_vec[test_idx]
    valid  = ~np.isnan(y_pred)

    if valid.sum() < 2:
        r_val = float('nan')
    else:
        r_val = float(np.corrcoef(y_true[valid], y_pred[valid])[0, 1])
        if np.isnan(r_val):
            r_val = 0.0

    label = family_label or f"Fold {fold_idx + 1}"
    r_str = f"{r_val:.4f}" if not np.isnan(r_val) else "ERROR"
    print(f"{label:20s} | GBLUP r: {r_str}")
    return r_val, {"fold": fold_idx + 1, "gblup/r": r_val, "test_family": label}


def main():
    wandb.init(
        project="genomic-resnet-prediction-hy",
        job_type="gblup_baseline",
        name="gblup-lofo"
    )

    PROCESSED = './processed_data_hy/'
    y_df      = pd.read_csv(os.path.join(PROCESSED, 'y_phenotype_hy.csv'), index_col=0)
    y_all     = y_df['Yld (kg/ha)'].values.astype(np.float32)
    X_all     = np.load(os.path.join(PROCESSED, 'X_genotype_int8.npy')).astype(np.float32)
    family_ids = y_df['family_id'].values
    n          = len(y_all)

    print(f"G行列を計算中 ({n}×{n}, {X_all.shape[1]} SNPs)...")
    G   = compute_G(X_all)
    ids = [f"g{i:04d}" for i in range(n)]

    print("R環境にG行列を転送中 (約38MB)...")
    setup_r_globals(G, n, ids)
    del G
    print("転送完了。LOFO-CV 開始。\n")

    splitter = LeaveOneGroupOut()
    all_r    = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_all, y_all, family_ids)):
        family_label = str(np.unique(family_ids[test_idx])[0])
        r_val, log_dict = run_gblup_fold(fold, test_idx, y_all, family_label)
        wandb.log(log_dict)
        if not np.isnan(r_val):
            all_r.append(r_val)

    mean_r = float(np.mean(all_r)) if all_r else float('nan')
    print(f"\n{'='*45}")
    print(f"GBLUP LOFO 平均 r: {mean_r:.4f} ({len(all_r)}/{fold+1} フォールド成功)")
    wandb.log({"summary/mean_gblup": mean_r})
    wandb.finish()


if __name__ == "__main__":
    main()
