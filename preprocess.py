import gc
import glob
import os

import numpy as np
import pandas as pd

import legacy_guard

DESCRIPTION = "legacy SoyNAM preprocessing (experimental, not a verified baseline)"
EXPERIMENTAL_NOTICE = """このディレクトリは legacy preprocess.py の出力です（experimental）。

- 表現型は family 内で標準化済みで、kg/ha ではありません。検証済み経路
  （gblup_baseline.py / resnet_baseline.py）の raw kg/ha 評価と同一視できません。
- phenotype/genotype はファイル名のsorted順で対応付けており、family ID照合・
  founder除外・marker ID検証は行っていません。
- 未知アレル・欠損は fillna(0) で埋めており、低分散・MAFフィルターは
  fold内ではなく全個体で適用しています。

この出力および main.py / train_gnn.py の結果は、検証済みOOF性能の証跡として
扱わないでください。
"""


def preprocess_to_numpy():
    data_dir = "./data"
    pheno_files = sorted(glob.glob(os.path.join(data_dir, "*_phenotype_data.tsv.gz")))
    geno_files = sorted(
        glob.glob(os.path.join(data_dir, "*_SNP_genotype_Wm82.a1.tsv.gz"))
    )

    all_y_std = []
    all_X_list = []
    all_family_ids = []  # 個体ごとのファミリーID（CV用）
    mapping = {"A": -1, "B": 1, "H": 0, "A/A": -1, "B/B": 1, "A/B": 0}

    print(f"📊 {len(pheno_files)} 家族のデータを数値化・家族内標準化中...")

    for p_file, g_file in zip(pheno_files, geno_files):
        family_id = os.path.basename(p_file).split("_")[0]

        # 1. 表現型読み込み
        y_df = pd.read_table(p_file, compression="gzip")
        y_df["Yld (kg/ha)"] = pd.to_numeric(y_df["Yld (kg/ha)"], errors="coerce")
        y_df = y_df.dropna(subset=["Yld (kg/ha)", "Corrected Strain"])
        y_df = y_df.drop_duplicates(subset="Corrected Strain")

        # 2. 遺伝型読み込み（転置）
        X_df_raw = pd.read_table(g_file, compression="gzip", index_col=0).T
        X_df_raw = X_df_raw[~X_df_raw.index.duplicated(keep="first")]

        # 3. 同期
        common_strains = y_df["Corrected Strain"].isin(X_df_raw.index)
        y_subset = y_df[common_strains].set_index("Corrected Strain")[["Yld (kg/ha)"]]
        X_subset_raw = X_df_raw.loc[y_subset.index]

        # 【重要】家族内標準化 (Z-score)
        # 家族ごとの平均を0、分散を1に揃えることで環境ノイズを排除
        y_values = y_subset["Yld (kg/ha)"].values
        y_std = (y_values - np.mean(y_values)) / (np.std(y_values) + 1e-6)
        y_subset["Yld (kg/ha)"] = y_std

        # 4. 数値変換 (int8)
        X_numeric = np.zeros(X_subset_raw.shape, dtype=np.int8)
        for i, col in enumerate(X_subset_raw.columns):
            X_numeric[:, i] = X_subset_raw[col].map(mapping).fillna(0).values

        all_y_std.append(y_subset)
        all_X_list.append(X_numeric)
        all_family_ids.extend([family_id] * len(y_subset))

        print(
            f" ✅ {family_id}: {len(y_subset)} 個体完了 (Mean: {np.mean(y_values):.1f})"
        )
        del X_df_raw, X_subset_raw
        gc.collect()

    # 統合
    final_y = pd.concat(all_y_std)
    final_y["family_id"] = all_family_ids
    final_X_array = np.vstack(all_X_list)

    # 5. 低分散SNPの除去（全員同じ値のSNPは予測に役立たないため除外）
    variances = np.var(final_X_array, axis=0)
    valid_snp_mask = variances > 1e-6
    final_X_array = final_X_array[:, valid_snp_mask]
    print(f"低分散SNP除去後のSNP数: {final_X_array.shape[1]}")

    # 6. MAF (Minor Allele Frequency) フィルタリング
    print("\n🧬 MAFフィルタリングを実行中...")
    # Allele 'B' (coded as 1) frequency. 'H' (0) is counted as 0.5 for both alleles.
    p = (
        np.sum(final_X_array == 1, axis=0) + 0.5 * np.sum(final_X_array == 0, axis=0)
    ) / final_X_array.shape[0]
    maf = np.minimum(p, 1 - p)
    maf_threshold = 0.05
    maf_mask = maf > maf_threshold
    final_X_array = final_X_array[:, maf_mask]
    print(f"MAF > {maf_threshold} のSNP数: {final_X_array.shape[1]}")

    # TODO: 連鎖不平衡(LD)プルーニングを実装する
    # 高度に相関するSNPを除去することで、多重共線性を減らしモデルの安定性を向上させることができます。
    # 一般的にはPLINKのような外部ツールが使われます (例: plink --bfile data --indep-pairwise 50 5 0.2)

    # 最終保存
    output_dir = "processed_data_hy"
    os.makedirs(output_dir, exist_ok=True)
    final_y.to_csv(f"{output_dir}/y_phenotype_hy.csv")
    np.save(f"{output_dir}/X_genotype_int8.npy", final_X_array)
    # 出力自体にもexperimentalであることを残す（この成果物を検証済み証跡として
    # 引用させないため）
    with open(f"{output_dir}/EXPERIMENTAL.txt", "w", encoding="utf-8") as handle:
        handle.write(EXPERIMENTAL_NOTICE)

    print("\n✨ 前処理完了！")
    print(
        f"合計個体数: {final_X_array.shape[0]} | 残ったSNP数: {final_X_array.shape[1]}"
    )

    # 相関チェック（全SNPの中から最大相関を探す）
    corrs = [
        np.corrcoef(final_X_array[:, i], final_y.iloc[:, 0].values)[0, 1]
        for i in range(min(500, final_X_array.shape[1]))
    ]
    print(f"最大相関サンプル(先頭500中): {max(corrs, key=abs):.4f}")

    # 注: 線形パス用のゲノム主成分(PC)は、CVリークを防ぐため main.py の各fold内で
    #     train個体のみから学習する（compute_fold_pcs）。ここでは全個体PCAを保存しない。


if __name__ == "__main__":
    # 明示的な --allow-legacy が無い限り、入力読み込みも出力生成も行わない。
    legacy_guard.require_opt_in("preprocess.py", DESCRIPTION)
    preprocess_to_numpy()
