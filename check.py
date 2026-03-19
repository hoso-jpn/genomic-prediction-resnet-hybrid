import pandas as pd
import glob
import os

# データディレクトリの設定
data_dir = './data'
pheno_files = sorted(glob.glob(os.path.join(data_dir, '*_phenotype_data.tsv.gz')))
geno_files = sorted(glob.glob(os.path.join(data_dir, '*_SNP_genotype_Wm82.a1.tsv.gz')))

if not pheno_files or not geno_files:
    print("❌ ファイルが見つかりません。パスを確認してください。")
    exit()

def check_identity(idx):
    p_file = pheno_files[idx]
    g_file = geno_files[idx]
    family_id = os.path.basename(p_file).split('_')[0]
    
    print(f"\n--- 家族 [ {family_id} ] の照合レポート ---")
    
    # 表現型データの読み込み
    y_df = pd.read_table(p_file, compression='gzip')
    y_names = set(y_df['Corrected Strain'].dropna().unique())
    
    # 遺伝型データの読み込み（ヘッダーのみ高速読み込み）
    x_df_head = pd.read_table(g_file, compression='gzip', nrows=1, index_col=0)
    x_names = set(x_df_head.columns)
    
    common = y_names & x_names
    
    print(f"📋 表現型内の個体数: {len(y_names)}")
    print(f"🧬 遺伝型内の個体数: {len(x_names)}")
    print(f"🤝 一致した個体数: {len(common)}")
    
    if len(common) == 0:
        print("⚠️ 警告: 一致する個体がゼロです！")
        print(f"  表現型の例: {list(y_names)[:3]}")
        print(f"  遺伝型の例: {list(x_names)[:3]}")
    elif len(common) < len(y_names) * 0.5:
        print("⚠️ 警告: 一致率が50%以下です。名前の形式（スペース等）を確認してください。")

# 最初の家族と、適当な中間の家族をチェック
check_identity(0)
if len(pheno_files) > 1:
    check_identity(len(pheno_files) // 2)