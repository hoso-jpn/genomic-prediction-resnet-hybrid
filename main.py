import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import gc

# 別ファイル model.py に GatedGenomicResNet を定義している前提
from model import GatedGenomicResNet 

# --- 設定：最新の最適化パラメータ ---
config_dict = {
    "lr": 0.0001,           # 同時学習のため、少し慎重な学習率
    "batch_size": 64,       # 更新回数を増やして非線形シグナルを拾う
    "epochs": 150,          
    "l2_reg": 0.05,         # 過学習を防ぐためのWeight Decay
    "folds": 10             
}

def main():
    wandb.init(project="genomic-resnet-prediction-hy", config=config_dict)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ローカルのパス設定
    PROCESSED_DATA_PATH = './processed_data_hy/' 
    
    print(f"🚀 データを読み込み中... (Path: {PROCESSED_DATA_PATH})")
    
    # 1. 表現型データの読み込み (preprocess.py の出力に合わせる)
    y_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_phenotype_hy.csv'), index_col=0)
    y_all = y_df['Yld (kg/ha)'].values.astype(np.float32).reshape(-1, 1)
    
    # 2. 遺伝型データの読み込み (.npy を想定。もし .pkl なら pd.read_pickle に変更)
    # Colabでは .npy を使用していたため、こちらの方が高速です。
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_genotype_int8.npy')
    if os.path.exists(X_path):
        X_all = np.load(X_path).astype(np.float32)
    else:
        # pickle 読み込み + 数値変換 (以前の形式)
        print("🔢 遺伝型データを数値に変換中...")
        X_raw = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, 'X_genotype_hy.pkl'))
        mapping = {'A': 1, 'B': -1, 'H': 0, 'A/A': 1, 'B/B': -1, 'A/B': 0}
        X_raw = X_raw.replace(mapping)
        X_all = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        del X_raw
    
    print(f"✅ 解析開始 | 有効個体数: {len(y_all)} | SNP数: {X_all.shape[1]}")
    
    kf = KFold(n_splits=config.folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        # データの準備
        train_x, test_x = torch.from_numpy(X_all[train_idx]), torch.from_numpy(X_all[test_idx])
        train_y, test_y = torch.from_numpy(y_all[train_idx]), torch.from_numpy(y_all[test_idx])
        
        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        
        # モデル初期化
        model = GatedGenomicResNet(X_all.shape[1]).to(device)
        
        # 線形・非線形を同時に最適化する AdamW
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2_reg)
        criterion = nn.MSELoss()
        
        # --- 学習フェーズ ---
        model.train()
        for epoch in range(config.epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        
        # --- 評価フェーズ ---
        model.eval()
        with torch.no_grad():
            X_test_t = test_x.to(device)
            # 全体予測 (Hybrid)
            y_pred = model(X_test_t).cpu().numpy().flatten()
            # 線形パスのみの予測 (Base)
            y_lin_only = model.linear_path(X_test_t).cpu().numpy().flatten()
            
            y_true = test_y.numpy().flatten()
        
        # 相関係数の算出
        h_acc = np.corrcoef(y_true, y_pred)[0, 1]
        l_acc = np.corrcoef(y_true, y_lin_only)[0, 1]
        
        # ゲートの値 (相乗効果の寄与度)
        gate_val = torch.tanh(model.gate).item()
        
        print(f"Fold {fold+1} | Hybrid: {h_acc:.4f} | Linear Only: {l_acc:.4f} | Gate: {gate_val:.4f}")
        
        # W&Bにログを送信
        wandb.log({
            "fold": fold + 1,
            "accuracy/hybrid": h_acc,
            "accuracy/linear": l_acc,
            "gate_contribution": gate_val,
            "improvement": h_acc - l_acc
        })

        # メモリ解放
        del model, optimizer, train_loader, train_ds
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()