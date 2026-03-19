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

# 1. 自作モデルのインポート（model.pyにGatedGenomicResNetがある前提）
from model import GatedGenomicResNet 

# ハイパーパラメータ設定
config_dict = {
    "lr": 0.0001,
    "batch_size": 64,
    "epochs": 150,
    "l2_reg": 0.05,
    "folds": 10
}

# --- ここが定義されていないのがエラーの原因でした ---
def main():
    wandb.init(project="genomic-resnet-prediction-hy", config=config_dict)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PROCESSED_DATA_PATH = './processed_data_hy/' 
    
    print(f"🚀 データを読み込み中... (Path: {PROCESSED_DATA_PATH})")
    
    y_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_phenotype_hy.csv'), index_col=0)
    y_all = y_df['Yld (kg/ha)'].values.astype(np.float32).reshape(-1, 1)
    
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_genotype_int8.npy')
    X_all = np.load(X_path).astype(np.float32)
    
    print(f"✅ 解析開始 | 有効個体数: {len(y_all)} | SNP数: {X_all.shape[1]}")
    
    kf = KFold(n_splits=config.folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        train_x, test_x = torch.from_numpy(X_all[train_idx]), torch.from_numpy(X_all[test_idx])
        train_y, test_y = torch.from_numpy(y_all[train_idx]), torch.from_numpy(y_all[test_idx])
        
        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        
        model = GatedGenomicResNet(X_all.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2_reg)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(config.epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_test_t = test_x.to(device)
            y_pred = model(X_test_t).cpu().numpy().flatten()
            y_lin_only = model.linear_path(X_test_t).cpu().numpy().flatten()
            y_true = test_y.numpy().flatten()
        
        h_acc = np.corrcoef(y_true, y_pred)[0, 1]
        l_acc = np.corrcoef(y_true, y_lin_only)[0, 1]
        gate_val = torch.tanh(model.gate).item()
        
        print(f"Fold {fold+1} | Hybrid: {h_acc:.4f} | Linear Only: {l_acc:.4f} | Gate: {gate_val:.4f}")
        
        wandb.log({
            "fold": fold + 1,
            "accuracy/hybrid": h_acc,
            "accuracy/linear": l_acc,
            "gate_contribution": gate_val
        })

        del model, optimizer, train_loader, train_ds
        torch.cuda.empty_cache()
        gc.collect()

# --- ファイルの最後で main() を呼び出す ---
if __name__ == "__main__":
    main()