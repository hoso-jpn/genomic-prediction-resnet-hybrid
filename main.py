import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold
import numpy as np
from scipy import stats
from model import GenomicResNet
from utils import load_genomic_data, calculate_gblup_residuals

# 1. W&B初期化（設定を最適化）
wandb.init(project="genomic-resnet-prediction", config={
    "lr": 0.005,             # 0.0001から引き上げ
    "weight_decay": 0.05,    # 正則化を少し調整
    "epochs": 250,           # 残差学習のために少し延長
    "repeats": 10,
    "folds": 10,
    "dropout_rate": 0.2      # 適合力を上げるため少し下げる
})

def main():
    X, y_multi, strain_ids = load_genomic_data('processed_data/y_phenotype.csv', 'processed_data/X_genotype.pkl')
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_hybrid_acc = []
    all_mt_gblup_acc = []

    print(f"解析開始: {config.repeats}反復 x {config.folds}分割 CV")

    for r in range(config.repeats):
        kf = KFold(n_splits=config.folds, shuffle=True, random_state=42+r)
        hybrid_fold_acc = []
        mt_fold_acc = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # 1. GBLUP計算
            u_mt, res_yield = calculate_gblup_residuals(X, y_multi, train_idx, test_idx, strain_ids)
            y_true = y_multi.iloc[test_idx]['Yield'].values
            
            # --- MT-GBLUP単体の精度 ---
            mt_acc = np.corrcoef(y_true, u_mt[test_idx])[0, 1]
            mt_fold_acc.append(mt_acc)
            
            # --- 2. ResNet学習 (ここが重要) ---
            # 残差を標準化 (平均0, 分散1) して学習しやすくする
            res_train = res_yield[train_idx]
            res_mean = res_train.mean()
            res_std = res_train.std() + 1e-6
            y_res_scaled = (res_train - res_mean) / res_std
            
            model = GenomicResNet(X.shape[1]).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            criterion = nn.MSELoss()
            
            X_train_tensor = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
            y_res_tensor = torch.tensor(y_res_scaled.reshape(-1,1), dtype=torch.float32).to(device)
            
            model.train()
            for epoch in range(config.epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_res_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 50 == 0:
                    wandb.log({"train_loss": loss.item(), "repeat": r, "fold": fold})
            
            # --- 3. Hybrid 予測 (スケールを戻す) ---
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
                dl_res_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
                # 標準化を逆算して元に戻す
                dl_res_pred = (dl_res_pred_scaled * res_std) + res_mean
            
            hybrid_pred = u_mt[test_idx] + dl_res_pred
            h_acc = np.corrcoef(y_true, hybrid_pred)[0, 1]
            hybrid_fold_acc.append(h_acc)

            # メモリ管理
            del X_train_tensor, y_res_tensor, X_test_tensor, model
            if device.type == 'cuda': torch.cuda.empty_cache()
            
        # リピートごとの集計
        avg_h_acc = np.mean(hybrid_fold_acc)
        avg_mt_acc = np.mean(mt_fold_acc)
        all_hybrid_acc.append(avg_h_acc)
        all_mt_gblup_acc.append(avg_mt_acc)
        
        # 統計的有意差
        t_stat, p_val = stats.ttest_rel(all_hybrid_acc, all_mt_gblup_acc) if len(all_hybrid_acc) > 1 else (0, 1)
        
        wandb.log({
            "repeat": r + 1,
            "accuracy/hybrid": avg_h_acc,
            "accuracy/mt_gblup": avg_mt_acc,
            "p_value": p_val
        })
        
        print(f"Repeat {r+1}/{config.repeats} | Hybrid: {avg_h_acc:.4f} | GBLUP: {avg_mt_acc:.4f} | p-val: {p_val:.4f}")

    print("\n--- 最終解析完了 ---")
    print(f"Hybrid 平均: {np.mean(all_hybrid_acc):.4f} | GBLUP 平均: {np.mean(all_mt_gblup_acc):.4f}")

if __name__ == "__main__":
    main()