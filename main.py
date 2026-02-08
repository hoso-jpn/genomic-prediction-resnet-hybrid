import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold
import numpy as np
from model import GenomicResNet
from utils import load_genomic_data, calculate_gblup_residuals

# 1. W&B初期化
wandb.init(project="genomic-resnet-prediction", config={
    "lr": 0.0001,           # Dropout導入に伴い、学習率を少し調整
    "weight_decay": 0.1,
    "epochs": 100,
    "repeats": 20,
    "folds": 10,
    "dropout_rate": 0.4     # configから変更可能に
})

def main():
    X, y_multi, strain_ids = load_genomic_data('data/4J105-3-4_pheno.csv', 'data/4J105-3-4_geno.csv')
    config = wandb.config
    
    all_hybrid_acc = []
    all_mt_gblup_acc = [] # MT-GBLUP比較用

    for r in range(config.repeats):
        kf = KFold(n_splits=config.folds, shuffle=True, random_state=42+r)
        hybrid_fold_acc = []
        mt_fold_acc = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # GBLUPの計算 (u_mtはMT-GBLUP単体の予測値)
            u_mt, res_yield = calculate_gblup_residuals(X, y_multi, train_idx, test_idx, strain_ids)
            y_true = y_multi.iloc[test_idx]['Yield'].values
            
            # --- MT-GBLUP単体の精度計算 ---
            mt_acc = np.corrcoef(y_true, u_mt[test_idx])[0, 1]
            mt_fold_acc.append(mt_acc)
            
            # ResNet学習
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = GenomicResNet(X.shape[1]).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            criterion = nn.MSELoss()
            
            X_train = torch.tensor(X[train_idx]).to(device)
            y_res = torch.tensor(res_yield[train_idx].reshape(-1,1), dtype=torch.float32).to(device)
            
            model.train()
            for epoch in range(config.epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_res)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    wandb.log({"train_loss": loss.item(), "repeat_idx": r, "fold_idx": fold})
            
            # --- Hybrid (MT-GBLUP + ResNet) の予測 ---
            model.eval()
            with torch.no_grad():
                dl_res_pred = model(torch.tensor(X[test_idx]).to(device)).cpu().numpy().flatten()
            
            hybrid_pred = u_mt[test_idx] + dl_res_pred
            h_acc = np.corrcoef(y_true, hybrid_pred)[0, 1]
            hybrid_fold_acc.append(h_acc)
            
        # リピートごとの平均精度
        avg_h_acc = np.mean(hybrid_fold_acc)
        avg_mt_acc = np.mean(mt_fold_acc)
        
        all_hybrid_acc.append(avg_h_acc)
        all_mt_gblup_acc.append(avg_mt_acc)
        
        # W&Bに両方の精度をログ
        wandb.log({
            "repeat": r + 1,
            "accuracy/hybrid": avg_h_acc,
            "accuracy/mt_gblup": avg_mt_acc,
            "mean_so_far/hybrid": np.mean(all_hybrid_acc),
            "mean_so_far/mt_gblup": np.mean(all_mt_gblup_acc)
        })
        
        print(f"Repeat {r+1}/{config.repeats} | Hybrid: {avg_h_acc:.4f} | MT-GBLUP: {avg_mt_acc:.4f}")

if __name__ == "__main__":
    main()
