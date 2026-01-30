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
    "lr": 0.0005,
    "weight_decay": 0.02,
    "epochs": 80,
    "repeats": 50,
    "folds": 10
})

def main():
    X, y_multi = load_genomic_data('data/4J105-3-4_pheno.csv', 'data/4J105-3-4_geno.csv')
    strain_ids = [f"S{i}" for i in range(len(X))]
    config = wandb.config
    
    all_hybrid_acc = []

    for r in range(config.repeats):
        kf = KFold(n_splits=config.folds, shuffle=True, random_state=42+r)
        fold_acc = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # GBLUPの計算
            u_mt, res_yield = calculate_gblup_residuals(X, y_multi, train_idx, test_idx, strain_ids)
            
            # ResNet学習
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = GenomicResNet(X.shape[1]).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            
            X_train = torch.tensor(X[train_idx]).to(device)
            y_res = torch.tensor(res_yield[train_idx].reshape(-1,1), dtype=torch.float32).to(device)
            
            model.train()
            for _ in range(config.epochs):
                optimizer.zero_grad()
                nn.MSELoss()(model(X_train), y_res).backward()
                optimizer.step()
            
            # 予測
            model.eval()
            with torch.no_grad():
                dl_res_pred = model(torch.tensor(X[test_idx]).to(device)).cpu().numpy().flatten()
            
            hybrid_pred = u_mt[test_idx] + dl_res_pred
            y_true = y_multi.iloc[test_idx]['Yield'].values
            acc = np.corrcoef(y_true, hybrid_pred)[0, 1]
            fold_acc.append(acc)
            
        avg_acc = np.mean(fold_acc)
        all_hybrid_acc.append(avg_acc)
        wandb.log({"repeat_accuracy": avg_acc, "mean_so_far": np.mean(all_hybrid_acc)})
        print(f"Repeat {r+1}/{config.repeats} - Acc: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
