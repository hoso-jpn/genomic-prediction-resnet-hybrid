# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
import pandas as pd
import os
import gc
import copy
from model import GraphGenomicNet
from main import CorrelationLoss

# --- Config ---
config_dict = {
    "lr": 0.001, "batch_size": 32, "epochs": 10, "l2_reg": 0.01,
    "hidden_dim": 128, "num_layers": 3, "dropout_rate": 0.4, "num_genes": 500
}
PROCESSED_DATA_PATH = './processed_data_hy/'

# --- データ読み込み ---
def load_data():
    print("DEBUG: Loading data now.")
    y_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_phenotype_hy.csv'), index_col=0)
    y_all = y_df['Yld (kg/ha)'].values.astype(np.float32).reshape(-1, 1)
    X_all = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_genotype_int8.npy')).astype(np.float32)
    family_ids = y_df['family_id'].values
    snp_map_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'snp_to_gene_map.csv'))
    adj_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'gene_adj.csv'))
    edge_index = torch.tensor(adj_df.values, dtype=torch.long).t().contiguous()
    # 無向グラフとして扱うため逆方向エッジを追加 (GCNConvは対称グラフを前提とする)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    snp_to_gene_map = torch.tensor(snp_map_df['gene_id'].values, dtype=torch.long)
    print("DEBUG: Data loading complete.")
    return X_all, y_all, family_ids, snp_to_gene_map, edge_index

# --- 訓練ループ ---
def run_gnn_training():
    #wandb.init(project="genomic-gnn-prediction", config=config_dict)
    #config = wandb.config
    print("DEBUG: wandb disabled for this test run.")
    config = config_dict
    device = torch.device("cpu")

    X_all, y_all, family_ids, snp_to_gene_map, edge_index = load_data()
    
    logo = LeaveOneGroupOut()
    all_test_corrs = []

    print("DEBUG: Starting CV loop.")
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups=family_ids)):
        print("DEBUG: Fold start.")

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        model = GraphGenomicNet(
            num_genes=config['num_genes'], hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'], dropout_rate=config['dropout_rate']
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['l2_reg'])
        criterion = CorrelationLoss().to(device)
        
        edge_index_dev = edge_index.to(device)
        snp_to_gene_map_dev = snp_to_gene_map.to(device)

        print("DEBUG: Epoch loop starting.")
        for epoch in range(config['epochs']):
            model.train()
            permutation = torch.randperm(X_train.shape[0])
            for i in range(0, X_train.shape[0], config['batch_size']):
                indices = permutation[i:i+config['batch_size']]
                batch_X = torch.from_numpy(X_train[indices]).to(device).flatten()
                batch_y = torch.from_numpy(y_train[indices]).to(device)
                N = batch_y.size(0)
                batch_snp_map = snp_to_gene_map_dev.repeat(N) + torch.arange(N, device=device).repeat_interleave(X_train.shape[1]) * config['num_genes']
                batch_indices = torch.arange(N, device=device).repeat_interleave(config['num_genes'])
                optimizer.zero_grad()
                pred = model(batch_X, batch_snp_map, edge_index_dev, batch_indices)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print("DEBUG: Epoch milestone reached.")

        print("DEBUG: Evaluation start.")
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).to(device).flatten()
            N_test = X_test.shape[0]
            test_snp_map = snp_to_gene_map_dev.repeat(N_test) + torch.arange(N_test, device=device).repeat_interleave(X_test.shape[1]) * config['num_genes']
            test_batch_indices = torch.arange(N_test, device=device).repeat_interleave(config['num_genes'])
            y_pred = model(X_test_t, test_snp_map, edge_index_dev, test_batch_indices).cpu().numpy().flatten()
            y_true = y_test.flatten()

        test_corr = np.corrcoef(y_true, y_pred)[0, 1]
        print("DEBUG: Fold test correlation calculated.")
        all_test_corrs.append(test_corr)
        del model, optimizer
        gc.collect()

    print("DEBUG: CV Finished.")
    mean_corr = np.nanmean(all_test_corrs)
    print("DEBUG: Final result calculated.")
    # wandb.log({"summary/mean_correlation": mean_corr})

if __name__ == "__main__":
    run_gnn_training()
