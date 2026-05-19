import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
import numpy as np
import pandas as pd
import os
import gc
import copy

from model import GatedGenomicResNet

# cv_strategy:
#   "random"      : 従来のランダムKFold（ファミリー構造無視）
#   "group_kfold" : ファミリー単位でグループ化したKFold
#   "lofo"        : Leave-One-Family-Out（未知ファミリーへの真の汎化性評価）
config_dict = {
    "lr": 0.0001,
    "batch_size": 64,
    "epochs": 150,
    "l2_reg": 0.05,
    "folds": 10,
    "cv_strategy": "lofo",
    "early_stopping_patience": 40,
    "hidden_dim": 256,
    "num_blocks": 3,
    "dropout_rate": 0.4,
}


def build_cv_splitter(strategy, n_folds):
    if strategy == "lofo":
        return LeaveOneGroupOut()
    elif strategy == "group_kfold":
        return GroupKFold(n_splits=n_folds)
    else:
        return KFold(n_splits=n_folds, shuffle=True, random_state=42)


def run_fold(fold, train_idx, test_idx, X_all, y_all, G_pc, config, device, family_label=None):
    train_x = torch.from_numpy(X_all[train_idx])
    test_x  = torch.from_numpy(X_all[test_idx])
    train_y = torch.from_numpy(y_all[train_idx])
    test_y  = torch.from_numpy(y_all[test_idx])

    # G行列固有ベクトルのスライス（G_pc が None のときは SNP をそのまま使う）
    if G_pc is not None:
        train_pc = torch.from_numpy(G_pc[train_idx])
        test_pc  = torch.from_numpy(G_pc[test_idx])
    else:
        train_pc = test_pc = None

    # Early stopping 用に訓練データの10%をランダムにバリデーションへ分割
    # ※ 末尾固定切り出しはデータがファミリー順に並んでいるため偏りが生じる
    rng = np.random.default_rng(seed=fold)
    n_total = len(train_idx)
    n_val = max(1, int(n_total * 0.1))
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[rng.choice(n_total, n_val, replace=False)] = True

    val_x  = train_x[val_mask].to(device)
    val_y  = train_y[val_mask].to(device)
    val_pc = train_pc[val_mask].to(device) if train_pc is not None else None

    train_x_fit  = train_x[~val_mask]
    train_y_fit  = train_y[~val_mask]
    train_pc_fit = train_pc[~val_mask] if train_pc is not None else None

    if train_pc_fit is not None:
        train_ds = TensorDataset(train_x_fit, train_y_fit, train_pc_fit)
    else:
        train_ds = TensorDataset(train_x_fit, train_y_fit)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    pc_dim = G_pc.shape[1] if G_pc is not None else None
    model = GatedGenomicResNet(
        X_all.shape[1],
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
        pc_dim=pc_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = CorrelationLoss()

    best_val_corr = -float('inf')
    best_state    = None
    patience_count = 0
    val_y_np = val_y.cpu().numpy().flatten()

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            if train_pc_fit is not None:
                batch_X, batch_y, batch_pc = [t.to(device) for t in batch]
                pred = model(batch_X, batch_pc)
            else:
                batch_X, batch_y = [t.to(device) for t in batch]
                pred = model(batch_X)
            optimizer.zero_grad()
            criterion(pred, batch_y).backward()
            optimizer.step()
        scheduler.step()

        # Early stopping: 評価指標（Pearson r）と統一
        model.eval()
        with torch.no_grad():
            val_pred_np = model(val_x, val_pc).cpu().numpy().flatten()
        val_corr = np.corrcoef(val_y_np, val_pred_np)[0, 1]
        if np.isnan(val_corr):
            val_corr = -1.0
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.early_stopping_patience:
                model.load_state_dict(best_state)
                break

    model.eval()
    with torch.no_grad():
        X_test_t  = test_x.to(device)
        pc_test_t = test_pc.to(device) if test_pc is not None else None
        y_pred     = model(X_test_t, pc_test_t).cpu().numpy().flatten()
        lin_in     = pc_test_t if pc_test_t is not None else X_test_t
        y_lin_only = model.linear_path(lin_in).cpu().numpy().flatten()
        y_true     = test_y.numpy().flatten()

    h_acc    = np.corrcoef(y_true, y_pred)[0, 1]
    l_acc    = np.corrcoef(y_true, y_lin_only)[0, 1]
    gate_val = torch.tanh(model.gate).item()

    label      = family_label if family_label else f"Fold {fold + 1}"
    stopped_at = epoch + 1
    print(f"{label:20s} | Hybrid: {h_acc:.4f} | Linear: {l_acc:.4f} | Gate: {gate_val:.4f} | Stopped: ep{stopped_at}")

    log_dict = {
        "fold":               fold + 1,
        "accuracy/hybrid":    h_acc,
        "accuracy/linear":    l_acc,
        "gate_contribution":  gate_val,
        "stopped_epoch":      stopped_at,
    }
    if family_label:
        log_dict["test_family"] = family_label

    del model, optimizer, scheduler, train_loader, train_ds, best_state
    torch.cuda.empty_cache()
    gc.collect()

    return h_acc, l_acc, log_dict


def main():
    wandb.init(project="genomic-resnet-prediction-hy", config=config_dict)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROCESSED_DATA_PATH = './processed_data_hy/'

    print(f"データを読み込み中... (Path: {PROCESSED_DATA_PATH})")

    y_df  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_phenotype_hy.csv'), index_col=0)
    y_all = y_df['Yld (kg/ha)'].values.astype(np.float32).reshape(-1, 1)
    X_all = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_genotype_int8.npy')).astype(np.float32)

    # ファミリーIDの読み込み
    if 'family_id' in y_df.columns:
        family_ids = y_df['family_id'].values
    else:
        # 再前処理前の旧データ互換: family_id列がない場合はランダムCVにフォールバック
        print("警告: family_id列が見つかりません。cv_strategy を 'random' に変更します。")
        print("      preprocess.py を再実行して processed_data_hy/ を更新してください。")
        family_ids = None
        if config.cv_strategy != "random":
            wandb.config.update({"cv_strategy": "random"}, allow_val_change=True)

    strategy = config.cv_strategy if family_ids is not None else "random"
    splitter  = build_cv_splitter(strategy, config.folds)

    n_families = len(np.unique(family_ids)) if family_ids is not None else "N/A"
    # G行列固有ベクトルの読み込み（preprocess.py で生成）
    G_pc_path = os.path.join(PROCESSED_DATA_PATH, 'G_eigenvec.npy')
    if os.path.exists(G_pc_path):
        G_pc = np.load(G_pc_path)
        print(f"G固有ベクトル読み込み完了 (shape: {G_pc.shape}) → 線形パスを GBLUP 近似モードで実行")
    else:
        G_pc = None
        print("G_eigenvec.npy が未生成のため、線形パスに生SNPを使用します。")
        print("  → docker compose run --rm preprocess を実行すると G 固有ベクトルが生成されます。")

    print(f"解析開始 | 個体数: {len(y_all)} | SNP数: {X_all.shape[1]} | "
          f"ファミリー数: {n_families} | CV戦略: {strategy}")

    all_h_acc, all_l_acc = [], []

    split_args = (X_all, y_all, family_ids) if strategy != "random" else (X_all,)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        family_label = None
        if strategy == "lofo" and family_ids is not None:
            family_label = str(np.unique(family_ids[test_idx])[0])

        h_acc, l_acc, log_dict = run_fold(
            fold, train_idx, test_idx, X_all, y_all, G_pc, config, device, family_label
        )
        wandb.log(log_dict)
        all_h_acc.append(h_acc)
        all_l_acc.append(l_acc)

    mean_h = np.mean(all_h_acc)
    mean_l = np.mean(all_l_acc)
    print(f"\n{'='*55}")
    print(f"平均 Hybrid: {mean_h:.4f} | 平均 Linear: {mean_l:.4f} | 改善: {mean_h - mean_l:+.4f}")
    wandb.log({"summary/mean_hybrid": mean_h, "summary/mean_linear": mean_l,
               "summary/improvement": mean_h - mean_l})


if __name__ == "__main__":
    main()
fold(
            fold, train_idx, test_idx, X_all, y_all, G_pc, config, device, family_label
        )
        wandb.log(log_dict)
        all_h_acc.append(h_acc)
        all_l_acc.append(l_acc)

    mean_h = np.mean(all_h_acc)
    mean_l = np.mean(all_l_acc)
    print(f"\n{'='*55}")
    print(f"平均 Hybrid: {mean_h:.4f} | 平均 Linear: {mean_l:.4f} | 改善: {mean_h - mean_l:+.4f}")
    wandb.log({"summary/mean_hybrid": mean_h, "summary/mean_linear": mean_l,
               "summary/improvement": mean_h - mean_l})


if __name__ == "__main__":
    main()
