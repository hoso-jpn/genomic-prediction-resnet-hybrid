import copy
import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from torch.utils.data import DataLoader, TensorDataset

import external_logging
import legacy_guard
from losses import CorrelationLoss
from model import GatedGenomicResNet

DESCRIPTION = "legacy ResNet training loop (experimental, not a verified baseline)"
FAMILY_ID_COLUMN = "family_id"
WANDB_PROJECT = "genomic-resnet-prediction-hy"

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
    "kernel_size": 7,
    # 線形パスにゲノム主成分(PC)を使う。PCはリーク防止のため各fold内でtrainのみから学習する。
    "use_pca": True,
    "pca_var_target": 0.90,
    "pca_max_components": 200,
    # 事前学習済みCNN重みのパス。ダミー(ランダム)重みは予測に無益なためデフォルトは None。
    # 実際に事前学習した重みがある場合のみパスを指定する。
    "pretrained_path": None,
    "seed": 42,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_fold_pcs(X_train, X_test, var_target=0.90, max_components=200):
    """SNP空間でPCAを学習し、train/testを主成分スコアに射影する。

    リーク防止のため、標準化統計・主成分ローディングは **train のみ** から推定し、
    同じ変換を test に適用する（GRM固有ベクトルを全個体で計算する旧方式のリークを排除）。
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std

    # 経済的SVD: Xtr = U S Vt。Vt.T がSNP空間の主成分ローディング。
    _, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    explained = S**2
    cum_var = np.cumsum(explained) / np.sum(explained)
    k = min(int(np.searchsorted(cum_var, var_target)) + 1, max_components, Vt.shape[0])

    V = Vt[:k].T  # (n_snp, k)
    train_pc = (Xtr @ V).astype(np.float32)
    test_pc = (Xte @ V).astype(np.float32)
    return train_pc, test_pc


def require_family_ids(y_df):
    """family_id列を必須にする。

    family_idが無い入力でrandom CVへ自動的に切り替えると、family単位の
    汎化性能を求めた実行が、黙って個体ランダム分割の数値を返してしまう。
    暗黙のフォールバックは行わず、ここで明確に失敗させる。
    """
    if FAMILY_ID_COLUMN not in y_df.columns:
        raise RuntimeError(
            f"{FAMILY_ID_COLUMN}列が見つかりません。family単位のgroup-aware評価には"
            f"{FAMILY_ID_COLUMN}が必須です。random CVへ自動的に切り替えることはしません。"
            " family_id付きで processed_data_hy/y_phenotype_hy.csv を再生成するか、"
            "検証済み経路（gblup_baseline.py / resnet_baseline.py）を使用してください。"
        )
    return y_df[FAMILY_ID_COLUMN].values


def build_cv_splitter(strategy, n_folds):
    if strategy == "lofo":
        return LeaveOneGroupOut()
    elif strategy == "group_kfold":
        return GroupKFold(n_splits=n_folds)
    else:
        return KFold(n_splits=n_folds, shuffle=True, random_state=42)


def run_fold(
    fold, train_idx, test_idx, X_all, y_all, config, device, family_label=None
):
    train_x = torch.from_numpy(X_all[train_idx])
    test_x = torch.from_numpy(X_all[test_idx])
    train_y = torch.from_numpy(y_all[train_idx])
    test_y = torch.from_numpy(y_all[test_idx])

    if config.get("use_pca", True):
        # PCはfold内のtrainのみから学習し、同じ変換をtestへ適用（リーク防止）
        train_pc_np, test_pc_np = compute_fold_pcs(
            X_all[train_idx],
            X_all[test_idx],
            var_target=config.get("pca_var_target", 0.90),
            max_components=config.get("pca_max_components", 200),
        )
        train_pc = torch.from_numpy(train_pc_np)
        test_pc = torch.from_numpy(test_pc_np)
    else:
        train_pc = test_pc = None

    rng = np.random.default_rng(seed=fold)
    n_total = len(train_idx)
    n_val = max(1, int(n_total * 0.1))
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[rng.choice(n_total, n_val, replace=False)] = True

    val_x = train_x[val_mask].to(device)
    val_y = train_y[val_mask].to(device)
    val_pc = train_pc[val_mask].to(device) if train_pc is not None else None

    train_x_fit = train_x[~val_mask]
    train_y_fit = train_y[~val_mask]
    train_pc_fit = train_pc[~val_mask] if train_pc is not None else None

    if train_pc_fit is not None:
        train_ds = TensorDataset(train_x_fit, train_y_fit, train_pc_fit)
    else:
        train_ds = TensorDataset(train_x_fit, train_y_fit)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    pc_dim = train_pc.shape[1] if train_pc is not None else None
    model = GatedGenomicResNet(
        X_all.shape[1],
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
        pc_dim=pc_dim,
        kernel_size=config.get("kernel_size", 7),
    ).to(device)

    pretrained_path = config.get("pretrained_path")
    if pretrained_path:
        model.load_pretrained_cnn(pretrained_path)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.l2_reg
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = CorrelationLoss()

    best_val_corr = -float("inf")
    best_state = None
    patience_count = 0
    val_y_np = val_y.cpu().numpy().flatten()

    for epoch in range(config.epochs):  # noqa: B007 (ループ後の stopped_at で使用)
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
        X_test_t = test_x.to(device)
        pc_test_t = test_pc.to(device) if test_pc is not None else None
        y_pred = model(X_test_t, pc_test_t).cpu().numpy().flatten()
        lin_in = pc_test_t if pc_test_t is not None else X_test_t
        y_lin_only = model.linear_path(lin_in).cpu().numpy().flatten()
        y_true = test_y.numpy().flatten()

    h_acc = np.corrcoef(y_true, y_pred)[0, 1]
    l_acc = np.corrcoef(y_true, y_lin_only)[0, 1]
    gate_val = torch.tanh(model.gate).item()

    label = family_label if family_label else f"Fold {fold + 1}"
    stopped_at = epoch + 1
    print(
        f"{label:20s} | Hybrid: {h_acc:.4f} | Linear: {l_acc:.4f} | Gate: {gate_val:.4f} | Stopped: ep{stopped_at}"
    )

    log_dict = {
        "fold": fold + 1,
        "accuracy/hybrid": h_acc,
        "accuracy/linear": l_acc,
        "gate_contribution": gate_val,
        "stopped_epoch": stopped_at,
    }
    if family_label:
        log_dict["test_family"] = family_label

    del model, optimizer, scheduler, train_loader, train_ds, best_state
    torch.cuda.empty_cache()
    gc.collect()

    return h_acc, l_acc, log_dict


def main(argv=None):
    # legacy許可は外部ロギング許可とは独立。--allow-legacy を付けても
    # W&Bは --wandb-mode で明示しない限り初期化しない。
    args = legacy_guard.require_opt_in("main.py", DESCRIPTION, argv)
    logger = external_logging.create_run_logger(
        args.wandb_mode, project=WANDB_PROJECT, config=config_dict
    )
    config = logger.run_config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))

    PROCESSED_DATA_PATH = "./processed_data_hy/"
    print(f"データを読み込み中... (Path: {PROCESSED_DATA_PATH})")

    y_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "y_phenotype_hy.csv"), index_col=0
    )
    y_all = y_df["Yld (kg/ha)"].values.astype(np.float32).reshape(-1, 1)
    X_all = np.load(os.path.join(PROCESSED_DATA_PATH, "X_genotype_int8.npy")).astype(
        np.float32
    )

    family_ids = require_family_ids(y_df)

    strategy = config.cv_strategy
    splitter = build_cv_splitter(strategy, config.folds)

    n_families = len(np.unique(family_ids))
    pc_mode = "fold内PCA(train限定)" if config.get("use_pca", True) else "生SNP"

    print(
        f"解析開始 | 個体数: {len(y_all)} | SNP数: {X_all.shape[1]} | "
        f"ファミリー数: {n_families} | CV戦略: {strategy} | 線形パス入力: {pc_mode}"
    )

    all_h_acc, all_l_acc = [], []
    split_args = (X_all, y_all, family_ids) if strategy != "random" else (X_all,)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        family_label = None
        if strategy == "lofo":
            family_label = str(np.unique(family_ids[test_idx])[0])

        h_acc, l_acc, log_dict = run_fold(
            fold, train_idx, test_idx, X_all, y_all, config, device, family_label
        )
        logger.log(log_dict)
        all_h_acc.append(h_acc)
        all_l_acc.append(l_acc)

    mean_h = np.mean(all_h_acc)
    mean_l = np.mean(all_l_acc)
    print("\n" + "=" * 55)
    print(
        f"平均 Hybrid: {mean_h:.4f} | 平均 Linear: {mean_l:.4f} | 改善: {mean_h - mean_l:+.4f}"
    )
    print(legacy_guard.EXPERIMENTAL_BANNER)
    print(
        "[EXPERIMENTAL] 上記の指標はfamily内標準化済み表現型に対するものです（kg/haではありません）。"
    )
    logger.log(
        {
            "summary/mean_hybrid": mean_h,
            "summary/mean_linear": mean_l,
            "summary/improvement": mean_h - mean_l,
        }
    )
    logger.finish()


if __name__ == "__main__":
    main()
