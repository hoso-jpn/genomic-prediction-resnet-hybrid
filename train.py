import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb

# 1. ハイパーパラメータ設定
config = {
    "lr": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "dropout": 0.3,
    "hidden_dim": 64
}

wandb.init(project="soybean-resnet-hybrid", config=config)

# 2. データの読み込みと「欠測値」の徹底除去
# インデックスを揃えるため、一度DataFrameとして読み込みます
X_df = pd.read_pickle("processed_data/X_genotype.pkl")
df_y = pd.read_csv("processed_data/y_phenotype.csv")

# 収量データの '.' を NaN に変換し、数値型へ強制変換
df_y['Yld (kg/ha)'] = pd.to_numeric(df_y['Yld (kg/ha)'], errors='coerce')

#X_df と df_y をインデックスで結合（Inner Join）
if len(X_df) != len(df_y):
    print(f"警告: 行数が不一致です (X: {len(X_df)}, y: {len(df_y)})。Xのサイズに合わせてyを切り出します。")
    # SoyNAMの個体IDなど共通の列がある場合は merge を推奨しますが、
    # preprocess.py の順序が維持されているなら、単純に先頭から切り出します。
    df_y = df_y.iloc[:len(X_df)]

# 収量データが存在する行（NaNでない行）だけを抽出
valid_indices = df_y['Yld (kg/ha)'].notna()
df_y_clean = df_y[valid_indices].reset_index(drop=True)
X_clean = X_df[valid_indices.values].values.astype(np.float32)

# 学習ターゲットとWide系情報（家系ID）の準備
y = df_y_clean['Yld (kg/ha)'].values.astype(np.float32)
wide_info = pd.get_dummies(df_y_clean['Family_ID']).values.astype(np.float32)

print(f"同期完了。有効個体数: {len(y)} / 元データ: {len(X_df)}")

# 3. 前処理：収量の標準化
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# データの分割
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_clean, y_scaled, wide_info, test_size=0.2, random_state=42
)

# TensorDataset化
train_ds = TensorDataset(
    torch.FloatTensor(X_train).unsqueeze(1), 
    torch.FloatTensor(w_train), 
    torch.FloatTensor(y_train)
)
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

# 4. モデル定義：Hybrid ResNet
class GenomicResNet(nn.Module):
    def __init__(self, input_dim, wide_dim):
        super(GenomicResNet, self).__init__()
        # Deep側: ResNet
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        
        # Wide側
        self.wide_layer = nn.Linear(wide_dim, 8)
        
        # 統合出力層 (GELU採用)
        self.fc = nn.Sequential(
            nn.Linear(16 * input_dim + 8, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], 1)
        )

    def forward(self, x_deep, x_wide):
        identity = x_deep
        out = torch.relu(self.bn1(self.conv1(x_deep)))
        out = self.bn2(self.conv2(out))
        out += identity 
        out = torch.flatten(out, 1)
        
        w_out = torch.relu(self.wide_layer(x_wide))
        combined = torch.cat((out, w_out), dim=1)
        return self.fc(combined)

# 5. 学習ループ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenomicResNet(input_dim=X_clean.shape[1], wide_dim=wide_info.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.MSELoss()

print("学習を開始します...")
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for bx, bw, by in train_loader:
        bx, bw, by = bx.to(device), bw.to(device), by.to(device)
        optimizer.zero_grad()
        pred = model(bx, bw)
        loss = criterion(pred.flatten(), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    wandb.log({"epoch": epoch, "loss": avg_loss})
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

print("学習完了！")
wandb.finish()