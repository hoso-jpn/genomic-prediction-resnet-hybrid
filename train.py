import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import GenomicResNet  # 先ほど修正したハイブリッド版

def train():
    # 1. ハイパーパラメータの設定
    INPUT_DIM = 1000  # 実際のSNP数に合わせて変更してください
    HIDDEN_DIM = 256
    NUM_BLOCKS = 3
    BATCH_SIZE = 16   # EliteBookのメモリに優しいサイズ
    EPOCHS = 20
    LEARNING_RATE = 0.001

    # 2. ダミーデータの生成 (本番はここを実データ読み込みに差し替え)
    # 実データがある場合は torch.from_numpy(genotype_data) などを使います
    X_train = torch.randn(100, INPUT_DIM) 
    y_train = torch.randn(100, 1)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. モデル・損失関数・最適化手法の定義
    model = GenomicResNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS)
    criterion = nn.MSELoss()  # 表現型の数値予測用
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 学習ループ
    print(f"Starting training on {INPUT_DIM} SNPs...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(loader):.4f}")

    print("Training Complete!")
    # 学習済みモデルの保存
    torch.save(model.state_dict(), "genomic_resnet_hybrid.pth")

if __name__ == "__main__":
    train()