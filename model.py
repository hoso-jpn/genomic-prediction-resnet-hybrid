import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

# クラス名を main.py のインポート名に合わせる
class GatedGenomicResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        
        # 1. Wideパス (線形モデル: RR-BLUP的役割)
        self.linear_path = nn.Linear(input_dim, 1)

        # 2. Deepパス (ResNet: 非線形相互作用の抽出)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        # 複数の残差ブロックを重ねる
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.nonlinear_output = nn.Linear(hidden_dim, 1, bias=False)

        # 3. 学習可能なゲート (非線形パスの寄与度を調整)
        self.gate = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        # 線形出力
        lin_out = self.linear_path(x)
        
        # 非線形出力
        res_out = self.input_layer(x)
        res_out = self.res_blocks(res_out)
        nonlin_out = self.nonlinear_output(res_out)
        
        # ゲートを適用して統合
        return lin_out + (nonlin_out * torch.tanh(self.gate))