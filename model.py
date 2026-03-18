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
        # 以前の残差接続を維持
        return x + self.block(x)

class GenomicResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        
        # --- Wideパス (線形モデル) ---
        # 入力(SNPデータ)から直接出力へ。RR-BLUP的な役割を担います。
        self.wide_linear = nn.Linear(input_dim, 1)

        # --- Deepパス (ResNet) ---
        # 複雑な相互作用を学習します。
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 1. Wideパスの計算
        wide_out = self.wide_linear(x)
        
        # 2. Deepパスの計算
        deep_out = self.input_layer(x)
        deep_out = self.res_blocks(deep_out)
        deep_out = self.output_layer(deep_out)
        
        # 3. 両方の出力を足し合わせる (ハイブリッド)
        return wide_out + deep_out