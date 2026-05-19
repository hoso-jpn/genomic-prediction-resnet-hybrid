import torch
import torch.nn as nn

class ConvResidualBlock(nn.Module):
    """1D Convolutional Residual Block."""
    def __init__(self, channels, kernel_size=7, dropout_rate=0.4):
        super().__init__()
        # 膨張畳み込みなども検討可能
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class GatedGenomicResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_blocks=3, dropout_rate=0.4, pc_dim=None, kernel_size=7):
        super().__init__()
        self.pc_dim = pc_dim

        # 1. 線形パス: G行列固有ベクトル or 生SNP
        lin_in = pc_dim if pc_dim is not None else input_dim
        self.linear_path = nn.Linear(lin_in, 1)

        # 2. 1D CNN ResNet パス: 常に生SNPを入力 (N, L)
        self.cnn_path = nn.Sequential(
            # (N, L) -> (N, 1, L)
            nn.Unsqueeze(1),
            # (N, 1, L) -> (N, hidden_dim, L)
            nn.Conv1d(1, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            # (N, hidden_dim, L) -> (N, hidden_dim, L)
            *[ConvResidualBlock(hidden_dim, kernel_size=kernel_size, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )

        # Global Average Pooling + 全結合
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.nonlinear_output = nn.Linear(hidden_dim, 1, bias=False)

        # 3. 学習可能なゲート
        self.gate = nn.Parameter(torch.tensor([0.3]))

    def forward(self, x_snp, x_pc=None):
        # 線形パス
        lin_in = x_pc if (self.pc_dim is not None and x_pc is not None) else x_snp
        lin_out = self.linear_path(lin_in)

        # CNNパス
        # x_snp: (N, L)
        res_out = self.cnn_path(x_snp)
        
        # Global Average Pooling: (N, C, L) -> (N, C, 1) -> (N, C)
        pooled_out = self.global_avg_pool(res_out).squeeze(-1)
        
        nonlin_out = self.nonlinear_output(pooled_out)

        return lin_out + (nonlin_out * torch.tanh(self.gate))
