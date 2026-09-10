import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class _Unsqueeze(nn.Module):
    """nn.Sequential 内で unsqueeze を使うためのユーティリティ (nn.Unsqueeze は存在しないため)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class ConvResidualBlock(nn.Module):
    """1D Convolutional Residual Block."""

    def __init__(self, channels, kernel_size=7, dropout_rate=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GatedGenomicResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_blocks=3,
        dropout_rate=0.4,
        pc_dim=None,
        kernel_size=7,
    ):
        super().__init__()
        self.pc_dim = pc_dim

        lin_in = pc_dim if pc_dim is not None else input_dim
        self.linear_path = nn.Linear(lin_in, 1)

        self.cnn_path = nn.Sequential(
            _Unsqueeze(1),
            nn.Conv1d(1, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            *[
                ConvResidualBlock(
                    hidden_dim, kernel_size=kernel_size, dropout_rate=dropout_rate
                )
                for _ in range(num_blocks)
            ],
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.nonlinear_output = nn.Linear(hidden_dim, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor([0.3]))

    def load_pretrained_cnn(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.cnn_path.load_state_dict(state, strict=False)

    def forward(self, x_snp, x_pc=None):
        lin_in = x_pc if (self.pc_dim is not None and x_pc is not None) else x_snp
        lin_out = self.linear_path(lin_in)

        res_out = self.cnn_path(x_snp)
        pooled_out = self.global_avg_pool(res_out).squeeze(-1)
        nonlin_out = self.nonlinear_output(pooled_out)

        return lin_out + nonlin_out * torch.tanh(self.gate)


class GraphGenomicNet(nn.Module):
    """
    SNP情報を遺伝子グラフ上で集約・伝播するGNNモデル。
    """

    def __init__(self, num_genes, hidden_dim=128, num_layers=3, dropout_rate=0.4):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x_snp, snp_to_gene_map, edge_index, batch_mapping):
        """
        Args:
            x_snp (Tensor): バッチ全体のSNP遺伝子型 (N * L,)
            snp_to_gene_map (Tensor): 各SNPのグローバル遺伝子ノードID (N * L,)
                                      値の範囲 [0, N * num_genes)
            edge_index (Tensor): 単一グラフのエッジ情報 (2, E), 値の範囲 [0, num_genes)
            batch_mapping (Tensor): 各遺伝子ノードの個体インデックス (N * num_genes,)
        """
        num_nodes = batch_mapping.size(0)  # N * num_genes
        N = int(batch_mapping.max().item()) + 1

        # 1. SNP特徴量を遺伝子特徴量に集約 (純PyTorchのscatter操作)
        #    (N * L,) -> (N * num_genes, 1)
        #    Bug fix: dim_size は N * num_genes = batch_mapping.size(0) が正しい。
        #    旧コードの self.num_genes * x_snp.size(0) は num_genes * N * L になり誤り。
        gene_features = torch.zeros(
            num_nodes, 1, device=x_snp.device, dtype=x_snp.dtype
        )
        gene_counts = torch.zeros(num_nodes, 1, device=x_snp.device, dtype=x_snp.dtype)
        idx = snp_to_gene_map.unsqueeze(1)
        gene_features.scatter_add_(0, idx, x_snp.unsqueeze(1))
        gene_counts.scatter_add_(0, idx, torch.ones_like(x_snp.unsqueeze(1)))
        gene_features = gene_features / gene_counts.clamp(min=1.0)

        # 2. バッチ対応のedge_indexを構築
        #    Bug fix: 単一グラフ用のedge_index (ノード0〜num_genes-1) をそのまま使うと
        #    個体1以降のノード(num_genes〜N*num_genes-1)にエッジが届かない。
        #    各個体ごとに i * num_genes のオフセットを加えて N コピーを連結する。
        E = edge_index.size(1)
        offsets = (
            torch.arange(N, device=edge_index.device).repeat_interleave(E)
            * self.num_genes
        )
        batched_edge_index = edge_index.repeat(1, N) + offsets  # (2, N * E)

        # 3. GCNによる特徴伝播
        x = gene_features
        for conv in self.conv_layers:
            x = conv(x, batched_edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # 4. グラフ全体の情報を集約 (N * num_genes, hidden_dim) -> (N, hidden_dim)
        graph_embedding = global_mean_pool(x, batch_mapping)

        # 5. 最終予測 (N, hidden_dim) -> (N, 1)
        return self.output_layer(graph_embedding)
