"""Loss functions shared by the training entry points.

Importing this module has no side effects: it starts no run, reads no
data, and initializes no external logging. Previously ``train_gnn.py``
imported ``CorrelationLoss`` from the executable script ``main.py``,
which pulled a whole entry point (and its imports) in just to reuse one
``nn.Module``.
"""

from __future__ import annotations

import torch
from torch import nn


class CorrelationLoss(nn.Module):
    """損失としてピアソン相関係数の負の値を計算する"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_c = y_pred - torch.mean(y_pred)
        y_true_c = y_true - torch.mean(y_true)
        pearson_num = torch.sum(y_pred_c * y_true_c)
        # sqrt(0) の勾配は inf になるため epsilon を内側に入れて数値安定化
        pearson_den = torch.sqrt(torch.sum(y_pred_c**2) + 1e-8) * torch.sqrt(
            torch.sum(y_true_c**2) + 1e-8
        )
        return -pearson_num / (pearson_den + 1e-8)
