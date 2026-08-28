"""Label-faithful sum / norm-sum convs for GCN/GIN routing synthetic (Track A)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree


class RoutingSumConv(nn.Module):
    """Aggregate neighbor signal channel with an unnormalized sum (GIN rule)."""

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.d_h = int(d_h)
        self.out_proj = nn.Identity() if self.d_h == 1 else nn.Linear(1, self.d_h)

    def forward(self, x_signal: Tensor, edge_index: Tensor) -> Tensor:
        """Propagate scalar signals along ``edge_index``."""
        if x_signal.dim() == 1:
            x_signal = x_signal.unsqueeze(-1)
        num_nodes = int(x_signal.size(0))
        row, col = edge_index
        out = torch.zeros(
            num_nodes,
            1,
            device=x_signal.device,
            dtype=x_signal.dtype,
        )
        out.index_add_(0, row, x_signal[col])
        projected = self.out_proj(out)
        return projected


class RoutingNormSumConv(nn.Module):
    """Aggregate neighbor signals with GCN-style degree normalization."""

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.d_h = int(d_h)
        self.out_proj = nn.Identity() if self.d_h == 1 else nn.Linear(1, self.d_h)

    def forward(self, x_signal: Tensor, edge_index: Tensor) -> Tensor:
        """Propagate normalized scalar signals along ``edge_index``."""
        if x_signal.dim() == 1:
            x_signal = x_signal.unsqueeze(-1)
        num_nodes = int(x_signal.size(0))
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=x_signal.dtype).clamp(min=1.0)
        norm = (deg[row] + 1.0).rsqrt() * (deg[col] + 1.0).rsqrt()
        msg = x_signal[col] * norm.view(-1, 1)
        out = torch.zeros(
            num_nodes,
            1,
            device=x_signal.device,
            dtype=x_signal.dtype,
        )
        out.index_add_(0, row, msg)
        projected = self.out_proj(out)
        return projected
