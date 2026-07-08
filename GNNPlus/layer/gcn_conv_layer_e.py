"""GCN-e convolution layer (edge-aware GCNConv + BN + activation)."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch_geometric.graphgym import cfg
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor

GateMode = Literal["", "headwise", "elementwise"]


def _parse_gate_mode(raw: object) -> GateMode:
    """Return normalized gate mode for ``GCNConvLayer``."""
    mode = str(raw or "").strip().lower()
    if mode in ("", "none", "false", "0"):
        return ""
    if mode in ("headwise", "elementwise"):
        return mode  # type: ignore[return-value]
    raise ValueError(f"Unknown gnn.gate mode: {raw!r} (expected headwise|elementwise)")


class GCNConvWithEdges(GCNConv):
    """GCNConv that adds edge features into messages before ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            bias,
            add_self_loops=False,
            normalize=False,
        )
        self.edge_dim = edge_dim
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """Incorporate edge features into neighbor messages."""
        return (x_j + edge_attr).relu()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """Run edge-aware GCN propagation."""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            size=None,
        )

        if self.bias is not None:
            out = out + self.bias

        return out


class GCNConvLayer(nn.Module):
    """GCN-e block: conv → BN → act → optional γ gate on output."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dropout: float,
        residual: bool,
        ffn: bool,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = True
        self.ffn = ffn
        self.gate_mode: GateMode = _parse_gate_mode(getattr(cfg.gnn, "gate", ""))
        self.gate_proj: Optional[nn.Linear] = None
        if self.gate_mode == "headwise":
            self.gate_proj = nn.Linear(dim_in, 1)
        elif self.gate_mode == "elementwise":
            self.gate_proj = nn.Linear(dim_in, dim_out)
        self.last_gate_mean: Optional[float] = None

        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(dim_out)
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        edge_dim = dim_in
        if edge_dim is not None:
            self.model = GCNConvWithEdges(dim_in, dim_out, edge_dim, bias=True)
        else:
            self.model = GCNConvWithEdges(dim_in, dim_out, bias=True)

        if self.ffn:
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_in)
            self.ff_linear1 = nn.Linear(dim_in, dim_in * 2)
            self.ff_linear2 = nn.Linear(dim_in * 2, dim_in)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_in)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward sub-block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch: pyg_nn.data.Data) -> pyg_nn.data.Data:
        """Apply GCNE conv, optional output gate γ(x_in), residual, FFN."""
        x_in = batch.x
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        if self.batch_norm:
            batch.x = self.bn_node_x(batch.x)
        batch.x = self.act(batch.x)

        if self.gate_proj is not None:
            gamma = torch.sigmoid(self.gate_proj(x_in))
            batch.x = batch.x * gamma
            self.last_gate_mean = float(gamma.detach().mean().item())
        else:
            self.last_gate_mean = None

        if self.residual:
            batch.x = x_in + batch.x

        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)

            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch
