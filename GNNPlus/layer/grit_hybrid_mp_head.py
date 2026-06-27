"""GRIT layer as a gated hybrid message-passing head."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.graphgym.config import cfg

from GNNPlus.layer.grit_layer import (
    GritTransformerLayer,
    build_grit_layer_cfg,
    resolve_grit_num_heads,
)


class _GritHybridBatch:
    """Minimal batch object for :class:`GritTransformerLayer` inside hybrid blocks."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Optional[Tensor]
    num_nodes: int
    log_deg: Optional[Tensor]
    deg: Optional[Tensor]

    def __init__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        *,
        log_deg: Optional[Tensor] = None,
        deg: Optional[Tensor] = None,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = int(x.size(0))
        self.log_deg = log_deg
        self.deg = deg

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like accessor used by GRIT attention."""
        return getattr(self, key, default)


class _GRITHybridMPHead(nn.Module):
    """One GRIT layer at hidden width ``d_h`` (sparse graph edges)."""

    def __init__(
        self,
        d_h: int,
        *,
        edge_dim: int,
        gnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.edge_proj = nn.Linear(edge_dim, d_h)
        self._gnn_dropout = float(gnn_dropout)

        grit_cfg = getattr(cfg.gnn, "grit", None)
        n_heads = int(getattr(grit_cfg, "n_heads", 8)) if grit_cfg is not None else 8
        n_heads = resolve_grit_num_heads(d_h, n_heads)

        layer_cfg = build_grit_layer_cfg(
            dropout=float(getattr(grit_cfg, "dropout", 0.0)) if grit_cfg else 0.0,
            attn_dropout=float(getattr(grit_cfg, "attn_dropout", 0.2)) if grit_cfg else 0.2,
            layer_norm=bool(getattr(grit_cfg, "layer_norm", False)) if grit_cfg else False,
            batch_norm=bool(getattr(grit_cfg, "batch_norm", True)) if grit_cfg else True,
            residual=bool(getattr(grit_cfg, "residual", True)) if grit_cfg else True,
            norm_e=bool(getattr(grit_cfg, "norm_e", True)) if grit_cfg else True,
            update_e=bool(getattr(grit_cfg, "update_e", False)) if grit_cfg else False,
        )

        self.layer = GritTransformerLayer(
            in_dim=d_h,
            out_dim=d_h,
            num_heads=n_heads,
            dropout=float(getattr(grit_cfg, "dropout", 0.0)) if grit_cfg else 0.0,
            attn_dropout=float(getattr(grit_cfg, "attn_dropout", 0.2)) if grit_cfg else 0.2,
            layer_norm=bool(getattr(grit_cfg, "layer_norm", False)) if grit_cfg else False,
            batch_norm=bool(getattr(grit_cfg, "batch_norm", True)) if grit_cfg else True,
            residual=bool(getattr(grit_cfg, "residual", True)) if grit_cfg else True,
            act=str(getattr(grit_cfg, "act", cfg.gnn.act)) if grit_cfg else str(cfg.gnn.act),
            norm_e=bool(getattr(grit_cfg, "norm_e", True)) if grit_cfg else True,
            update_e=bool(getattr(grit_cfg, "update_e", False)) if grit_cfg else False,
            cfg=layer_cfg,
        )

    def forward(
        self,
        x_h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one GRIT update on hidden node features."""
        num_e = edge_index.size(1)
        dev, dt = x_h.device, x_h.dtype
        if edge_attr is None:
            eh: Optional[Tensor] = torch.zeros((num_e, self.d_h), device=dev, dtype=dt)
        else:
            feat = edge_attr.float()
            if feat.size(-1) == self.d_h:
                eh = feat.to(dtype=dt)
            else:
                eh = self.edge_proj(feat).to(dtype=dt)

        batch = _GritHybridBatch(x_h, edge_index, eh)
        out = self.layer(batch).x
        return F.dropout(out, p=self._gnn_dropout, training=self.training)
