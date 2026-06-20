"""Hybrid GNN: gated attention + message-passing heads (Heterogeneity_Profile)."""

from __future__ import annotations

from typing import List, Optional, cast

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from GNNPlus.layer.gated_hybrid_layer import (
    AttnMaskType,
    GateMode,
    GatedHybridGraphLayer,
    NormType,
    parse_hybrid_gnn_types,
)


class _PostHybridFFN(nn.Module):
    """Optional FFN after each hybrid block (matches GNNPlus conv layers)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.ff_linear1 = nn.Linear(dim, dim * 2)
        self.ff_linear2 = nn.Linear(dim * 2, dim)
        self.act = register.act_dict[cfg.gnn.act]()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout2(self.ff_linear2(self.dropout1(self.act(self.ff_linear1(self.norm(x))))))
        return x


@register_network('hybrid_gnn')
class HybridGNN(torch.nn.Module):
    """GNNPlus network with stacked :class:`GatedHybridGraphLayer` blocks.

    Set ``model.type: hybrid_gnn`` and tune ``gnn.hybrid.*`` in the yaml.
    Uses the same encoders, heads, and ``train.mode: custom`` pipeline as
    ``custom_gnn`` (MNIST, COCO, OGB, etc.).
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, 'The inner and hidden dims must match.'

        hcfg = cfg.gnn.hybrid
        num_attn = int(hcfg.num_attn_heads)
        num_gnn = int(hcfg.num_gnn_heads)
        if num_attn < 0 or num_gnn < 0 or num_attn + num_gnn < 1:
            raise ValueError(
                'hybrid_gnn requires num_attn_heads + num_gnn_heads >= 1 '
                f'(got {num_attn} + {num_gnn})'
            )

        d_h = int(hcfg.d_h)
        attn_mask = cast(AttnMaskType, str(hcfg.attn_mask))
        gate_mode = cast(GateMode, str(hcfg.gate))
        norm_type = cast(NormType, str(hcfg.norm))
        mp_drop = float(hcfg.mp_dropout) if float(hcfg.mp_dropout) > 0 else float(cfg.gnn.dropout)
        gnn_types: List[str] = parse_hybrid_gnn_types(
            str(hcfg.gnn_types) if hcfg.gnn_types else None,
            num_gnn,
        )

        self.layers = nn.ModuleList([
            GatedHybridGraphLayer(
                d_model=cfg.gnn.dim_inner,
                num_attn_heads=num_attn,
                num_gnn_heads=num_gnn,
                d_h=d_h,
                attn_mask_type=attn_mask,
                gate_mode=gate_mode,
                norm_type=norm_type,
                gnn_types=gnn_types,
                attn_dropout=float(hcfg.attn_dropout),
                mp_gnn_dropout=mp_drop,
                block_bn=bool(hcfg.block_bn),
                block_dropout=mp_drop,
            )
            for _ in range(cfg.gnn.layers_mp)
        ])

        self.ffn_blocks: Optional[nn.ModuleList] = None
        if cfg.gnn.ffn:
            self.ffn_blocks = nn.ModuleList([
                _PostHybridFFN(cfg.gnn.dim_inner, cfg.gnn.dropout)
                for _ in range(cfg.gnn.layers_mp)
            ])

        gnn_head = register.head_dict[cfg.gnn.head]
        self.post_mp = gnn_head(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch: Batch) -> Batch:
        """Run encoder, hybrid blocks, optional FFN, and prediction head."""
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        edge_attr = getattr(batch, 'edge_attr', None)
        x = batch.x
        for i, layer in enumerate(self.layers):
            x = cast(
                torch.Tensor,
                layer(x, batch.edge_index, batch.batch, edge_attr),
            )
            if self.ffn_blocks is not None:
                x = self.ffn_blocks[i](x)
            batch.x = x

        return self.post_mp(batch)
