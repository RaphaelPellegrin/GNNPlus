"""Standalone GRIT network (Ma et al., ICML 2023)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_geometric.graphgym.models.head  # noqa: F401 — register heads
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import register_network

from GNNPlus.layer.grit_layer import GritTransformerLayerRegistered


class GritFeatureEncoder(nn.Module):
    """Node/edge encoders plus optional RRWP projections."""

    def __init__(self, dim_in: int) -> None:
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            node_encoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = node_encoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            self.dim_in = cfg.gnn.dim_inner

        if cfg.dataset.edge_encoder:
            if int(cfg.gnn.dim_edge) <= 0:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            edge_encoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = edge_encoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )

        self.rrwp_abs_encoder: nn.Module | None = None
        self.rrwp_rel_encoder: nn.Module | None = None
        if cfg.posenc_RRWP.enable:
            ksteps = int(cfg.posenc_RRWP.ksteps)
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"](
                ksteps,
                cfg.gnn.dim_inner,
            )
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"](
                ksteps,
                cfg.gnn.dim_edge,
                pad_to_full_graph=bool(cfg.gt.attn.full_attn),
                add_node_attr_as_self_loop=False,
                fill_value=0.0,
            )

    def forward(self, batch: object) -> object:
        """Run registered encoders."""
        for name, module in self.named_children():
            if name.startswith("rrwp") and module is None:
                continue
            batch = module(batch)
        return batch


@register_network("GritTransformer")
class GritTransformer(nn.Module):
    """Graph Inductive Bias Transformer without message passing."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.encoder = GritFeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, (
            "gt.dim_hidden and gnn.dim_inner must match"
        )

        layer_type = str(cfg.gt.layer_type)
        transformer_layer = register.layer_dict.get(layer_type, GritTransformerLayerRegistered)
        self.layers = nn.ModuleList(
            [
                transformer_layer(
                    in_dim=cfg.gt.dim_hidden,
                    out_dim=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=cfg.gt.residual,
                    act=cfg.gnn.act,
                    norm_e=cfg.gt.attn.norm_e,
                    O_e=cfg.gt.attn.O_e,
                    cfg=cfg.gt,
                )
                for _ in range(cfg.gt.layers)
            ]
        )

        gnn_head = register.head_dict[cfg.gnn.head]
        self.post_mp = gnn_head(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch: object) -> object:
        """Forward through encoder, GRIT layers, and task head."""
        for module in self.children():
            batch = module(batch)
        return batch
