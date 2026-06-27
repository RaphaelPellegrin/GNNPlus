"""Tests for GRIT hybrid MP head integration."""

from __future__ import annotations

import torch
from yacs.config import CfgNode as CN

from GNNPlus.layer.gated_hybrid_layer import _ProjectedMPHead
from GNNPlus.layer.grit_hybrid_mp_head import _GRITHybridMPHead
from GNNPlus.layer.grit_layer import GritTransformerLayer, resolve_grit_num_heads


def _minimal_cfg() -> CN:
    cfg = CN(new_allowed=True)
    cfg.gnn = CN(new_allowed=True)
    cfg.gnn.act = "relu"
    cfg.gnn.dropout = 0.0
    cfg.gnn.residual = True
    cfg.gnn.ffn = False
    cfg.gnn.grit = CN(new_allowed=True)
    cfg.gnn.grit.n_heads = 8
    cfg.gnn.grit.dropout = 0.0
    cfg.gnn.grit.attn_dropout = 0.0
    cfg.gnn.grit.layer_norm = False
    cfg.gnn.grit.batch_norm = False
    cfg.gnn.grit.residual = True
    cfg.gnn.grit.norm_e = True
    cfg.gnn.grit.update_e = False
    cfg.gnn.grit.act = "relu"
    return cfg


def test_resolve_grit_num_heads() -> None:
    """Head count must divide hidden width."""
    assert resolve_grit_num_heads(64, 8) == 8
    assert resolve_grit_num_heads(64, 10) == 8
    assert resolve_grit_num_heads(16, 8) == 8


def test_grit_transformer_layer_forward() -> None:
    """Single GRIT layer runs on a tiny graph."""
    d_h = 16
    num_nodes = 4
    num_edges = 6
    x = torch.randn(num_nodes, d_h)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr = torch.randn(num_edges, d_h)

    class _Batch:
        def __init__(self) -> None:
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.num_nodes = num_nodes

        def get(self, key: str, default: object = None) -> object:
            return getattr(self, key, default)

    layer = GritTransformerLayer(
        in_dim=d_h,
        out_dim=d_h,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        batch_norm=False,
        layer_norm=False,
    )
    out = layer(_Batch()).x
    assert out.shape == (num_nodes, d_h)
    assert torch.isfinite(out).all()


def test_grit_hybrid_mp_head_forward(monkeypatch: object) -> None:
    """GRIT hybrid MP head integrates with projected gating."""
    import torch_geometric.graphgym.config as graphgym_cfg

    cfg = _minimal_cfg()
    monkeypatch.setattr(graphgym_cfg, "cfg", cfg)

    d_model = 32
    d_h = 16
    num_nodes = 5
    num_edges = 8
    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, d_model)

    head = _GRITHybridMPHead(d_h, edge_dim=d_model, gnn_dropout=0.0)
    out = head(x[:, :d_h], edge_index, edge_attr)
    assert out.shape == (num_nodes, d_h)
    assert torch.isfinite(out).all()


def test_projected_mp_head_grit_kind(monkeypatch: object) -> None:
    """GRIT string routes to _GRITHybridMPHead."""
    import torch_geometric.graphgym.config as graphgym_cfg

    cfg = _minimal_cfg()
    monkeypatch.setattr(graphgym_cfg, "cfg", cfg)

    head = _ProjectedMPHead("GRIT", d_model=16, d_h=16, gate_mode="headwise")
    assert head.kind == "GRIT"
    assert isinstance(head.conv, _GRITHybridMPHead)
