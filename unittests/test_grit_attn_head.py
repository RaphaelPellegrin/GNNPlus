"""Tests for GRIT as a SiGMA hybrid attention-head type."""

from __future__ import annotations

import torch
from yacs.config import CfgNode as CN

from GNNPlus.layer.gated_hybrid_layer import GatedHybridGraphLayer
from GNNPlus.layer.grit_attn_head import _GRITAttnHead


def _minimal_cfg() -> CN:
    """Build a minimal GraphGym-like cfg for MP heads used in hybrid layers."""
    cfg = CN(new_allowed=True)
    cfg.gnn = CN(new_allowed=True)
    cfg.gnn.act = "relu"
    cfg.gnn.dropout = 0.0
    cfg.gnn.residual = True
    cfg.gnn.ffn = False
    cfg.gnn.use_hermitian = False
    cfg.gnn.unitary_taylor_order = 8
    return cfg


def test_grit_attn_head_forward() -> None:
    """GRIT attn head returns ``[N, d_h]`` gated outputs."""
    d_model = 16
    d_h = 8
    num_nodes = 5
    num_edges = 10
    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, d_model)

    head = _GRITAttnHead(
        d_model=d_model,
        d_h=d_h,
        gate_mode="elementwise",
        edge_dim=d_model,
        attn_dropout=0.0,
    )
    out, gamma = head(x, edge_index, edge_attr)
    assert out.shape == (num_nodes, d_h)
    assert gamma.shape == (num_nodes, d_h)
    assert torch.isfinite(out).all()
    assert torch.isfinite(gamma).all()


def test_gated_hybrid_layer_grit_attn(monkeypatch: object) -> None:
    """Hybrid layer with ``attn_type=grit`` and one MP head runs forward."""
    import GNNPlus.layer.gated_hybrid_layer as hybrid_mod

    cfg = _minimal_cfg()
    monkeypatch.setattr(hybrid_mod, "cfg", cfg)

    d_model = 16
    d_h = 8
    num_nodes = 6
    num_edges_mp = 8
    num_edges_attn = 20
    x = torch.randn(num_nodes, d_model)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    edge_index_mp = torch.randint(0, num_nodes, (2, num_edges_mp))
    edge_attr_mp = torch.randn(num_edges_mp, d_model)
    edge_index_attn = torch.randint(0, num_nodes, (2, num_edges_attn))
    edge_attr_attn = torch.randn(num_edges_attn, d_model)

    layer = GatedHybridGraphLayer(
        d_model=d_model,
        num_attn_heads=1,
        num_gnn_heads=1,
        d_h=d_h,
        attn_mask_type="full",
        gate_mode="headwise",
        norm_type="layernorm",
        gnn_types=["GCN"],
        attn_dropout=0.0,
        mp_gnn_dropout=0.0,
        residual=True,
        attn_type="grit",
        edge_dim=d_model,
        grit_clamp=5.0,
        grit_edge_enhance=True,
        grit_act="relu",
        grit_use_bias=False,
    )
    out = layer(
        x,
        edge_index_mp,
        batch,
        edge_attr_mp,
        edge_index_attn=edge_index_attn,
        edge_attr_attn=edge_attr_attn,
        edge_index_mp=edge_index_mp,
        edge_attr_mp=edge_attr_mp,
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (num_nodes, d_model)
    assert torch.isfinite(out).all()
