"""Tests for GatedGCN+ hybrid MP head (edge-aware GatedGCNLayer)."""

from __future__ import annotations

import pytest
import torch

from GNNPlus.layer.gated_hybrid_layer import (
    _GatedGCNHybridMPHead,
    _ProjectedMPHead,
)


def _ensure_graphgym_cfg() -> None:
    """Minimal GraphGym cfg for GatedGCNLayer FFN activation lookup."""
    yacs = pytest.importorskip("yacs")
    from torch_geometric.graphgym.config import set_cfg

    CfgNode = yacs.config.CfgNode
    node = CfgNode(new_allowed=True)
    node.gnn = CfgNode(new_allowed=True)
    node.gnn.act = "relu"
    node.gnn.dropout = 0.15
    node.gnn.residual = True
    set_cfg(node)


def test_gatedgcn_hybrid_mp_head_forward() -> None:
    """GatedGCN head runs forward with edge features projected to d_h."""
    _ensure_graphgym_cfg()
    d_model = 35
    d_h = 16
    num_nodes = 6
    num_edges = 10

    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, d_model)

    standalone = _GatedGCNHybridMPHead(
        d_h,
        edge_dim=d_model,
        dropout=0.15,
        residual=True,
        ffn=True,
        act="relu",
    )
    raw = standalone(x[:, :d_h], edge_index, edge_attr)
    assert raw.shape == (num_nodes, d_h)
    assert torch.isfinite(raw).all()

    mp = _ProjectedMPHead("GATEDGCN", d_model, d_h, gate_mode="headwise")
    out, gate = mp(x, edge_index, edge_attr)
    assert out.shape == (num_nodes, d_h)
    assert gate.shape == (num_nodes, 1)
    assert torch.isfinite(out).all()


def test_projected_mp_head_gatedgcn_kind() -> None:
    """GATEDGCN maps to GatedGCNLayer-based hybrid head."""
    _ensure_graphgym_cfg()
    head = _ProjectedMPHead("GATEDGCN", d_model=35, d_h=8, gate_mode="headwise")
    assert head.kind == "GATEDGCN"
