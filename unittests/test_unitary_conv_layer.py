"""Tests for unitary (UniGCN) convolution layers."""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg

from GNNPlus.layer.gated_hybrid_layer import (
    _ProjectedMPHead,
    _UnitaryGCNHybridMPHead,
)
from GNNPlus.layer.unitary_conv_layer import (
    UnitaryGCNConvLayer,
    build_unitary_taylor_conv,
)


def _ensure_unitary_cfg() -> None:
    """Minimal GraphGym cfg for unitary layers."""
    cfg.gnn.act = "relu"
    cfg.gnn.dropout = 0.0
    cfg.gnn.residual = False
    cfg.gnn.use_hermitian = False
    cfg.gnn.unitary_taylor_order = 4
    cfg.gnn.unitary_return_real = True


def test_build_unitary_taylor_conv_forward() -> None:
    """Taylor unitary GCN produces finite real outputs on a small graph."""
    d_h = 8
    num_nodes = 6
    num_edges = 10
    x = torch.randn(num_nodes, d_h)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    conv = build_unitary_taylor_conv(
        d_h,
        d_h,
        taylor_order=4,
        return_real=True,
        conv_bias=False,
    )
    out = conv(x, edge_index)
    assert out.shape == (num_nodes, d_h)
    assert not torch.is_complex(out)
    assert torch.isfinite(out).all()


def test_unitary_gcn_conv_layer_batch_forward() -> None:
    """GraphGym ``UnitaryGCNConvLayer`` runs on a ``Data`` batch."""
    _ensure_unitary_cfg()
    dim = 8
    num_nodes = 5
    num_edges = 8

    layer = UnitaryGCNConvLayer(dim, dim, dropout=0.0, residual=True, ffn=False)
    x = torch.randn(num_nodes, dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = Data(x=x, edge_index=edge_index)

    out = layer(batch)
    assert out.x.shape == (num_nodes, dim)
    assert torch.isfinite(out.x).all()


def test_unitary_hybrid_mp_head_forward() -> None:
    """Hybrid UniGCN MP head runs at hidden width ``d_h``."""
    d_h = 16
    num_nodes = 5
    num_edges = 8

    x = torch.randn(num_nodes, d_h)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    head = _UnitaryGCNHybridMPHead(d_h, gnn_dropout=0.0, taylor_order=4)
    out = head(x, edge_index)
    assert out.shape == (num_nodes, d_h)
    assert torch.isfinite(out).all()


def test_projected_mp_head_unigcn_kind() -> None:
    """UNIGCN kind routes to ``_UnitaryGCNHybridMPHead``."""
    _ensure_unitary_cfg()
    head = _ProjectedMPHead("UNIGCN", d_model=16, d_h=8, gate_mode="headwise")
    assert head.kind == "UNIGCN"
    assert isinstance(head.conv, _UnitaryGCNHybridMPHead)
