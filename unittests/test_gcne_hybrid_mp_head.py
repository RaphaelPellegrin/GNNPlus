"""Tests for edge-aware GCNE hybrid MP heads."""

from __future__ import annotations

import torch

from GNNPlus.layer.gated_hybrid_layer import (
    _GCNEConvLayerHybridMPHead,
    _GCNEHybridMPHead,
    _ProjectedMPHead,
)


def test_gcne_conv_layer_hybrid_mp_head_forward() -> None:
    """Full GCNConvLayer GCNE head runs forward with edge features at d_h."""
    d_model = 32
    d_h = 16
    num_nodes = 5
    num_edges = 8
    edge_dim = 32

    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    head = _GCNEConvLayerHybridMPHead(
        d_h, edge_dim=edge_dim, dropout=0.0, gnn_dropout=0.0
    )
    out = head(x[:, :d_h], edge_index, edge_attr)
    assert out.shape == (num_nodes, d_h)
    assert torch.isfinite(out).all()


def test_gcne_legacy_conv_hybrid_mp_head_forward() -> None:
    """Legacy GCNE_CONV head (raw GCNConvWithEdges) still runs forward."""
    d_h = 16
    num_nodes = 5
    num_edges = 8
    edge_dim = 32

    x = torch.randn(num_nodes, d_h)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    standalone = _GCNEHybridMPHead(d_h, edge_dim=edge_dim, gnn_dropout=0.0)
    raw = standalone(x, edge_index, edge_attr)
    assert raw.shape == (num_nodes, d_h)
    assert torch.isfinite(raw).all()


def test_projected_mp_head_gcne_uses_full_layer() -> None:
    """GCNE kind routes to _GCNEConvLayerHybridMPHead, not raw conv."""
    head = _ProjectedMPHead("GCNE", d_model=16, d_h=8, gate_mode="headwise")
    assert head.kind == "GCNE"
    assert isinstance(head.conv, _GCNEConvLayerHybridMPHead)


def test_projected_mp_head_gcne_conv_uses_legacy() -> None:
    """GCNE_CONV kind keeps legacy raw-conv path."""
    head = _ProjectedMPHead("GCNE_CONV", d_model=16, d_h=8, gate_mode="headwise")
    assert head.kind == "GCNE_CONV"
    assert isinstance(head.conv, _GCNEHybridMPHead)
