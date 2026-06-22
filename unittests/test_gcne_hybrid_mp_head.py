"""Tests for edge-aware GCNE hybrid MP head."""

from __future__ import annotations

import torch

from GNNPlus.layer.gated_hybrid_layer import _GCNEHybridMPHead, _ProjectedMPHead


def test_gcne_hybrid_mp_head_forward() -> None:
    """GCNE head runs forward with edge features projected to d_h."""
    d_model = 32
    d_h = 16
    num_nodes = 5
    num_edges = 8
    edge_dim = 32

    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    standalone = _GCNEHybridMPHead(d_h, edge_dim=edge_dim, gnn_dropout=0.0)
    raw = standalone(x[:, :d_h], edge_index, edge_attr)
    assert raw.shape == (num_nodes, d_h)
    assert torch.isfinite(raw).all()

    mp = _ProjectedMPHead("GCNE", d_model, d_h, gate_mode="headwise")
    out, gate = mp(x, edge_index, edge_attr)

    assert out.shape == (num_nodes, d_h)
    assert gate.shape == (num_nodes, 1)
    assert torch.isfinite(out).all()
    assert torch.isfinite(gate).all()


def test_projected_mp_head_accepts_gcne_kind() -> None:
    """GCNE is a valid projected MP head kind."""
    head = _ProjectedMPHead("GCNE", d_model=16, d_h=8, gate_mode="headwise")
    assert head.kind == "GCNE"
