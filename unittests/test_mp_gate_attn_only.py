"""Tests for attention-gated / MP-ungated hybrid layers (``mp_gate=none``)."""

from __future__ import annotations

import torch

from GNNPlus.layer.gated_hybrid_layer import GatedHybridGraphLayer, _ProjectedMPHead


def test_mp_gate_mode_none_leaves_attention_gated() -> None:
    """Attention heads keep ``gate_mode``; MP heads use ``mp_gate_mode=none``."""
    d_model = 16
    d_h = 8
    layer = GatedHybridGraphLayer(
        d_model=d_model,
        num_attn_heads=1,
        num_gnn_heads=1,
        d_h=d_h,
        gate_mode="elementwise",
        mp_gate_mode="none",
        gnn_types=["GCN"],
    )
    assert layer.gate_mode == "elementwise"
    assert layer.mp_gate_mode == "none"
    # Elementwise attention Q/gate proj is ``d_h + d_h`` (gated).
    assert layer.qg_linears[0].out_features == d_h + d_h
    mp = layer.mp_heads[0]
    assert isinstance(mp, _ProjectedMPHead)
    assert mp.gate_mode == "none"


def test_mp_gate_none_forward_runs() -> None:
    """Forward pass succeeds with attention gates and ungated MP."""
    d_model = 16
    n = 6
    layer = GatedHybridGraphLayer(
        d_model=d_model,
        num_attn_heads=1,
        num_gnn_heads=1,
        d_h=8,
        gate_mode="headwise",
        mp_gate_mode="none",
        gnn_types=["GCN"],
    )
    x = torch.randn(n, d_model)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)
    out = layer(x, edge_index, batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (n, d_model)
    assert torch.isfinite(out).all()


def test_default_mp_gate_matches_gate() -> None:
    """Omitting ``mp_gate_mode`` keeps MP and attention on the same gate style."""
    layer = GatedHybridGraphLayer(
        d_model=16,
        num_attn_heads=1,
        num_gnn_heads=1,
        d_h=8,
        gate_mode="headwise",
        gnn_types=["GCN"],
    )
    assert layer.mp_gate_mode == "headwise"
    mp = layer.mp_heads[0]
    assert isinstance(mp, _ProjectedMPHead)
    assert mp.gate_mode == "headwise"
