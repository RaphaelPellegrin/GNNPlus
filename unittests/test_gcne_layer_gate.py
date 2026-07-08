"""Tests for optional γ gating on ``GCNConvLayer`` (Level-1 fairness repro)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from GNNPlus.layer.gcn_conv_layer_e import GCNConvLayer


def _ensure_graphgym_cfg(*, gate: str = "headwise") -> None:
    """Minimal GraphGym cfg for ``GCNConvLayer``."""
    yacs = pytest.importorskip("yacs")
    from torch_geometric.graphgym.config import set_cfg

    CfgNode = yacs.config.CfgNode
    node = CfgNode(new_allowed=True)
    node.gnn = CfgNode(new_allowed=True)
    node.gnn.act = "gelu"
    node.gnn.dropout = 0.2
    node.gnn.residual = False
    node.gnn.gate = gate
    set_cfg(node)


def test_gcne_layer_headwise_gate_forward() -> None:
    """Headwise gate scales conv output and records ``last_gate_mean``."""
    _ensure_graphgym_cfg(gate="headwise")
    dim = 16
    num_nodes = 5
    num_edges = 8

    layer = GCNConvLayer(dim, dim, dropout=0.0, residual=False, ffn=False)
    assert layer.gate_proj is not None

    x = torch.randn(num_nodes, dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, dim)
    batch = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    out = layer(batch)
    assert out.x.shape == (num_nodes, dim)
    assert layer.last_gate_mean is not None
    assert 0.0 <= layer.last_gate_mean <= 1.0


def test_gcne_layer_no_gate_forward() -> None:
    """Empty gate mode skips gate projection."""
    _ensure_graphgym_cfg(gate="")
    dim = 8
    layer = GCNConvLayer(dim, dim, dropout=0.0, residual=False, ffn=False)
    assert layer.gate_proj is None

    x = torch.randn(4, dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, dim)
    batch = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    out = layer(batch)
    assert out.x.shape == (4, dim)
    assert layer.last_gate_mean is None
