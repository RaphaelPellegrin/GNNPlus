"""Tests for the SiGMA-lite layer and network (one gate on the mixed MMA update)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from GNNPlus.layer.sigma_lite_layer import SigmaLiteLayer


def _toy_graph(num_nodes: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(edge_index, batch, edge_attr)`` for two small path graphs."""
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])[:num_nodes]
    return edge_index, batch, torch.randn(edge_index.size(1), 8)


@pytest.mark.parametrize("num_attn,num_gnn", [(2, 2), (1, 0), (0, 2)])
@pytest.mark.parametrize("gate_act", ["sigmoid", "gelu"])
def test_sigma_lite_forward_shapes(num_attn: int, num_gnn: int, gate_act: str) -> None:
    """Any head mix (either side may be empty) keeps ``[N, d_model]``."""
    torch.manual_seed(0)
    layer = SigmaLiteLayer(
        d_model=8,
        num_attn_heads=num_attn,
        num_gnn_heads=num_gnn,
        d_h=4,
        gnn_types=["GCN", "GIN"][:num_gnn],
        gate_act=gate_act,
    )
    edge_index, batch, edge_attr = _toy_graph()
    x = torch.randn(6, 8, requires_grad=True)
    out = layer(x, edge_index, batch, edge_attr)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (6, 8)
    out.sum().backward()
    assert x.grad is not None
    assert layer.gate_proj.weight.grad is not None


def test_sigma_lite_heads_are_ungated() -> None:
    """The inner MMA block has no per-head gate parameters."""
    layer = SigmaLiteLayer(d_model=8, num_attn_heads=1, num_gnn_heads=1, d_h=4, gnn_types=["GCN"])
    assert layer.mma.gate_mode == "none"
    assert layer.mma.mp_gate_mode == "none"
    assert layer.mma.qg_linears[0].out_features == 4


def test_sigma_lite_open_gate_matches_mma() -> None:
    """With gate = 1 and ``W_1 = I`` the block reduces to MMA with residual."""
    torch.manual_seed(0)
    layer = SigmaLiteLayer(d_model=8, num_attn_heads=1, num_gnn_heads=1, d_h=4, gnn_types=["GCN"])
    layer.eval()
    with torch.no_grad():
        layer.gated_proj.weight.copy_(torch.eye(8))
        layer.gated_proj.bias.zero_()
    edge_index, batch, edge_attr = _toy_graph()
    x = torch.randn(6, 8)
    out = layer(x, edge_index, batch, edge_attr, gate_override="ones")
    mixed = layer.mma(x, edge_index, batch, edge_attr)
    assert isinstance(out, torch.Tensor) and isinstance(mixed, torch.Tensor)
    assert torch.allclose(out, x + mixed, atol=1e-6)


def test_sigma_lite_gate_stats() -> None:
    """Gate stats expose one sigmoid gate in ``(0, 1)`` per node and feature."""
    layer = SigmaLiteLayer(d_model=8, num_attn_heads=1, num_gnn_heads=1, d_h=4, gnn_types=["GIN"])
    edge_index, batch, edge_attr = _toy_graph()
    out = layer(torch.randn(6, 8), edge_index, batch, edge_attr, return_gate_stats=True)
    assert isinstance(out, tuple)
    _, aux = out
    assert 0.0 < aux["gate_stats"]["lite_gate_mean"] < 1.0
    assert aux["gate_values"]["lite"][0].shape == (6, 8)


def test_sigma_lite_rejects_unknown_gate_act() -> None:
    """Only ``sigmoid`` and ``gelu`` are accepted."""
    with pytest.raises(ValueError, match="gate_act"):
        SigmaLiteLayer(d_model=8, num_attn_heads=1, num_gnn_heads=0, d_h=4, gate_act="relu")


def test_sigma_lite_network_builds_sigma_lite_layers() -> None:
    """``model.type: sigma_lite`` builds a HybridGNN whose blocks are SiGMA-lite."""
    import torch_geometric.graphgym.models.head  # noqa: F401  (registers the graph head)
    import torch_geometric.graphgym.register as register
    from torch_geometric.graphgym.config import cfg

    import GNNPlus  # noqa: F401  (registers networks, encoders, configs)

    cfg.dataset.node_encoder = False
    cfg.dataset.edge_encoder = False
    cfg.gnn.layers_pre_mp = 0
    cfg.gnn.layers_mp = 2
    cfg.gnn.dim_inner = 8
    cfg.gnn.ffn = False
    cfg.gnn.head = "graph"
    cfg.gnn.hybrid.num_attn_heads = 1
    cfg.gnn.hybrid.num_gnn_heads = 1
    cfg.gnn.hybrid.d_h = 4
    cfg.gnn.hybrid.gnn_types = "GCN"
    cfg.gnn.sigma_lite.gate_act = "gelu"

    model = register.network_dict["sigma_lite"](dim_in=8, dim_out=2)
    assert all(isinstance(layer, SigmaLiteLayer) for layer in model.layers)
    assert all(isinstance(layer, nn.Module) for layer in model.layers)
    assert model.layers[0].gate_act == "gelu"
