"""Tests for virtual nodes and readout MLP presets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data

_REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str) -> ModuleType:
    """Import a module without pulling in the full GNNPlus training stack."""
    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_graph_aug = _load_module(
    'gnnplus_graph_aug_test',
    'GNNPlus/preprocessing/graph_augmentations.py',
)


def test_add_virtual_nodes_extends_graph() -> None:
    """Virtual nodes increase node count and connect to all real nodes."""
    n, d, r = 5, 8, 2
    x = torch.randn(n, d)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1), 3)
    pestat = torch.randn(n, 20)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pestat_RWSE=pestat)

    out = _graph_aug.add_virtual_nodes(graph, r)

    assert out.x.shape == (n + r, d)
    assert out.pestat_RWSE.shape == (n + r, 20)
    assert torch.allclose(out.x[n:], torch.zeros(r, d))
    assert torch.allclose(out.pestat_RWSE[n:], torch.zeros(r, 20))
    assert out.edge_index.size(1) == edge_index.size(1) + 2 * r * n


def test_add_virtual_nodes_zero_is_noop() -> None:
    """``r=0`` returns a clone with unchanged shape."""
    graph = Data(x=torch.ones(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]]))
    out = _graph_aug.add_virtual_nodes(graph, 0)
    assert out.x.shape == graph.x.shape
    assert out is not graph


def _mock_cfg(act: str = 'relu', dropout: float = 0.0) -> MagicMock:
    cfg = MagicMock()
    cfg.gnn.act = act
    cfg.gnn.dropout = dropout
    return cfg


@pytest.mark.parametrize(
    ('preset', 'dim_in', 'dim_out'),
    [
        ('linear', 96, 11),
        ('narrow2', 96, 11),
        ('pyramid', 96, 11),
        ('deep4', 96, 11),
    ],
)
def test_build_readout_mlp_output_shape(
    preset: str,
    dim_in: int,
    dim_out: int,
) -> None:
    """Readout presets produce the expected output dimension."""
    register_mod = MagicMock()
    register_mod.act_dict = {'relu': torch.nn.ReLU}
    cfg_mod = MagicMock()
    cfg_mod.gnn.act = 'relu'
    cfg_mod.gnn.dropout = 0.0

    readout_path = _REPO / 'GNNPlus/head/readout_mlp.py'
    spec = importlib.util.spec_from_file_location('readout_mlp_test', readout_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules['readout_mlp_test'] = mod
    mod.register = register_mod
    mod.cfg = cfg_mod
    spec.loader.exec_module(mod)

    mlp = mod.build_readout_mlp(dim_in, dim_out, preset)
    y = mlp(torch.randn(4, dim_in))
    assert y.shape == (4, dim_out)


def test_build_readout_mlp_unknown_preset() -> None:
    """Unknown presets raise a clear error."""
    register_mod = MagicMock()
    register_mod.act_dict = {'relu': torch.nn.ReLU}
    cfg_mod = MagicMock()
    cfg_mod.gnn.act = 'relu'
    cfg_mod.gnn.dropout = 0.0

    readout_path = _REPO / 'GNNPlus/head/readout_mlp.py'
    spec = importlib.util.spec_from_file_location('readout_mlp_test2', readout_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules['readout_mlp_test2'] = mod
    mod.register = register_mod
    mod.cfg = cfg_mod
    spec.loader.exec_module(mod)

    with pytest.raises(ValueError, match='gnn.readout_mlp'):
        mod.build_readout_mlp(16, 1, 'invalid')
