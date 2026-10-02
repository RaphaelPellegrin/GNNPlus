"""Tests for virtual nodes and weighted CE ignore handling on node tasks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

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


def test_add_virtual_nodes_pads_per_node_labels() -> None:
    """Per-node ``y`` is extended with ignore labels for virtual nodes."""
    n, d, r = 5, 8, 2
    y = torch.arange(n)
    graph = Data(
        x=torch.randn(n, d),
        y=y,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
    )

    out = _graph_aug.add_virtual_nodes(graph, r)

    assert out.y.shape == (n + r,)
    assert torch.equal(out.y[:n], y)
    assert torch.all(out.y[n:] == _graph_aug.VIRTUAL_NODE_LABEL_IGNORE)


def test_add_virtual_nodes_zero_is_noop() -> None:
    """``r=0`` returns a clone with unchanged shape."""
    graph = Data(x=torch.ones(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]]))
    out = _graph_aug.add_virtual_nodes(graph, 0)
    assert out.x.shape == graph.x.shape
    assert out is not graph


def test_weighted_cross_entropy_keeps_pred_length_with_vn_ignore() -> None:
    """Loss ignores VN labels but returns full-length scores for the logger."""
    mock_cfg = MagicMock()
    mock_cfg.model.loss_fun = 'weighted_cross_entropy'
    register_mod = MagicMock()
    register_mod.register_loss = lambda _name: (lambda fn: fn)

    # Load loss module without importing the full GNNPlus package / torch_scatter.
    gg_cfg = MagicMock(cfg=mock_cfg)
    sys.modules['torch_geometric.graphgym.config'] = gg_cfg
    sys.modules['torch_geometric.graphgym.register'] = register_mod
    sys.modules['GNNPlus.preprocessing.graph_augmentations'] = _graph_aug
    wce = _load_module(
        'gnnplus_wce_vn_test',
        'GNNPlus/loss/weighted_cross_entropy.py',
    )

    n_real, n_vn, n_classes = 4, 2, 3
    pred = torch.randn(n_real + n_vn, n_classes, requires_grad=True)
    true = torch.tensor(
        [0, 1, 2, 1] + [_graph_aug.VIRTUAL_NODE_LABEL_IGNORE] * n_vn,
        dtype=torch.long,
    )
    loss, pred_score = wce.weighted_cross_entropy(pred, true)
    assert pred_score.shape == pred.shape
    assert true.shape[0] == pred_score.shape[0]
    loss.backward()
    assert pred.grad is not None

