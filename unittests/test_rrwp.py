"""Unit tests for RRWP precomputation (standalone GRIT)."""

from __future__ import annotations

import importlib.util
import logging
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data


def _load_rrwp_module() -> object:
    """Load ``rrwp.py`` without importing the full ``GNNPlus`` package."""
    path = Path(__file__).resolve().parents[1] / "GNNPlus" / "transform" / "rrwp.py"
    spec = importlib.util.spec_from_file_location("rrwp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain_graph(num_nodes: int) -> Data:
    """Simple undirected chain graph."""
    row = torch.arange(num_nodes - 1)
    col = row + 1
    edge_index = torch.cat(
        [torch.stack([row, col], dim=0), torch.stack([col, row], dim=0)],
        dim=1,
    )
    return Data(edge_index=edge_index, num_nodes=num_nodes, x=torch.randn(num_nodes, 4))


class TestRRWP(unittest.TestCase):
    """RRWP transform tests."""

    def test_rrwp_shapes_pattern_size(self) -> None:
        """RRWP on ~PATTERN-sized graphs produces expected tensor shapes."""
        rrwp = _load_rrwp_module()
        n = 118
        walk_length = 21
        data = _chain_graph(n)
        out = rrwp.add_full_rrwp(data, walk_length=walk_length, add_identity=True)

        self.assertEqual(tuple(out.rrwp.shape), (n, walk_length))
        self.assertEqual(tuple(out.rrwp_index.shape), (2, n * n))
        self.assertEqual(tuple(out.rrwp_val.shape), (n * n, walk_length))
        self.assertEqual(tuple(out.log_deg.shape), (n,))
        self.assertEqual(tuple(out.deg.shape), (n,))

    def test_rrwp_small_graph_no_warning_spam(self) -> None:
        """Small graphs use the dense path without per-graph warnings."""
        rrwp = _load_rrwp_module()
        data = _chain_graph(32)

        with self.assertNoLogs(level=logging.WARNING):
            for _ in range(5):
                rrwp.add_full_rrwp(data, walk_length=8, add_identity=True)


if __name__ == "__main__":
    unittest.main()
