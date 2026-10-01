"""Unit tests for multi-layer activation aggregation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_MOD_PATH = (
    Path(__file__).resolve().parents[1]
    / "GNNPlus"
    / "experiments"
    / "last_layer_activations.py"
)
_SPEC = importlib.util.spec_from_file_location("last_layer_activations", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LLA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LLA)


def test_per_graph_mean_node_l2_two_graphs() -> None:
    """Mean node L2 should reduce per-graph independently."""
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 1])
    out = _LLA.per_graph_mean_node_l2(x, batch)
    assert out.shape == (2,)
    assert torch.allclose(out[0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out[1], torch.tensor(2.0), atol=1e-5)


def test_per_graph_pooled_l2() -> None:
    """Pooled L2 is L2 of mean-pooled node features."""
    x = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )
    batch = torch.tensor([0, 0, 1])
    out = _LLA.per_graph_pooled_l2(x, batch)
    assert torch.allclose(out[0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out[1], torch.tensor(5.0), atol=1e-5)


def test_write_all_layers_csv_and_plots(tmp_path: Path) -> None:
    """CSV + overlay/heatmap writers succeed on a tiny [L,G] array."""
    idxs = np.arange(4, dtype=np.int64)
    # 3 layers × 4 graphs
    acts = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.4, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    paths = _LLA.dump_all_layer_plots(
        tmp_path,
        dataset_name="toy",
        model_tag="sigma",
        graph_indices=idxs,
        layer_activations=acts,
        labels=labels,
        epoch_tag="last",
    )
    assert Path(paths["csv"]).is_file()
    assert Path(paths["overlay"]).is_file()
    assert Path(paths["heatmap"]).is_file()
    assert Path(paths["last_by_index"]).is_file()
    text = Path(paths["csv"]).read_text()
    assert "layer0_mean_node_l2" in text
    assert "layer2_mean_node_l2" in text
