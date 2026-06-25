"""Tests for heterogeneity profile utilities."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from yacs.config import CfgNode as CN

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    """Import a module file without loading GNNPlus/__init__.py (avoids torch_scatter)."""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_metrics = _load_module(
    "gnnplus_heterogeneity_metrics",
    "GNNPlus/experiments/heterogeneity_metrics.py",
)
_track = _load_module(
    "gnnplus_track_avg_accuracy",
    "GNNPlus/experiments/track_avg_accuracy.py",
)

per_graph_performance_value = _metrics.per_graph_performance_value
summarize_trial_metrics = _metrics.summarize_trial_metrics
compute_average_per_graph = _track.compute_average_per_graph
get_gnnplus_model_slug = _track.get_gnnplus_model_slug
infer_plot_task_type = _track.infer_plot_task_type
load_and_plot_average_per_graph = _track.load_and_plot_average_per_graph


def test_compute_average_per_graph() -> None:
    """Mean per-graph values are computed correctly."""
    graph_dict = {0: [1.0, 1.0, 0.0], 1: [0.0, 0.0]}
    indices, values = compute_average_per_graph(graph_dict)
    assert list(indices) == [0, 1]
    assert np.isclose(values[0], 2.0 / 3.0)
    assert np.isclose(values[1], 0.0)


def test_per_graph_classification_value() -> None:
    """Single-label classification returns 0 or 1."""
    pred = torch.tensor([[0.1, 0.9]])
    true = torch.tensor([1])
    assert per_graph_performance_value(pred, true, "classification") == 1.0


def test_summarize_trial_metrics() -> None:
    """Trial metric summary includes mean and std."""
    summary = summarize_trial_metrics([0.8, 0.9, 0.7], "classification")
    assert summary["count"] == 3.0
    assert np.isclose(summary["test_accuracy_mean"], 0.8)
    assert summary["test_accuracy_std"] >= 0.0


def test_get_gnnplus_model_slug_hybrid() -> None:
    """Hybrid slug encodes head counts and MP types."""
    local = CN()
    local.model = CN({"type": "hybrid_gnn"})
    local.gnn = CN(
        {
            "layer_type": "gcne",
            "hybrid": CN(
                {
                    "num_attn_heads": 2,
                    "num_gnn_heads": 2,
                    "gnn_types": "GCNE,GINE",
                    "d_h": 16,
                }
            ),
        }
    )
    slug = get_gnnplus_model_slug(local)
    assert slug == "hybrid_a2g2_GCNE-GINE_dh16"


def test_load_and_plot_from_pickle() -> None:
    """Pickle round-trip produces two plot files."""
    graph_dict = {0: [1.0, 0.0], 1: [0.5, 0.5]}
    payload = {
        "graph_dict": graph_dict,
        "test_appearances": {0: 2, 1: 2},
        "required_test_appearances": 2,
        "dataset_name": "MUTAG",
        "model_slug": "gcne",
        "num_layers": 4,
        "task_type": "classification",
    }
    with tempfile.TemporaryDirectory() as tmp:
        layers_dir = Path(tmp) / "4_layers"
        layers_dir.mkdir()
        pickle_path = layers_dir / "MUTAG_gcne_graph_dict.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(payload, f)
        by_index, by_acc = load_and_plot_average_per_graph(
            str(pickle_path),
            dataset_name="MUTAG",
            layer_type="gcne",
            encoding=None,
            num_layers=4,
            output_dir=tmp,
            model_slug="gcne",
        )
        assert Path(by_index).is_file()
        assert Path(by_acc).is_file()
        assert by_index.endswith("_by_index.png")
        assert by_acc.endswith("_by_accuracy.png")


def test_infer_plot_task_type() -> None:
    """Regression maps to regression plots; others to classification."""
    assert infer_plot_task_type("regression") == "regression"
    assert infer_plot_task_type("classification_multilabel") == "classification"
