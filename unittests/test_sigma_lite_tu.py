"""Tests for the SiGMA-lite TU helpers and the ``scatter`` fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
import torch

import GNNPlus.scatter_compat as scatter_compat
from GNNPlus.sigma_lite_tu import (
    TAG_RE,
    RunSpec,
    best_test_accuracy,
    build_command,
    make_specs,
    summarize,
)


def _write_stats(path: Path, accuracies: List[float]) -> None:
    """Write a GraphGym-style ``stats.json`` with one accuracy per epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"epoch": i, "accuracy": a}) + "\n" for i, a in enumerate(accuracies)))


def test_tag_round_trips_through_regex() -> None:
    """Run tags match the aggregation regex for both tables."""
    for spec in (RunSpec("IMDB-BINARY", 16, 0.001, 3), RunSpec("MUTAG", 4, 0.01, 0, "gelu")):
        match = TAG_RE.match(spec.tag)
        assert match is not None
        assert match["dataset"] == spec.dataset
        assert int(match["d_h"]) == spec.d_h
        assert match["gate"] == spec.gate_act
        assert float(match["lr"]) == spec.lr
        assert int(match["seed"]) == spec.seed


def test_paper_batch_sizes() -> None:
    """COLLAB and REDDIT-BINARY use the paper's smaller batch sizes."""
    assert RunSpec("MUTAG", 16, 0.001, 0).resolved_batch_size == 64
    assert RunSpec("COLLAB", 16, 0.001, 0).resolved_batch_size == 32
    assert RunSpec("REDDIT-BINARY", 4, 0.001, 0).resolved_batch_size == 16
    assert RunSpec("REDDIT-BINARY", 4, 0.001, 0, batch_size=8).resolved_batch_size == 8


def test_make_specs_grid() -> None:
    """Tables map to d_h and the grid has tables x datasets x lrs x seeds runs."""
    specs = make_specs(["MUTAG", "ENZYMES"], tables=(17, 18))
    assert len(specs) == 2 * 2 * 2 * 5
    assert {s.d_h for s in specs} == {16, 4}
    with pytest.raises(ValueError):
        make_specs(["MUTAG"], tables=(19,))


def test_build_command_overrides(tmp_path: Path) -> None:
    """The command carries the per-run overrides and the W&B entity."""
    spec = RunSpec("COLLAB", 4, 0.01, 2)
    cmd = build_command(spec, tmp_path, tmp_path / "data", wandb=True, wandb_entity="team")
    pairs = dict(zip(cmd[4::2], cmd[5::2]))
    assert cmd[1:4] == ["main.py", "--cfg", "configs/sigma_lite/tu-sigma-lite-a2g4.yaml"]
    assert pairs["dataset.name"] == "COLLAB"
    assert pairs["gnn.hybrid.d_h"] == "4"
    assert pairs["optim.base_lr"] == "0.01"
    assert pairs["train.batch_size"] == "32"
    assert pairs["wandb.use"] == "True"
    assert pairs["wandb.entity"] == "team"
    assert pairs["out_dir"].endswith(spec.tag)


def test_best_test_accuracy_uses_first_best_val_epoch(tmp_path: Path) -> None:
    """Ties in validation accuracy resolve to the earliest epoch."""
    run_dir = tmp_path / "run"
    _write_stats(run_dir / "val" / "stats.json", [0.5, 0.8, 0.8, 0.7])
    _write_stats(run_dir / "test" / "stats.json", [0.4, 0.6, 0.9, 0.7])
    assert best_test_accuracy(run_dir) == (0.6, 1, 4)
    assert best_test_accuracy(tmp_path / "missing") is None


def test_summarize_picks_best_lr(tmp_path: Path) -> None:
    """The summary reports the learning rate with the highest mean."""
    for lr, test in ((0.001, [0.8, 0.9]), (0.01, [0.5, 0.6])):
        for seed, acc in enumerate(test):
            run_dir = tmp_path / RunSpec("MUTAG", 16, lr, seed).tag
            _write_stats(run_dir / "val" / "stats.json", [0.7])
            _write_stats(run_dir / "test" / "stats.json", [acc])
    text = summarize(tmp_path)
    assert "85.00±7.07  (lr=0.001, n=2)" in text
    assert "No finished runs" in summarize(tmp_path / "empty")


@pytest.mark.parametrize("reduce", ["sum", "add", "mean", "max"])
def test_scatter_fallback_matches_dense(reduce: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The PyG fallback reproduces a dense reference reduction."""
    monkeypatch.setattr(scatter_compat, "_torch_scatter", None)
    src = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    index = torch.tensor([0, 0, 2, 2])
    out = scatter_compat.scatter(src, index, 0, None, 3, reduce=reduce)
    expected = {
        "sum": [[3.0], [0.0], [7.0]],
        "add": [[3.0], [0.0], [7.0]],
        "mean": [[1.5], [0.0], [3.5]],
        "max": [[2.0], [0.0], [4.0]],
    }[reduce]
    assert torch.equal(out, torch.tensor(expected))
    with pytest.raises(NotImplementedError):
        scatter_compat.scatter(src, index, 0, torch.zeros(3, 1), 3)
