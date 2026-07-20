#!/usr/bin/env python3
"""Generate graph-level heterogeneity profiles with GNNPlus models.

Protocol (SiGMA paper §3 / Heterogeneity_Profile):
  - random train/val/test = 50/25/25 each trial
  - train for ``optim.max_epoch`` (default 300), keep val-best weights
  - record per-graph test correctness (0/1)
  - repeat until every graph appeared in the test set ≥ N times (default 100)

Example::

    python scripts/heterogeneity/run_heterogeneity_profiles.py \\
        --cfg configs/heterogeneity/mutag-gcn.yaml \\
        --required_test_appearances 100

    # Smoke test
    python scripts/heterogeneity/run_heterogeneity_profiles.py \\
        --cfg configs/heterogeneity/mutag-gcn.yaml \\
        --required_test_appearances 2 --max_trials 20
"""

from __future__ import annotations

import argparse
import copy
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow ``python scripts/heterogeneity/...`` without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.graphgym.cmd_args import parse_args as gg_parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import (
    OptimizerConfig,
    create_optimizer,
    create_scheduler,
)
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

import GNNPlus  # noqa: F401  — register custom modules
from GNNPlus.experiments.track_avg_accuracy import load_and_plot_average_per_graph
from GNNPlus.optimizer.extra_optimizers import ExtendedSchedulerConfig


def _optimizer_config() -> OptimizerConfig:
    """Build GraphGym optimizer config from ``cfg``."""
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )


def _scheduler_config() -> ExtendedSchedulerConfig:
    """Build extended scheduler config from ``cfg``."""
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def _accuracy_from_pred(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Return per-graph correctness (0/1) for a classification batch."""
    if pred.ndim > 1 and pred.size(-1) > 1:
        pred_cls = pred.argmax(dim=-1)
    else:
        pred_cls = (pred > 0).long().view(-1)
    true_cls = true.view(-1).long()
    return (pred_cls == true_cls).float()


@torch.no_grad()
def _split_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    """Mean graph classification accuracy on a loader."""
    model.eval()
    device = torch.device(cfg.accelerator)
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        batch.split = "val"
        pred, true = model(batch)
        # ``compute_loss`` returns (loss, pred_score); pred_score is logits.
        _, pred_score = compute_loss(pred, true)
        batch_correct = _accuracy_from_pred(pred_score, true)
        correct += int(batch_correct.sum().item())
        total += int(batch_correct.numel())
    return float(correct) / float(max(total, 1))


@torch.no_grad()
def _per_graph_correctness(
    model: torch.nn.Module,
    dataset: Any,
    test_indices: Sequence[int],
) -> Dict[int, int]:
    """Evaluate each test graph and return ``{global_idx: 0|1}``."""
    model.eval()
    device = torch.device(cfg.accelerator)
    out: Dict[int, int] = {}
    for idx in test_indices:
        data = dataset[int(idx)]
        if not isinstance(data, Data):
            raise TypeError(f"Expected PyG Data, got {type(data)}")
        batch = DataLoader([data], batch_size=1)
        graph = next(iter(batch)).to(device)
        graph.split = "test"
        pred, true = model(graph)
        _, pred_score = compute_loss(pred, true)
        ok = int(_accuracy_from_pred(pred_score, true).item())
        out[int(idx)] = ok
    return out


def _train_one_trial(
    model: torch.nn.Module,
    loaders: List[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> Tuple[torch.nn.Module, float, float]:
    """Train for ``cfg.optim.max_epoch``, restore val-best weights.

    Returns:
        (model, best_val_acc, test_acc_at_best_val)
    """
    device = torch.device(cfg.accelerator)
    model.to(device)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -1.0
    test_at_best = 0.0

    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]

    for epoch in range(int(cfg.optim.max_epoch)):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            batch.split = "train"
            optimizer.zero_grad()
            pred, true = model(batch)
            loss, _ = compute_loss(pred, true)
            loss.backward()
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.optim.clip_grad_norm_value,
                )
            optimizer.step()

        val_acc = _split_accuracy(model, val_loader)
        if cfg.optim.scheduler == "reduce_on_plateau":
            scheduler.step(1.0 - val_acc)
        else:
            scheduler.step()

        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            test_at_best = _split_accuracy(model, test_loader)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            logging.info(
                "  epoch %d/%d  val_acc=%.4f  best_val=%.4f  test@best=%.4f",
                epoch + 1,
                cfg.optim.max_epoch,
                val_acc,
                best_val,
                test_at_best,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, test_at_best


def _model_tag() -> str:
    """Short tag for filenames / plots."""
    mtype = str(cfg.model.type)
    if mtype == "hybrid_gnn":
        return "SiGMA"
    layer = str(getattr(cfg.gnn, "layer_type", mtype))
    return layer.upper()


def run_profile(
    *,
    required_test_appearances: int,
    max_trials: int,
    seed0: int,
    output_dir: Path,
) -> Path:
    """Run the multi-trial heterogeneity protocol for the loaded ``cfg``.

    Args:
        required_test_appearances: Stop when every graph has this many test hits.
        max_trials: Hard cap on trials (safety).
        seed0: First trial seed; trial ``t`` uses ``seed0 + t - 1``.
        output_dir: Directory for pickle + plots.

    Returns:
        Path to the saved ``graph_dict`` pickle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_printing()
    auto_select_device()

    # Probe dataset size with seed0.
    cfg.seed = seed0
    seed_everything(cfg.seed)
    loaders0 = create_loader()
    # Full dataset lives under the Subset's .dataset
    train_subset = loaders0[0].dataset
    full_dataset = train_subset.dataset if hasattr(train_subset, "dataset") else train_subset
    n_graphs = len(full_dataset)
    logging.info("Dataset %s: %d graphs", cfg.dataset.name, n_graphs)

    graph_dict: Dict[int, List[int]] = {i: [] for i in range(n_graphs)}
    test_appearances: Dict[int, int] = {i: 0 for i in range(n_graphs)}

    trial = 0
    while True:
        trial += 1
        min_app = min(test_appearances.values())
        if min_app >= required_test_appearances:
            logging.info(
                "All graphs have ≥%d test appearances; stopping before trial %d",
                required_test_appearances,
                trial,
            )
            break
        if trial > max_trials:
            logging.warning(
                "Hit max_trials=%d with min appearances=%d/%d",
                max_trials,
                min_app,
                required_test_appearances,
            )
            break

        trial_seed = seed0 + trial - 1
        cfg.seed = trial_seed
        cfg.run_id = trial_seed
        seed_everything(cfg.seed)
        logging.info(
            "=== Trial %d  seed=%d  min_test_appearances=%d/%d ===",
            trial,
            trial_seed,
            min_app,
            required_test_appearances,
        )

        loaders = create_loader()
        model = create_model()
        cfg.params = params_count(model)
        optimizer = create_optimizer(model.parameters(), _optimizer_config())
        scheduler = create_scheduler(optimizer, _scheduler_config())

        model, best_val, test_at_best = _train_one_trial(
            model, loaders, optimizer, scheduler
        )
        logging.info(
            "Trial %d done: best_val=%.4f test@best=%.4f",
            trial,
            best_val,
            test_at_best,
        )

        test_subset = loaders[2].dataset
        if not hasattr(test_subset, "indices"):
            raise RuntimeError(
                "Expected a Subset test loader with .indices for global graph IDs"
            )
        test_indices = [int(i) for i in test_subset.indices]
        # Evaluate on the same underlying dataset object as the Subset.
        base = test_subset.dataset
        correctness = _per_graph_correctness(model, base, test_indices)
        for gidx, ok in correctness.items():
            test_appearances[gidx] += 1
            graph_dict[gidx].append(ok)

        logging.info(
            "After trial %d: min=%d max=%d mean_app=%.2f",
            trial,
            min(test_appearances.values()),
            max(test_appearances.values()),
            float(np.mean(list(test_appearances.values()))),
        )

    tag = _model_tag()
    ds = str(cfg.dataset.name).lower()
    layers = int(cfg.gnn.layers_mp)
    pickle_path = output_dir / f"{ds}_{tag}_L{layers}_graph_dict.pickle"
    payload = {
        "graph_dict": graph_dict,
        "test_appearances": test_appearances,
        "required_test_appearances": required_test_appearances,
        "dataset": cfg.dataset.name,
        "model_tag": tag,
        "model_type": cfg.model.type,
        "layer_type": getattr(cfg.gnn, "layer_type", None),
        "num_layers": layers,
        "n_trials_run": trial - 1 if min(test_appearances.values()) >= required_test_appearances else trial,
        "seed0": seed0,
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(payload, f)
    logging.info("Saved %s", pickle_path)

    load_and_plot_average_per_graph(
        str(pickle_path),
        dataset_name=ds,
        layer_type=tag,
        encoding=None,
        num_layers=layers,
        task_type="classification",
        output_dir=str(output_dir),
    )
    return pickle_path


def main() -> None:
    """CLI entrypoint."""
    # Allow GraphGym-style `--cfg` plus our extras. We parse known GraphGym args
    # via a thin argv shim, then our own flags.
    parser = argparse.ArgumentParser(
        description="GNNPlus heterogeneity profile generator (TU datasets).",
    )
    parser.add_argument("--cfg", required=True, help="Path to GraphGym YAML config")
    parser.add_argument(
        "--required_test_appearances",
        type=int,
        default=100,
        help="Stop when every graph has been in the test set this many times",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=2000,
        help="Hard cap on resampled train/val/test trials",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for trial 1")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Where to write pickle/plots (default: <out_dir>/<dataset>_<model>)",
    )
    # Remaining GraphGym overrides: key value pairs after --
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional GraphGym cfg overrides (e.g. optim.max_epoch 50)",
    )
    args = parser.parse_args()

    # GraphGym load_cfg expects sys.argv-style via parse_args; call it cleanly.
    gg_argv = ["--cfg", args.cfg]
    opts = [o for o in args.opts if o != "--"]
    gg_argv.extend(opts)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *gg_argv]
        gg_args = gg_parse_args()
        set_cfg(cfg)
        load_cfg(cfg, gg_args)
    finally:
        sys.argv = old_argv

    # Force protocol defaults if YAML forgot them.
    cfg.dataset.split_mode = "random"
    if not getattr(cfg.dataset, "split", None):
        cfg.dataset.split = [0.5, 0.25, 0.25]
    cfg.wandb.use = False
    cfg.train.enable_ckpt = False

    torch.set_num_threads(cfg.num_threads)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    tag = "SiGMA" if cfg.model.type == "hybrid_gnn" else str(
        getattr(cfg.gnn, "layer_type", cfg.model.type)
    ).upper()
    out = Path(args.output_dir) if args.output_dir else Path(cfg.out_dir) / (
        f"{str(cfg.dataset.name).lower()}_{tag}"
    )

    run_profile(
        required_test_appearances=args.required_test_appearances,
        max_trials=args.max_trials,
        seed0=args.seed,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
