#!/usr/bin/env python3
"""Train a TU model and plot **all-layer** activations per graph.

Activation (y-axis): mean over nodes of ``||h_v||_2`` after each GNN/hybrid
layer (before the classifier). X-axis: graph index.

Snapshots:
  * ``mid``  — weights at epoch ``max_epoch // 2``
  * ``last`` — weights after the final training epoch
  * ``best`` — validation-best weights (also used for Acc reporting)

Example::

    python scripts/heterogeneity/run_last_layer_activations.py \\
        --cfg configs/heterogeneity/mutag-sigma.yaml \\
        --max_epoch 5 --seed 0 --no-wandb \\
        --output_dir results/activations/mutag_smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data
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
from torch_geometric.loader import DataLoader

import GNNPlus  # noqa: F401
from GNNPlus.experiments.last_layer_activations import (
    dump_all_layer_plots,
    per_graph_mean_node_l2,
)
from GNNPlus.optimizer.extra_optimizers import ExtendedSchedulerConfig


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """Parse script args; remaining tokens are GraphGym cfg overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, type=str, help='Config yaml')
    parser.add_argument('--seed', type=int, default=0, help='Train/eval seed')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='Output dir (default: <out_dir>/activations/<ds>_<model>)',
    )
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=None,
        help='Override optim.max_epoch (default: cfg)',
    )
    parser.add_argument(
        '--mid_frac',
        type=float,
        default=0.5,
        help='Fraction of max_epoch for mid-training snapshot (default: 0.5)',
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _optimizer_config() -> OptimizerConfig:
    """Build optimizer config from ``cfg``."""
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )


def _scheduler_config() -> ExtendedSchedulerConfig:
    """Build scheduler config from ``cfg``."""
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


def _pred_classes(pred: torch.Tensor) -> torch.Tensor:
    """Class predictions from raw logits (binary or multiclass)."""
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if pred.ndim > 1 and pred.size(-1) > 1:
        return pred.argmax(dim=-1)
    return (pred > 0).long().view(-1)


@torch.no_grad()
def _split_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    """Mean graph accuracy using logit thresholding (not sigmoid>0)."""
    model.eval()
    device = torch.device(cfg.accelerator)
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        batch.split = 'val'
        pred, true = model(batch)
        if true.ndim > 1:
            true = true.squeeze(-1)
        pred_cls = _pred_classes(pred)
        correct += int((pred_cls == true.view(-1).long()).sum().item())
        total += int(true.numel())
    return float(correct) / float(max(total, 1))


def _train(
    model: torch.nn.Module,
    loaders: List[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    mid_epoch: int,
) -> Tuple[
    torch.nn.Module,
    float,
    float,
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict[str, torch.Tensor]],
    int,
]:
    """Train; return model + Acc + best/mid/last state dicts + mid epoch used."""
    device = torch.device(cfg.accelerator)
    model.to(device)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    mid_state: Optional[Dict[str, torch.Tensor]] = None
    last_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -1.0
    test_at_best = 0.0
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    max_epoch = int(cfg.optim.max_epoch)
    mid_epoch = int(max(1, min(mid_epoch, max_epoch)))

    for epoch in range(max_epoch):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            batch.split = 'train'
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
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(1.0 - val_acc)
        else:
            scheduler.step()

        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            test_at_best = _split_accuracy(model, test_loader)

        if (epoch + 1) == mid_epoch:
            mid_state = copy.deepcopy(model.state_dict())
            logging.info('  mid-epoch snapshot at epoch %d', mid_epoch)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            logging.info(
                '  epoch %d/%d  val_acc=%.4f  best_val=%.4f  test@best=%.4f',
                epoch + 1,
                max_epoch,
                val_acc,
                best_val,
                test_at_best,
            )

    last_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, test_at_best, best_state, mid_state, last_state, mid_epoch


def _unwrap_dataset(dataset: Any) -> Any:
    """Unwrap Subset wrappers."""
    from torch.utils.data import Subset

    cur: Any = dataset
    while isinstance(cur, Subset):
        cur = cur.dataset
    return cur


def _full_num_graphs(dataset: Any) -> int:
    """Total graphs ignoring active split filter."""
    base = _unwrap_dataset(dataset)
    len_fn = getattr(base, 'len', None)
    if callable(len_fn):
        try:
            return int(len_fn())
        except NotImplementedError:
            pass
    return int(len(base))


def _get_graph(dataset: Any, idx: int) -> Data:
    """Fetch graph by global index via ``Dataset.get`` when available."""
    base = _unwrap_dataset(dataset)
    get_fn = getattr(base, 'get', None)
    if callable(get_fn):
        data = get_fn(int(idx))
        transform = getattr(base, 'transform', None)
        if transform is not None:
            data = transform(data)
        return data
    return base[int(idx)]


def _inner_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap GraphGymModule to the registered network when present."""
    inner = getattr(model, 'model', None)
    return inner if isinstance(inner, torch.nn.Module) else model


@torch.no_grad()
def collect_all_layer_activations(
    model: torch.nn.Module,
    dataset: Any,
    n_graphs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-graph mean node L2 after every layer.

    Returns:
        ``(indices, layer_activations[L,G], labels[G])``.
    """
    model.eval()
    device = torch.device(cfg.accelerator)
    inner = _inner_model(model)
    if not hasattr(inner, 'forward_all_layer_features'):
        raise TypeError(
            f'{type(inner).__name__} lacks forward_all_layer_features; '
            'use hybrid_gnn or custom_gnn'
        )

    indices: List[int] = []
    labels: List[int] = []
    per_graph_layers: List[List[float]] = []

    for gidx in range(n_graphs):
        data = _get_graph(dataset, gidx)
        batch = next(iter(DataLoader([data], batch_size=1))).to(device)
        batch.split = 'test'
        layer_xs, batch = inner.forward_all_layer_features(batch)
        layer_vals = [
            float(per_graph_mean_node_l2(x, batch.batch).item()) for x in layer_xs
        ]
        y = batch.y
        y = y.view(-1)[0]
        indices.append(gidx)
        labels.append(int(y.item()))
        per_graph_layers.append(layer_vals)

    # Transpose to [L, G]
    layer_activations = np.asarray(per_graph_layers, dtype=np.float64).T
    return (
        np.asarray(indices, dtype=np.int64),
        layer_activations,
        np.asarray(labels, dtype=np.int64),
    )


def _model_tag() -> str:
    """Short model tag for filenames."""
    mtype = str(cfg.model.type)
    if mtype == 'hybrid_gnn':
        return 'sigma'
    layer = str(getattr(cfg.gnn, 'layer_type', '') or '')
    return layer or mtype


def _dump_snapshot(
    model: torch.nn.Module,
    state: Optional[Dict[str, torch.Tensor]],
    *,
    dataset: Any,
    n_graphs: int,
    out_dir: Path,
    ds_name: str,
    tag: str,
    epoch_tag: str,
) -> Dict[str, Any]:
    """Load ``state`` into ``model`` and dump all-layer activation artifacts."""
    if state is None:
        logging.warning('No state for snapshot %s — skipping', epoch_tag)
        return {}
    model.load_state_dict(state)
    idxs, layer_acts, labels = collect_all_layer_activations(model, dataset, n_graphs)
    snap_dir = out_dir / epoch_tag
    paths = dump_all_layer_plots(
        snap_dir,
        dataset_name=ds_name,
        model_tag=tag,
        graph_indices=idxs,
        layer_activations=layer_acts,
        labels=labels,
        epoch_tag=epoch_tag,
    )
    summary = {
        'epoch_tag': epoch_tag,
        'n_layers': int(layer_acts.shape[0]),
        'n_graphs': int(n_graphs),
        'per_layer_mean': [float(layer_acts[i].mean()) for i in range(layer_acts.shape[0])],
        'per_layer_std': [float(layer_acts[i].std()) for i in range(layer_acts.shape[0])],
        'paths': paths,
    }
    (snap_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    logging.info(
        'Wrote %s activations: %d layers × %d graphs → %s',
        epoch_tag,
        layer_acts.shape[0],
        n_graphs,
        snap_dir,
    )
    return summary


def main() -> None:
    """Train then dump all-layer activations at mid / last / val-best."""
    args, overrides = _parse_args()
    set_cfg(cfg)

    class _Opts:
        cfg_file = args.cfg
        opts = overrides
        repeat = 1

    load_cfg(cfg, _Opts())
    if args.max_epoch is not None:
        cfg.optim.max_epoch = int(args.max_epoch)
    cfg.seed = int(args.seed)
    cfg.run_id = int(args.seed)
    cfg.wandb.use = bool(args.wandb)

    set_printing()
    auto_select_device()
    seed_everything(cfg.seed)

    ds_name = str(cfg.dataset.name).lower()
    tag = _model_tag()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        base = Path(str(cfg.out_dir))
        out_dir = base / 'activations' / f'{ds_name}_{tag}_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = str(out_dir)

    loaders = create_loader()
    model = create_model()
    cfg.params = params_count(model)
    optimizer = create_optimizer(model.parameters(), _optimizer_config())
    scheduler = create_scheduler(optimizer, _scheduler_config())

    max_epoch = int(cfg.optim.max_epoch)
    mid_epoch = max(1, int(round(max_epoch * float(args.mid_frac))))

    logging.info(
        'Training %s / %s  seed=%d  max_epoch=%d  mid_epoch=%d  params=%s',
        ds_name,
        tag,
        args.seed,
        max_epoch,
        mid_epoch,
        cfg.params,
    )
    (
        model,
        best_val,
        best_test,
        best_state,
        mid_state,
        last_state,
        mid_epoch_used,
    ) = _train(model, loaders, optimizer, scheduler, mid_epoch=mid_epoch)
    logging.info(
        'Done training: best_val=%.4f  test@best=%.4f  mid_epoch=%d',
        best_val,
        best_test,
        mid_epoch_used,
    )

    for name, state in (
        ('best', best_state),
        ('mid', mid_state),
        ('last', last_state),
    ):
        if state is None:
            continue
        torch.save(
            {
                'model_state': state,
                'seed': int(args.seed),
                'snapshot': name,
                'best_val_acc': float(best_val),
                'best_test_acc': float(best_test),
                'cfg_file': args.cfg,
            },
            out_dir / f'{name}.pt',
        )

    full_ds = _unwrap_dataset(loaders[0].dataset)
    n_graphs = _full_num_graphs(full_ds)

    snap_summaries: Dict[str, Any] = {}
    for epoch_tag, state in (
        ('mid', mid_state),
        ('last', last_state),
        ('best', best_state),
    ):
        snap_summaries[epoch_tag] = _dump_snapshot(
            model,
            state,
            dataset=full_ds,
            n_graphs=n_graphs,
            out_dir=out_dir,
            ds_name=ds_name,
            tag=tag,
            epoch_tag=epoch_tag,
        )

    summary = {
        'dataset': ds_name,
        'model': tag,
        'seed': int(args.seed),
        'best_val_acc': float(best_val),
        'best_test_acc': float(best_test),
        'max_epoch': max_epoch,
        'mid_epoch': mid_epoch_used,
        'n_graphs': int(n_graphs),
        'snapshots': snap_summaries,
    }
    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    logging.info('Wrote %s', summary_path)

    if args.wandb and cfg.wandb.use:
        try:
            import wandb

            run = wandb.init(
                entity=cfg.wandb.entity or 'weber-geoml-harvard-university',
                project=cfg.wandb.project or 'GNNPlus',
                group=f'layer_act_{ds_name}',
                name=f'{ds_name}_{tag}_seed{args.seed}',
                config={
                    k: v
                    for k, v in summary.items()
                    if k != 'snapshots'
                },
                reinit=True,
            )
            run.summary.update(
                {
                    'best_val_acc': best_val,
                    'best_test_acc': best_test,
                    'mid_epoch': mid_epoch_used,
                    'n_graphs': n_graphs,
                }
            )
            for png in out_dir.rglob('*.png'):
                run.log({f'{png.parent.name}/{png.stem}': wandb.Image(str(png))})
            run.finish()
        except Exception as exc:  # pragma: no cover
            logging.warning('W&B logging failed: %s', exc)


if __name__ == '__main__':
    main()
