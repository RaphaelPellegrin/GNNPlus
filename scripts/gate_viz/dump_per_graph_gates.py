#!/usr/bin/env python3
"""Dump per-graph / per-head SiGMA gate values from a GraphGym checkpoint.

Given a training ``run_dir`` (contains ``ckpt/`` and was created with the same
yaml), rebuilds the model, loads the latest (or chosen) checkpoint, and saves:

  ``gate_values_per_graph.pt`` with keys:
    - ``attn``: FloatTensor ``[N_graphs, L, Na]`` (train+val+test concat order)
    - ``gnn``:  FloatTensor ``[N_graphs, L, Ng]``
    - ``y``:    labels when available
    - ``split``: LongTensor with 0=train, 1=val, 2=test
    - ``meta``: dict (cfg paths, epoch, seed, …)

Example:
  python scripts/gate_viz/dump_per_graph_gates.py \\
    --run_dir results/gate_viz_enzymes_ogpkubk9_plateau_seed2 \\
    --cfg configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml
"""

from __future__ import annotations

import argparse
import logging
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.graphgym.checkpoint import get_ckpt_epochs, load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

# Ensure repo root is on path when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401  — register custom modules
from GNNPlus.hybrid_gate_tracking import _unwrap_model


def _parse_dump_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for gate dump (keeps GraphGym ``--cfg`` via remaining argv)."""
    parser = argparse.ArgumentParser(
        description='Dump per-graph SiGMA gates from a checkpoint.',
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='GraphGym run directory containing ckpt/.',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=-1,
        help='Checkpoint epoch to load (-1 = latest).',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='',
        help='Output .pt path (default: <run_dir>/gate_values_per_graph.pt).',
    )
    parser.add_argument(
        '--splits',
        type=str,
        default='train,val,test',
        help='Comma-separated loader splits to dump (default: train,val,test).',
    )
    known, remaining = parser.parse_known_args(argv)
    known._remaining = remaining
    return known


def _pick_epoch(run_dir: str, epoch: int) -> int:
    """Resolve checkpoint epoch under ``run_dir/ckpt``."""
    cfg.run_dir = run_dir
    epochs = list(get_ckpt_epochs())
    if not epochs:
        raise FileNotFoundError(f'No checkpoints under {run_dir}/ckpt')
    if epoch < 0:
        return int(max(epochs))
    if epoch not in epochs:
        raise FileNotFoundError(
            f'Epoch {epoch} not in {epochs} under {run_dir}/ckpt'
        )
    return int(epoch)


def _collect_loader_gates(
    model: torch.nn.Module,
    loader: Any,
    split_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Run ``collect_per_graph_gates`` over a loader; concat batch results."""
    core = _unwrap_model(model)
    if not hasattr(core, 'collect_per_graph_gates'):
        raise TypeError(
            f'Model {type(core).__name__} has no collect_per_graph_gates'
        )

    attn_parts: List[torch.Tensor] = []
    gnn_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    split_parts: List[torch.Tensor] = []
    have_y = True

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = core.collect_per_graph_gates(batch)
                attn_parts.append(out['attn'])
                gnn_parts.append(out['gnn'])
                n = int(out['num_graphs'])
                split_parts.append(
                    torch.full((n,), split_id, dtype=torch.long)
                )
                if out['y'] is None:
                    have_y = False
                else:
                    y_parts.append(out['y'])
    finally:
        if was_training:
            model.train()

    if not attn_parts:
        empty = torch.zeros(0, 0, 0)
        return empty, empty, None, torch.zeros(0, dtype=torch.long)

    attn = torch.cat(attn_parts, dim=0)
    gnn = torch.cat(gnn_parts, dim=0)
    split = torch.cat(split_parts, dim=0)
    y: Optional[torch.Tensor] = torch.cat(y_parts, dim=0) if have_y and y_parts else None
    return attn, gnn, y, split


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load cfg + ckpt and write per-graph gate tensors."""
    dump_args = _parse_dump_args(argv)
    # GraphGym expects sys.argv-style: script --cfg path [opts...]
    # Rebuild argv for parse_args / load_cfg.
    gg_argv = list(dump_args._remaining)
    if '--cfg' not in gg_argv and '-cfg' not in gg_argv:
        raise SystemExit(
            'Pass GraphGym --cfg <yaml> (same config used for training).'
        )
    # Temporarily set argv for GraphGym parse_args.
    old_argv = sys.argv
    sys.argv = [old_argv[0], *gg_argv]
    try:
        args = parse_args()
        set_cfg(cfg)
        load_cfg(cfg, args)
    finally:
        sys.argv = old_argv

    run_dir = osp.abspath(dump_args.run_dir)
    cfg.run_dir = run_dir
    cfg.out_dir = osp.dirname(run_dir) or cfg.out_dir
    seed_everything(int(cfg.seed))
    auto_select_device()
    device = torch.device(cfg.accelerator)

    loaders = create_loader()
    model = create_model()
    epoch = _pick_epoch(run_dir, int(dump_args.epoch))
    # load_ckpt restores weights into ``model``; optimizer/scheduler unused.
    loaded = load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    logging.info('Loaded checkpoint epoch=%s (start_epoch=%s)', epoch, loaded)

    name_to_id = {'train': 0, 'val': 1, 'test': 2}
    wanted = [s.strip().lower() for s in dump_args.splits.split(',') if s.strip()]
    for name in wanted:
        if name not in name_to_id:
            raise ValueError(f'Unknown split {name!r}; use train,val,test')

    # GraphGym create_loader order: train, val, test.
    loader_by_name = {
        'train': loaders[0] if len(loaders) > 0 else None,
        'val': loaders[1] if len(loaders) > 1 else None,
        'test': loaders[2] if len(loaders) > 2 else None,
    }

    attn_all: List[torch.Tensor] = []
    gnn_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    split_all: List[torch.Tensor] = []
    have_y = True

    for name in wanted:
        loader = loader_by_name[name]
        if loader is None:
            logging.warning('Split %s missing; skipping.', name)
            continue
        attn, gnn, y, split = _collect_loader_gates(
            model, loader, name_to_id[name], device
        )
        logging.info(
            'Split %s: %d graphs, attn%s gnn%s',
            name,
            attn.size(0),
            tuple(attn.shape),
            tuple(gnn.shape),
        )
        attn_all.append(attn)
        gnn_all.append(gnn)
        split_all.append(split)
        if y is None:
            have_y = False
        else:
            y_all.append(y)

    payload: Dict[str, Any] = {
        'attn': torch.cat(attn_all, dim=0) if attn_all else torch.zeros(0, 0, 0),
        'gnn': torch.cat(gnn_all, dim=0) if gnn_all else torch.zeros(0, 0, 0),
        'split': torch.cat(split_all, dim=0) if split_all else torch.zeros(0),
        'y': torch.cat(y_all, dim=0) if have_y and y_all else None,
        'meta': {
            'run_dir': run_dir,
            'epoch': epoch,
            'seed': int(cfg.seed),
            'dataset': str(cfg.dataset.name),
            'num_attn_heads': int(cfg.gnn.hybrid.num_attn_heads),
            'num_gnn_heads': int(cfg.gnn.hybrid.num_gnn_heads),
            'layers_mp': int(cfg.gnn.layers_mp),
            'gate': str(cfg.gnn.hybrid.gate),
            'gnn_types': str(cfg.gnn.hybrid.gnn_types),
            'splits': wanted,
        },
    }
    out_path = dump_args.out or osp.join(run_dir, 'gate_values_per_graph.pt')
    torch.save(payload, out_path)
    logging.info(
        'Wrote %s (attn%s gnn%s)',
        out_path,
        tuple(payload['attn'].shape),
        tuple(payload['gnn'].shape),
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
