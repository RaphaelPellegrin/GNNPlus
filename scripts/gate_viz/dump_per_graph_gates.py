#!/usr/bin/env python3
"""Dump SiGMA gate values from a GraphGym checkpoint.

Given a training ``run_dir`` (contains ``ckpt/`` and was created with the same
yaml), rebuilds the model, loads the latest (or chosen) checkpoint, and saves:

  Graph-level (default ``gate_values_per_graph.pt``):
    - ``attn`` / ``gnn``: FloatTensor ``[G, L, H]`` — mean γ over nodes
    - ``y``, ``split``, ``meta``

  Node-level (``gate_values_per_node.pt`` when ``--level node|both``):
    - ``attn`` / ``gnn``: FloatTensor ``[N, L, H]`` — per-node γ
    - ``batch``: LongTensor ``[N]`` global graph index (matches graph dump order)
    - ``ptr``: LongTensor ``[G+1]`` CSR offsets into the node tensors
    - ``edge_index``: LongTensor ``[2, E]`` (node ids in the concatenated space)
    - ``edge_batch`` / ``edge_ptr``: edge → graph CSR (for drawings)
    - ``y``, ``split`` at graph level; ``meta``

  Split one graph ``g`` with::
    attn_nodes = payload['attn'][payload['ptr'][g] : payload['ptr'][g + 1]]
    edges = payload['edge_index'][:, payload['edge_ptr'][g] : payload['edge_ptr'][g + 1]]

Example:
  python scripts/gate_viz/dump_per_graph_gates.py \\
    --run_dir results/.../mutag_SiGMA_hetero_lr001_seed2 \\
    --level both \\
    --cfg configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml
"""

from __future__ import annotations

import argparse
import logging
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        description='Dump per-graph / per-node SiGMA gates from a checkpoint.',
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
        help='Checkpoint epoch to load (-1 = latest / best depending on ckpt).',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='',
        help='Graph-level .pt path (default: <run_dir>/gate_values_per_graph.pt).',
    )
    parser.add_argument(
        '--out-node',
        type=str,
        default='',
        help='Node-level .pt path (default: <run_dir>/gate_values_per_node.pt).',
    )
    parser.add_argument(
        '--level',
        type=str,
        default='graph',
        choices=('graph', 'node', 'both'),
        help='Dump graph means, packed per-node gates, or both (default: graph).',
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


def _ptr_from_batch(batch_ids: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Build CSR ``ptr`` of length ``num_graphs + 1`` from node→graph ids."""
    counts = torch.bincount(batch_ids, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def _collect_loader_gates(
    model: torch.nn.Module,
    loader: Any,
    split_id: int,
    device: torch.device,
    *,
    want_node: bool,
) -> Dict[str, Any]:
    """Run ``collect_per_graph_gates`` over a loader; concat with global ids."""
    core = _unwrap_model(model)
    if not hasattr(core, 'collect_per_graph_gates'):
        raise TypeError(
            f'Model {type(core).__name__} has no collect_per_graph_gates'
        )

    attn_g_parts: List[torch.Tensor] = []
    gnn_g_parts: List[torch.Tensor] = []
    attn_n_parts: List[torch.Tensor] = []
    gnn_n_parts: List[torch.Tensor] = []
    batch_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    edge_batch_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    split_parts: List[torch.Tensor] = []
    have_y = True
    graph_offset = 0
    node_offset = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                # Topology before encode (node ids local to this mini-batch).
                ei_local = batch.edge_index.detach().cpu().long()
                n_nodes_batch = int(batch.num_nodes)
                out = core.collect_per_graph_gates(batch)
                attn_g_parts.append(out['attn'])
                gnn_g_parts.append(out['gnn'])
                n_graphs = int(out['num_graphs'])
                split_parts.append(
                    torch.full((n_graphs,), split_id, dtype=torch.long)
                )
                if out['y'] is None:
                    have_y = False
                else:
                    y_parts.append(out['y'])
                if want_node:
                    attn_n_parts.append(out['attn_node'])
                    gnn_n_parts.append(out['gnn_node'])
                    batch_local = out['batch'].long()
                    batch_parts.append(batch_local + graph_offset)
                    # Remap edges to the concatenated node index space.
                    edge_parts.append(ei_local + int(node_offset))
                    edge_batch_parts.append(
                        batch_local[ei_local[0]].long() + graph_offset
                    )
                    node_offset += n_nodes_batch
                graph_offset += n_graphs
    finally:
        if was_training:
            model.train()

    empty3 = torch.zeros(0, 0, 0)
    result: Dict[str, Any] = {
        'attn': torch.cat(attn_g_parts, dim=0) if attn_g_parts else empty3,
        'gnn': torch.cat(gnn_g_parts, dim=0) if gnn_g_parts else empty3,
        'split': (
            torch.cat(split_parts, dim=0)
            if split_parts
            else torch.zeros(0, dtype=torch.long)
        ),
        'y': torch.cat(y_parts, dim=0) if have_y and y_parts else None,
        'num_graphs': graph_offset,
    }
    if want_node:
        if batch_parts:
            batch_ids = torch.cat(batch_parts, dim=0)
            result['attn_node'] = torch.cat(attn_n_parts, dim=0)
            result['gnn_node'] = torch.cat(gnn_n_parts, dim=0)
            result['batch'] = batch_ids
            result['ptr'] = _ptr_from_batch(batch_ids, graph_offset)
            result['num_nodes'] = int(batch_ids.numel())
            if edge_parts:
                edge_index = torch.cat(edge_parts, dim=1)
                edge_batch = torch.cat(edge_batch_parts, dim=0)
                result['edge_index'] = edge_index
                result['edge_batch'] = edge_batch
                result['edge_ptr'] = _ptr_from_batch(edge_batch, graph_offset)
                result['num_edges'] = int(edge_index.size(1))
            else:
                result['edge_index'] = torch.zeros(2, 0, dtype=torch.long)
                result['edge_batch'] = torch.zeros(0, dtype=torch.long)
                result['edge_ptr'] = torch.zeros(1, dtype=torch.long)
                result['num_edges'] = 0
        else:
            result['attn_node'] = empty3
            result['gnn_node'] = empty3
            result['batch'] = torch.zeros(0, dtype=torch.long)
            result['ptr'] = torch.zeros(1, dtype=torch.long)
            result['num_nodes'] = 0
            result['edge_index'] = torch.zeros(2, 0, dtype=torch.long)
            result['edge_batch'] = torch.zeros(0, dtype=torch.long)
            result['edge_ptr'] = torch.zeros(1, dtype=torch.long)
            result['num_edges'] = 0
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load cfg + ckpt and write per-graph / per-node gate tensors."""
    dump_args = _parse_dump_args(argv)
    gg_argv = list(dump_args._remaining)
    if '--cfg' not in gg_argv and '-cfg' not in gg_argv:
        raise SystemExit(
            'Pass GraphGym --cfg <yaml> (same config used for training).'
        )
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
    loaded = load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    logging.info('Loaded checkpoint epoch=%s (start_epoch=%s)', epoch, loaded)

    name_to_id = {'train': 0, 'val': 1, 'test': 2}
    wanted = [s.strip().lower() for s in dump_args.splits.split(',') if s.strip()]
    for name in wanted:
        if name not in name_to_id:
            raise ValueError(f'Unknown split {name!r}; use train,val,test')

    loader_by_name = {
        'train': loaders[0] if len(loaders) > 0 else None,
        'val': loaders[1] if len(loaders) > 1 else None,
        'test': loaders[2] if len(loaders) > 2 else None,
    }

    level = str(dump_args.level)
    want_graph = level in ('graph', 'both')
    want_node = level in ('node', 'both')

    attn_g_all: List[torch.Tensor] = []
    gnn_g_all: List[torch.Tensor] = []
    attn_n_all: List[torch.Tensor] = []
    gnn_n_all: List[torch.Tensor] = []
    batch_all: List[torch.Tensor] = []
    edge_all: List[torch.Tensor] = []
    edge_batch_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    split_all: List[torch.Tensor] = []
    have_y = True
    graph_offset = 0
    node_offset = 0

    for name in wanted:
        loader = loader_by_name[name]
        if loader is None:
            logging.warning('Split %s missing; skipping.', name)
            continue
        part = _collect_loader_gates(
            model,
            loader,
            name_to_id[name],
            device,
            want_node=want_node,
        )
        logging.info(
            'Split %s: %d graphs%s, attn_g%s gnn_g%s',
            name,
            int(part['num_graphs']),
            (
                f", {int(part.get('num_nodes', 0))} nodes"
                if want_node
                else ''
            ),
            tuple(part['attn'].shape),
            tuple(part['gnn'].shape),
        )
        attn_g_all.append(part['attn'])
        gnn_g_all.append(part['gnn'])
        split_all.append(part['split'])
        if part['y'] is None:
            have_y = False
        else:
            y_all.append(part['y'])
        if want_node:
            attn_n_all.append(part['attn_node'])
            gnn_n_all.append(part['gnn_node'])
            batch_all.append(part['batch'] + graph_offset)
            n_nodes_part = int(part.get('num_nodes', 0))
            if 'edge_index' in part and part['edge_index'].numel() > 0:
                edge_all.append(part['edge_index'] + int(node_offset))
                edge_batch_all.append(part['edge_batch'] + graph_offset)
            node_offset += n_nodes_part
        graph_offset += int(part['num_graphs'])

    meta: Dict[str, Any] = {
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
        'level': level,
        'aggregation_graph': 'mean_over_nodes',
    }

    y_cat: Optional[torch.Tensor] = (
        torch.cat(y_all, dim=0) if have_y and y_all else None
    )
    split_cat = (
        torch.cat(split_all, dim=0)
        if split_all
        else torch.zeros(0, dtype=torch.long)
    )

    if want_graph:
        graph_payload: Dict[str, Any] = {
            'attn': (
                torch.cat(attn_g_all, dim=0) if attn_g_all else torch.zeros(0, 0, 0)
            ),
            'gnn': (
                torch.cat(gnn_g_all, dim=0) if gnn_g_all else torch.zeros(0, 0, 0)
            ),
            'split': split_cat,
            'y': y_cat,
            'meta': meta,
        }
        out_path = dump_args.out or osp.join(run_dir, 'gate_values_per_graph.pt')
        torch.save(graph_payload, out_path)
        logging.info(
            'Wrote %s (attn%s gnn%s)',
            out_path,
            tuple(graph_payload['attn'].shape),
            tuple(graph_payload['gnn'].shape),
        )

    if want_node:
        if batch_all:
            batch_ids = torch.cat(batch_all, dim=0)
            num_graphs = graph_offset
            ptr = _ptr_from_batch(batch_ids, num_graphs)
            attn_n = torch.cat(attn_n_all, dim=0)
            gnn_n = torch.cat(gnn_n_all, dim=0)
        else:
            batch_ids = torch.zeros(0, dtype=torch.long)
            ptr = torch.zeros(1, dtype=torch.long)
            attn_n = torch.zeros(0, 0, 0)
            gnn_n = torch.zeros(0, 0, 0)
            num_graphs = 0
        if edge_all:
            edge_index = torch.cat(edge_all, dim=1)
            edge_batch = torch.cat(edge_batch_all, dim=0)
            edge_ptr = _ptr_from_batch(edge_batch, num_graphs)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_batch = torch.zeros(0, dtype=torch.long)
            edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long)
        node_payload: Dict[str, Any] = {
            'attn': attn_n,
            'gnn': gnn_n,
            'batch': batch_ids,
            'ptr': ptr,
            'edge_index': edge_index,
            'edge_batch': edge_batch,
            'edge_ptr': edge_ptr,
            'split': split_cat,
            'y': y_cat,
            'num_graphs': num_graphs,
            'num_nodes': int(batch_ids.numel()),
            'num_edges': int(edge_index.size(1)),
            'meta': meta,
        }
        out_node = dump_args.out_node or osp.join(
            run_dir, 'gate_values_per_node.pt'
        )
        torch.save(node_payload, out_node)
        logging.info(
            'Wrote %s (attn%s gnn%s batch%s ptr%s edges=%d)',
            out_node,
            tuple(node_payload['attn'].shape),
            tuple(node_payload['gnn'].shape),
            tuple(node_payload['batch'].shape),
            tuple(node_payload['ptr'].shape),
            int(node_payload['num_edges']),
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
