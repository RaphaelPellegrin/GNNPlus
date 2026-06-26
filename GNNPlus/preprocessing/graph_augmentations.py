"""Dataset-level graph augmentations (ported from Heterogeneity_Profile)."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data


def parse_cfg_bool(value: Any) -> bool:
    """Parse yacs / W&B sweep values into a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def _zero_pad_node_rows(
    tensor: torch.Tensor,
    n_real: int,
    r: int,
) -> torch.Tensor:
    """Append ``r`` zero rows along dim 0 for per-node tensors."""
    if r <= 0:
        return tensor
    tail_shape = list(tensor.shape[1:])
    zeros = torch.zeros(
        (r, *tail_shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, zeros], dim=0)


def _pad_node_level_attrs(graph: Data, n_real: int, r: int) -> None:
    """Zero-pad node-aligned tensors on ``graph`` after virtual nodes are added."""
    if r <= 0:
        return

    skip = {
        'x',
        'edge_index',
        'edge_attr',
        'y',
        'batch',
        'ptr',
        'num_nodes',
        'num_edges',
    }
    for key, value in list(graph.items()):
        if key in skip or not torch.is_tensor(value):
            continue
        if value.dim() < 1 or int(value.size(0)) != n_real:
            continue
        graph[key] = _zero_pad_node_rows(value, n_real, r)


def add_virtual_nodes(graph: Data, r: int) -> Data:
    """Return a copy of ``graph`` with ``r`` virtual nodes appended.

    Virtual node features are zero-initialized with the same feature dimension as
    ``x``. Each virtual node is connected bidirectionally to every original node.
    If ``edge_attr`` exists, new edge attributes are zero-initialized. Node-aligned
    tensors (e.g. ``pestat_RWSE``) are zero-padded to match the new node count.
    """
    if r <= 0:
        return graph.clone()

    out = graph.clone()
    x = out.x
    if x is None:
        raise ValueError('Graph is missing node features (x); cannot add virtual nodes.')

    n_real = int(x.size(0))
    d = int(x.size(1))
    dev = x.device
    dt = x.dtype

    out.x = torch.cat([x, torch.zeros((r, d), dtype=dt, device=dev)], dim=0)

    if n_real > 0:
        real = torch.arange(n_real, device=dev, dtype=torch.long)
        new_edges: list[torch.Tensor] = []
        for i in range(r):
            v = n_real + i
            v_nodes = torch.full_like(real, fill_value=v)
            new_edges.append(torch.stack([v_nodes, real], dim=0))
            new_edges.append(torch.stack([real, v_nodes], dim=0))
        ve = torch.cat(new_edges, dim=1)
        out.edge_index = torch.cat([out.edge_index, ve], dim=1)

        edge_attr = getattr(out, 'edge_attr', None)
        if edge_attr is not None:
            m = int(ve.size(1))
            if edge_attr.dim() == 1:
                add = torch.zeros(m, dtype=edge_attr.dtype, device=edge_attr.device)
            else:
                add = torch.zeros(
                    (m, int(edge_attr.size(-1))),
                    dtype=edge_attr.dtype,
                    device=edge_attr.device,
                )
            out.edge_attr = torch.cat([edge_attr, add], dim=0)

    _pad_node_level_attrs(out, n_real, r)
    if hasattr(out, 'num_nodes'):
        out.num_nodes = n_real + r

    return out


def maybe_add_virtual_nodes(graph: Data, cfg: Any) -> Data:
    """Apply virtual nodes when enabled in ``cfg.dataset``."""
    if not parse_cfg_bool(getattr(cfg.dataset, 'add_virtual_nodes', False)):
        return graph
    r = int(getattr(cfg.dataset, 'num_virtual_nodes', 0) or 0)
    if r < 0:
        raise ValueError('dataset.num_virtual_nodes must be non-negative')
    if r == 0:
        return graph
    return add_virtual_nodes(graph, r)
