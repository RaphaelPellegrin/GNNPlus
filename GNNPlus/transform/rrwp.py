"""Relative Random Walk Probabilities (RRWP) for GRIT."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def add_node_attr(data: Data, value: torch.Tensor, attr_name: Optional[str] = None) -> Data:
    """Append or set a node attribute on ``data``."""
    if attr_name is None:
        if hasattr(data, "x") and data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        setattr(data, attr_name, value)
    return data


@torch.no_grad()
def add_full_rrwp(
    data: Data,
    walk_length: int = 21,
    attr_name_abs: str = "rrwp",
    attr_name_rel: str = "rrwp",
    add_identity: bool = True,
) -> Data:
    """Precompute absolute and relative RRWP features on a graph."""
    from torch_sparse import SparseTensor

    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = getattr(data, "edge_weight", None)

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list: list[torch.Tensor] = []
    start = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
        start = 1

    out = adj
    pe_list.append(adj)
    if walk_length > 2:
        for _ in range(start + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)
    abs_pe = pe.diagonal().transpose(0, 1)

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.to(torch.long)
    return data


class AddFullRRWPTransform(BaseTransform):
    """PyG transform wrapper for RRWP precomputation."""

    def __init__(
        self,
        walk_length: int = 21,
        add_identity: bool = True,
    ) -> None:
        self.walk_length = int(walk_length)
        self.add_identity = bool(add_identity)

    def __call__(self, data: Data) -> Data:
        """Apply RRWP encoding."""
        return add_full_rrwp(
            data,
            walk_length=self.walk_length,
            add_identity=self.add_identity,
        )
