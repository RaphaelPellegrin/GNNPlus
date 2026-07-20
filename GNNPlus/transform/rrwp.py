"""Relative Random Walk Probabilities (RRWP) for GRIT."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj

logger = logging.getLogger(__name__)

# PATTERN (~118 nodes) and CLUSTER graphs are small; dense RRWP is fast and does
# not need the ``torch_sparse`` C++ extension (often broken on older cluster GLIBC).
_SMALL_GRAPH_DENSE_THRESHOLD = 512
_TORCH_SPARSE_EXT_LOGGED = False


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


def _normalized_adjacency_dense(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Row-normalized adjacency as a dense ``(N, N)`` tensor."""
    adj = to_dense_adj(
        edge_index,
        edge_attr=edge_weight,
        max_num_nodes=num_nodes,
    ).squeeze(0)
    adj = adj.to(device=device, dtype=torch.float)
    deg = adj.sum(dim=1)
    deg_inv = deg.pow(-1.0)
    deg_inv[deg_inv == float("inf")] = 0
    return adj * deg_inv.unsqueeze(-1)


def _dense_walk_powers(
    adj: torch.Tensor,
    walk_length: int,
    add_identity: bool,
    device: torch.device,
    num_nodes: int,
) -> torch.Tensor:
    """Return stacked walk powers ``(N, N, L)`` via dense matmul."""
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

    return torch.stack(pe_list, dim=-1)


def _relative_rrwp_from_dense_pe(pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RRWP COO index/value tensors from dense ``(N, N, L)`` powers."""
    num_nodes, _, walk_dim = pe.shape
    device = pe.device
    rows = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    cols = torch.arange(num_nodes, device=device).repeat(num_nodes)
    rel_pe_idx = torch.stack([cols, rows], dim=0)
    rel_pe_val = pe.permute(1, 0, 2).reshape(num_nodes * num_nodes, walk_dim)
    return rel_pe_idx, rel_pe_val


def _relative_rrwp_from_sparse_tensor(pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RRWP COO tensors via ``torch_sparse`` (large graphs only)."""
    from torch_sparse import SparseTensor

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)
    return rel_pe_idx, rel_pe_val


def _log_sparse_backend_once(message: str, *args: object) -> None:
    """Emit at most one RRWP backend notice per process."""
    global _TORCH_SPARSE_EXT_LOGGED
    if not _TORCH_SPARSE_EXT_LOGGED:
        logger.info(message, *args)
        _TORCH_SPARSE_EXT_LOGGED = True


def _torch_sparse_extension_available() -> bool:
    """Return whether the ``torch_sparse`` C++ extension loads."""
    try:
        from torch_sparse import SparseTensor  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def _normalized_adjacency_torch_sparse(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-normalized adjacency via ``torch_sparse`` (large graphs)."""
    from torch_sparse import SparseTensor

    adj_sparse = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )
    deg = adj_sparse.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = (adj_sparse * deg_inv.view(-1, 1)).to_dense()
    return adj, deg


@torch.no_grad()
def add_full_rrwp(
    data: Data,
    walk_length: int = 21,
    attr_name_abs: str = "rrwp",
    attr_name_rel: str = "rrwp",
    add_identity: bool = True,
) -> Data:
    """Precompute absolute and relative RRWP features on a graph."""
    device = data.edge_index.device
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index
    edge_weight = getattr(data, "edge_weight", None)

    adj: torch.Tensor
    deg: torch.Tensor
    use_sparse_rel = False

    if num_nodes <= _SMALL_GRAPH_DENSE_THRESHOLD:
        adj = _normalized_adjacency_dense(edge_index, edge_weight, num_nodes, device)
        deg = adj.sum(dim=1)
    elif _torch_sparse_extension_available():
        adj, deg = _normalized_adjacency_torch_sparse(
            edge_index, edge_weight, num_nodes
        )
        use_sparse_rel = True
    else:
        _log_sparse_backend_once(
            "torch_sparse extension unavailable on this node; using dense RRWP "
            "(expected for PATTERN/CLUSTER; rebuild env on GPU node for huge graphs)."
        )
        adj = _normalized_adjacency_dense(edge_index, edge_weight, num_nodes, device)
        deg = adj.sum(dim=1)

    pe = _dense_walk_powers(adj, walk_length, add_identity, device, num_nodes)
    abs_pe = pe.diagonal().transpose(0, 1)

    if use_sparse_rel:
        rel_pe_idx, rel_pe_val = _relative_rrwp_from_sparse_tensor(pe)
    else:
        rel_pe_idx, rel_pe_val = _relative_rrwp_from_dense_pe(pe)

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
