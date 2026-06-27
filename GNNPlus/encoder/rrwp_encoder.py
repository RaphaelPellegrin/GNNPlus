"""RRWP encoders for standalone GRIT models."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torch_geometric.graphgym.register import register_edge_encoder, register_node_encoder
from torch_geometric.utils import add_self_loops, scatter


def full_edge_index(edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
    """Return edge indices for the complement of ``edge_index`` within each graph."""
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = int(batch.max().item()) + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list: list[torch.Tensor] = []
    for i in range(batch_size):
        n = int(num_nodes[i].item())
        adj = torch.ones((n, n), dtype=torch.short, device=edge_index.device)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        negative_index_list.append(_edge_index + cum_nodes[i])

    return torch.cat(negative_index_list, dim=1).contiguous()


@register_node_encoder("rrwp_linear")
class RRWPLinearNodeEncoder(nn.Module):
    """Linear encoder for absolute RRWP node features."""

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        use_bias: bool = False,
        batchnorm: bool = False,
        layernorm: bool = False,
        pe_name: str = "rrwp",
    ) -> None:
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.fc.weight)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity()
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()

    def forward(self, batch: object) -> object:
        """Encode RRWP node features into ``batch.x``."""
        rrwp = getattr(batch, f"{self.name}")
        rrwp = self.fc(rrwp)
        rrwp = self.bn(rrwp)
        rrwp = self.ln(rrwp)
        if hasattr(batch, "x") and getattr(batch, "x") is not None:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp
        return batch


@register_edge_encoder("rrwp_linear")
class RRWPLinearEdgeEncoder(nn.Module):
    """Merge RRWP relative encodings with edge features (full-graph padding)."""

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        batchnorm: bool = False,
        layernorm: bool = False,
        use_bias: bool = False,
        pad_to_full_graph: bool = True,
        fill_value: float = 0.0,
        add_node_attr_as_self_loop: bool = False,
        overwrite_old_attr: bool = False,
    ) -> None:
        super().__init__()
        del add_node_attr_as_self_loop
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.overwrite_old_attr = overwrite_old_attr
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm on RRWP edges may affect shortest-path signal."
            )
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity()
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()

    def forward(self, batch: object) -> object:
        """Build full-graph edge attributes for GRIT attention."""
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            import torch_sparse

            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                num_nodes=batch.num_nodes,
                fill_value=0.0,
            )
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if self.pad_to_full_graph:
            import torch_sparse

            edge_index_full = full_edge_index(out_idx, batch=getattr(batch, "batch", None))
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx,
                out_val,
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        out_val = self.bn(out_val)
        out_val = self.ln(out_val)
        batch.edge_index = out_idx
        batch.edge_attr = out_val
        return batch
