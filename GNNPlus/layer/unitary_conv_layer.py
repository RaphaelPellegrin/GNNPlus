"""Unitary (complex-valued Taylor GCN) convolution layers for GNNPlus.

Adapted from Weber-GeoML Unitary_Convolutions:
https://github.com/Weber-GeoML/Unitary_Convolutions
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric.graphgym.register as register
from torch import Tensor
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops as add_self_loops_fn,
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

UnitaryBaseConv = Union["ComplexGCNConv", "HermitianGCNConv"]


class ComplexDropout(nn.Module):
    """Dropout that masks only the real part when the input is complex."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        if torch.is_complex(x):
            mask = F.dropout(
                torch.ones_like(x.real),
                p=self.dropout,
                training=self.training,
            )
            return x * mask
        return F.dropout(x, p=self.dropout, training=self.training)


def block_diagonal_complex_init(
    weight_matrix: Tensor,
    block_size: int = 2,
    bound: float = 0.5,
) -> Tensor:
    """Initialize a block-diagonal complex weight matrix (Hermitian path)."""
    n = weight_matrix.size(0)
    out = torch.zeros_like(weight_matrix)
    num_blocks = (n + block_size - 1) // block_size

    for i in range(num_blocks):
        actual_block_size = min(block_size, n - i * block_size)
        real_part = torch.randn(actual_block_size, actual_block_size) * bound
        imag_part = torch.randn(actual_block_size, actual_block_size) * bound
        block = torch.view_as_complex(torch.stack([real_part, imag_part], dim=-1))
        start_row = i * block_size
        end_row = start_row + actual_block_size
        out[start_row:end_row, start_row:end_row] = block

    return out


class ComplexGCNConv(MessagePassing):
    """Complex-valued GCN convolution with orthogonal real weights."""

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = True,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support adding self-loops "
                "when on-the-fly normalization is disabled"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        init.orthogonal_(self.lin.weight.data)
        self.lin.weight = nn.Parameter(
            torch.complex(self.lin.weight, torch.zeros_like(self.lin.weight))
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        w = torch.empty(self.out_channels, self.in_channels)
        init.orthogonal_(w)
        self.lin.weight = nn.Parameter(torch.complex(w, torch.zeros_like(w)))
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        apply_feature_lin: bool = True,
        return_feature_only: bool = False,
    ) -> Tensor:
        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' does not support bipartite message passing"
            )

        if apply_feature_lin:
            if not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            x = self.lin(x)
            if self.bias is not None:
                x = x + self.bias
            if return_feature_only:
                return x

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = 1j * self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class HermitianGCNConv(MessagePassing):
    """Hermitian-parameterized complex GCN convolution."""

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = False,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support adding self-loops "
                "when on-the-fly normalization is disabled"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.lin.weight = nn.Parameter(
            torch.complex(torch.zeros_like(self.lin.weight), self.lin.weight)
        )
        setattr(
            self.lin.weight,
            "complex_hermitian_params",
            in_channels * in_channels // 2,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.lin.weight.data = block_diagonal_complex_init(self.lin.weight.data)
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        apply_feature_lin: bool = True,
        return_feature_only: bool = False,
    ) -> Tensor:
        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' does not support bipartite message passing"
            )

        if apply_feature_lin:
            if not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            if return_feature_only:
                return x

        x = (x @ self.lin.weight + x @ self.lin.weight.conj().T) / 2
        if self.bias is not None:
            x = x + self.bias

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = 1j * self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class TaylorGCNConv(nn.Module):
    """Taylor expansion of a complex GCN operator."""

    def __init__(
        self,
        conv: UnitaryBaseConv,
        T: int = 16,
        return_real: bool = False,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.T = T
        self.return_real = return_real

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        x = self.conv(
            x,
            edge_index,
            edge_weight,
            apply_feature_lin=True,
            return_feature_only=True,
        )
        x_k = x.clone()

        for k in range(self.T):
            x_k = (
                self.conv(
                    x_k,
                    edge_index,
                    edge_weight,
                    apply_feature_lin=False,
                )
                / (k + 1)
            )
            x = x + x_k

        if self.return_real:
            x = x.real
        return x


def build_unitary_taylor_conv(
    in_channels: int,
    out_channels: int,
    *,
    use_hermitian: bool = False,
    taylor_order: int = 16,
    return_real: bool = True,
    conv_bias: bool = False,
    **kwargs: object,
) -> TaylorGCNConv:
    """Build a Taylor-expanded unitary GCN operator."""
    base_cls = HermitianGCNConv if use_hermitian else ComplexGCNConv
    base = base_cls(
        in_channels,
        out_channels,
        bias=conv_bias,
        **kwargs,
    )
    return TaylorGCNConv(base, T=taylor_order, return_real=return_real)


@register_layer("unitarygcnconv")
class UnitaryGCNConvLayer(nn.Module):
    """GraphGym batch layer wrapping Taylor unitary GCN (UniGCN)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dropout: float,
        residual: bool,
        global_bias: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        kwargs.pop("ffn", None)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.return_real = bool(cfg.gnn.unitary_return_real)

        if global_bias:
            if self.return_real:
                self.bias: Optional[nn.Parameter] = nn.Parameter(
                    torch.zeros(dim_out)
                )
            else:
                self.bias = nn.Parameter(torch.zeros(dim_out, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)

        use_hermitian = bool(cfg.gnn.use_hermitian)
        taylor_order = int(cfg.gnn.unitary_taylor_order)
        self.model = build_unitary_taylor_conv(
            dim_in,
            dim_out,
            use_hermitian=use_hermitian,
            taylor_order=taylor_order,
            return_real=self.return_real,
            **kwargs,
        )

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            ComplexDropout(self.dropout),
        )

    def forward(self, batch: object) -> object:
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        if self.bias is not None:
            batch.x = batch.x + self.bias
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x

        return batch


@torch.jit._overload
def gcn_norm(
    edge_index: Tensor,
    edge_weight: OptTensor,
    num_nodes: Optional[int],
    improved: bool,
    add_self_loops: bool,
    flow: str,
    dtype: Optional[torch.dtype],
) -> OptPairTensor:
    pass


@torch.jit._overload
def gcn_norm(
    edge_index: SparseTensor,
    edge_weight: OptTensor,
    num_nodes: Optional[int],
    improved: bool,
    add_self_loops: bool,
    flow: str,
    dtype: Optional[torch.dtype],
) -> SparseTensor:
    pass


def gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
) -> Union[OptPairTensor, SparseTensor]:
    """Symmetric GCN normalization (supports complex dtypes)."""
    fill_value = 2.0 if improved else 1.0

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError(
                "Sparse CSC matrices are not yet supported in 'gcn_norm'"
            )

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),),
            dtype=dtype,
            device=edge_index.device,
        )

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight
