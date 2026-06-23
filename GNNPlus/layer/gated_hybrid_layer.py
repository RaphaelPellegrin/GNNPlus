"""Gated hybrid graph layer for GNNPlus (attention + message-passing heads).

Ported from Heterogeneity_Profile ``graph_moes.architectures.layers.gated_hybrid_layer``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GatedGraphConv,
    GINConv,
    GINEConv,
    ResGatedGraphConv,
    SAGEConv,
)
from torch_geometric.utils import to_undirected

AttnMaskType = Literal['full', 'graph_restricted']
GateMode = Literal['elementwise', 'headwise']
NormType = Literal['layernorm', 'rmsnorm']


def _build_gin_conv(in_dim: int, out_dim: int) -> GINConv:
    """Return a GINConv with a two-layer MLP inside."""
    return GINConv(
        nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    )


from GNNPlus.layer.gcn_conv_layer_e import GCNConvWithEdges
from GNNPlus.layer.gatedgcn_layer import GatedGCNLayer
from torch_geometric.graphgym.config import cfg


class _GCNEHybridMPHead(nn.Module):
    """Edge-aware GCN MP head matching GNN+ ``gcne`` (:class:`GCNConvWithEdges`)."""

    def __init__(
        self,
        d_h: int,
        *,
        edge_dim: int,
        gnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.edge_proj = nn.Linear(edge_dim, d_h)
        self.conv = GCNConvWithEdges(d_h, d_h, edge_dim=d_h, bias=True)
        self._gnn_dropout = float(gnn_dropout)

    def forward(
        self,
        x_h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Run gcne-style message passing at hidden width ``d_h``."""
        num_e = edge_index.size(1)
        dev, dt = x_h.device, x_h.dtype
        if edge_attr is None:
            eh = torch.zeros((num_e, self.d_h), device=dev, dtype=dt)
        else:
            eh = self.edge_proj(edge_attr.float()).to(dtype=dt)
        out = self.conv(x_h, edge_index, eh)
        return F.dropout(out, p=self._gnn_dropout, training=self.training)


class _GINEHybridMPHead(nn.Module):
    """Message-passing head using GINEConv in hidden width ``d_h``."""

    _EDGE_EMB_VOCAB = 16
    _EDGE_LIN_PAD = 8

    def __init__(self, d_h: int, *, gnn_dropout: float = 0.0) -> None:
        super().__init__()
        self.d_h = d_h
        self.edge_emb_1d = nn.Embedding(self._EDGE_EMB_VOCAB, d_h)
        self.edge_lin_pad = nn.Linear(self._EDGE_LIN_PAD, d_h)
        inner = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.BatchNorm1d(d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        self.conv = GINEConv(nn=inner, edge_dim=d_h)
        self._gnn_dropout = float(gnn_dropout)

    def forward(
        self,
        x_h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Run GINE on hidden features; ``edge_attr`` may be 1D/2D or None."""
        num_e = edge_index.size(1)
        dev, dt = x_h.device, x_h.dtype
        if edge_attr is None:
            eh = torch.zeros((num_e, self.d_h), device=dev, dtype=dt)
        elif edge_attr.dim() == 1:
            idx = edge_attr.long().clamp(0, self._EDGE_EMB_VOCAB - 1)
            eh = self.edge_emb_1d(idx).to(dtype=dt)
        else:
            feat = edge_attr.float()
            c = feat.size(-1)
            if c < self._EDGE_LIN_PAD:
                pad = torch.zeros(
                    feat.size(0),
                    self._EDGE_LIN_PAD - c,
                    device=feat.device,
                    dtype=feat.dtype,
                )
                feat = torch.cat([feat, pad], dim=-1)
            elif c > self._EDGE_LIN_PAD:
                feat = feat[:, : self._EDGE_LIN_PAD]
            eh = self.edge_lin_pad(feat).to(dtype=dt)
        out = self.conv(x_h, edge_index, eh)
        return F.dropout(out, p=self._gnn_dropout, training=self.training)


class _ResGatedHybridMPHead(nn.Module):
    """MP head using PyG :class:`ResGatedGraphConv` (legacy ``GATEDGCN`` alias)."""

    def __init__(self, d_h: int, *, gnn_dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ResGatedGraphConv(d_h, d_h, edge_dim=None)
        self._gnn_dropout = float(gnn_dropout)

    def forward(
        self,
        x_h: Tensor,
        edge_index: Tensor,
        _edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        out = self.conv(x_h, edge_index)
        return F.dropout(out, p=self._gnn_dropout, training=self.training)


class _GatedGCNHybridBatch:
    """Minimal batch object for :class:`GatedGCNLayer` inside a hybrid MP head."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    pe_EquivStableLapPE: None

    def __init__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pe_EquivStableLapPE = None


class _GatedGCNHybridMPHead(nn.Module):
    """MP head using GNN+ :class:`GatedGCNLayer` at width ``d_h`` (edge-aware)."""

    def __init__(
        self,
        d_h: int,
        *,
        edge_dim: int,
        dropout: float = 0.15,
        residual: bool = True,
        ffn: bool = True,
        act: str = "relu",
        gnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.edge_proj = nn.Linear(edge_dim, d_h)
        self.layer = GatedGCNLayer(
            d_h,
            d_h,
            dropout=dropout,
            residual=residual,
            ffn=ffn,
            act=act,
        )
        self._gnn_dropout = float(gnn_dropout)

    def forward(
        self,
        x_h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Run GatedGCN+ message passing at ``d_h`` with encoded edge features."""
        num_e = edge_index.size(1)
        dev, dt = x_h.device, x_h.dtype
        if edge_attr is None:
            eh = torch.zeros((num_e, self.d_h), device=dev, dtype=dt)
        else:
            feat = edge_attr.float()
            if feat.size(-1) == self.d_h:
                eh = feat.to(dtype=dt)
            else:
                eh = self.edge_proj(feat).to(dtype=dt)
        batch = _GatedGCNHybridBatch(x_h, edge_index, eh)
        out = self.layer(batch)
        result = out.x
        return F.dropout(result, p=self._gnn_dropout, training=self.training)


class _GatedGraphHybridMPHead(nn.Module):
    """MP head using :class:`GatedGraphConv` (GG-NN style)."""

    def __init__(
        self,
        d_h: int,
        *,
        num_ggnn_internal_layers: int = 1,
        gnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = GatedGraphConv(
            out_channels=d_h,
            num_layers=int(num_ggnn_internal_layers),
            aggr='add',
        )
        self._gnn_dropout = float(gnn_dropout)

    def forward(
        self,
        x_h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        edge_weight: Optional[Tensor] = None
        if edge_attr is not None:
            ea = edge_attr.float()
            edge_weight = ea if ea.dim() == 1 else ea.mean(dim=-1)
        out = self.conv(x_h, edge_index, edge_weight)
        return F.dropout(out, p=self._gnn_dropout, training=self.training)


class _ProjectedMPHead(nn.Module):
    """Projected MP head with shared-source gating (hidden + gate from one linear)."""

    def __init__(
        self,
        kind: str,
        d_model: int,
        d_h: int,
        gate_mode: GateMode,
        *,
        gnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kind = kind.upper()
        self.d_h = d_h
        self.gate_mode = gate_mode
        self._gnn_dropout = float(gnn_dropout)

        gate_out = 1 if gate_mode == 'headwise' else d_h
        self.hg_proj = nn.Linear(d_model, d_h + gate_out)

        if self.kind == 'GCN':
            self.conv = cast(nn.Module, GCNConv(d_h, d_h))
        elif self.kind == 'SAGE':
            self.conv = cast(nn.Module, SAGEConv(d_h, d_h))
        elif self.kind == 'GIN':
            self.conv = cast(nn.Module, _build_gin_conv(d_h, d_h))
        elif self.kind == 'GAT':
            self.conv = cast(
                nn.Module,
                GATConv(
                    d_h,
                    d_h,
                    heads=1,
                    concat=True,
                    dropout=float(gnn_dropout),
                ),
            )
        elif self.kind == 'GINE':
            self.conv = cast(
                nn.Module, _GINEHybridMPHead(d_h, gnn_dropout=gnn_dropout)
            )
        elif self.kind == 'GCNE':
            self.conv = cast(
                nn.Module,
                _GCNEHybridMPHead(
                    d_h, edge_dim=d_model, gnn_dropout=gnn_dropout
                ),
            )
        elif self.kind in ('GGNN', 'GATEDGRAPH', 'GATEDGRAPHCONV'):
            self.conv = cast(
                nn.Module,
                _GatedGraphHybridMPHead(
                    d_h, num_ggnn_internal_layers=1, gnn_dropout=gnn_dropout
                ),
            )
        elif self.kind in ('GATEDGCN',):
            # GatedGCN+ (GNNPlus GatedGCNLayer). Pre-2f8ad6b this string mapped to
            # ResGatedGraphConv; use RESGATEDGCN for that legacy path.
            self.conv = cast(
                nn.Module,
                _GatedGCNHybridMPHead(
                    d_h,
                    edge_dim=d_model,
                    dropout=float(cfg.gnn.dropout),
                    residual=bool(cfg.gnn.residual),
                    ffn=True,
                    act=str(cfg.gnn.act),
                    gnn_dropout=gnn_dropout,
                ),
            )
        elif self.kind in ('RESGATEDGCN',):
            self.conv = cast(
                nn.Module,
                _ResGatedHybridMPHead(d_h, gnn_dropout=gnn_dropout),
            )
        else:
            raise ValueError(
                f'Unknown MP head type: {kind!r} '
                '(expected GCN, GCNE, GIN, GINE, GGNN, GATEDGCN, RESGATEDGCN, SAGE, or GAT)'
            )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return ``(gated_mp_output, gate_value)``."""
        hg = self.hg_proj(x)
        if self.gate_mode == 'headwise':
            h, g = torch.split(hg, [self.d_h, 1], dim=-1)
        else:
            h, g = torch.split(hg, [self.d_h, self.d_h], dim=-1)

        if isinstance(
            self.conv,
            (
                _GCNEHybridMPHead,
                _GatedGCNHybridMPHead,
                _GINEHybridMPHead,
                _GatedGraphHybridMPHead,
                _ResGatedHybridMPHead,
            ),
        ):
            raw = self.conv(h, edge_index, edge_attr)
        else:
            raw = self.conv(h, edge_index)

        gamma = torch.sigmoid(g)
        return raw * gamma, gamma


def _make_mp_head(
    kind: str,
    d_model: int,
    d_h: int,
    gate_mode: GateMode,
    *,
    gnn_dropout: float = 0.0,
) -> nn.Module:
    """Instantiate a projected MP head."""
    return _ProjectedMPHead(
        kind=kind,
        d_model=d_model,
        d_h=d_h,
        gate_mode=gate_mode,
        gnn_dropout=gnn_dropout,
    )


def parse_hybrid_gnn_types(raw: Optional[str], num_heads: int) -> List[str]:
    """Build a list of length ``num_heads`` of conv type names."""
    if num_heads <= 0:
        return []
    base = ['GCN', 'GIN', 'SAGE', 'GAT']
    if raw is None or str(raw).strip() == '':
        return [base[i % len(base)] for i in range(num_heads)]
    parts = [p.strip().upper() for p in str(raw).split(',') if p.strip()]
    if len(parts) == 0:
        return [base[i % len(base)] for i in range(num_heads)]
    if len(parts) == 1:
        return [parts[0]] * num_heads
    if len(parts) < num_heads:
        for i in range(num_heads - len(parts)):
            parts.append(base[i % len(base)])
    else:
        parts = parts[:num_heads]
    return parts


class RMSNorm(nn.Module):
    """Root mean square normalization (per row)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        return (x_f * self.weight.float()).to(dtype)


def _same_graph_mask(batch: Tensor, n: int) -> Tensor:
    """Boolean mask ``[n, n]`` — True iff nodes share the same graph id."""
    return batch.view(-1, 1).eq(batch.view(1, -1))


def _graph_adjacency_mask(
    edge_index: Tensor,
    batch: Tensor,
    num_nodes: int,
) -> Tensor:
    """Within-graph adjacency including self-loops (undirected)."""
    ei = to_undirected(edge_index.clone())
    row, col = ei[0], ei[1]
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=edge_index.device)
    adj[row, col] = True
    adj.fill_diagonal_(True)
    same = _same_graph_mask(batch, num_nodes)
    return adj & same


class GatedHybridGraphLayer(nn.Module):
    """Hybrid block: gated attention heads + gated message-passing heads + residual.

    See Heterogeneity_Profile ``gated_hybrid_layer.py`` for design notes.
    """

    def __init__(
        self,
        d_model: int,
        num_attn_heads: int,
        num_gnn_heads: int,
        d_h: int,
        attn_mask_type: AttnMaskType = 'full',
        gate_mode: GateMode = 'elementwise',
        norm_type: NormType = 'layernorm',
        gnn_types: Optional[List[str]] = None,
        attn_dropout: float = 0.0,
        mp_gnn_dropout: float = 0.0,
        block_bn: bool = False,
        block_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_attn_heads < 0 or num_gnn_heads < 0:
            raise ValueError('num_attn_heads and num_gnn_heads must be non-negative')
        if num_attn_heads + num_gnn_heads < 1:
            raise ValueError('Need at least one head total')
        if d_h < 1:
            raise ValueError('d_h must be positive')

        self.d_model = d_model
        self.num_attn_heads = num_attn_heads
        self.num_gnn_heads = num_gnn_heads
        self.d_h = d_h
        self.attn_mask_type: AttnMaskType = attn_mask_type
        self.gate_mode: GateMode = gate_mode
        self.attn_dropout = float(attn_dropout)
        self._mp_gnn_dropout = float(mp_gnn_dropout)
        self.block_bn = bool(block_bn)
        self._block_dropout = float(block_dropout)

        self.norm: nn.Module
        if self.block_bn:
            self.norm = nn.Identity()
        elif norm_type == 'layernorm':
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
        else:
            raise ValueError(f'Unknown norm_type: {norm_type!r}')

        self.qg_linears = nn.ModuleList()
        self.k_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        for _ in range(num_attn_heads):
            qg_out = d_h + (1 if gate_mode == 'headwise' else d_h)
            self.qg_linears.append(nn.Linear(d_model, qg_out))
            self.k_linears.append(nn.Linear(d_model, d_h))
            self.v_linears.append(nn.Linear(d_model, d_h))

        types = gnn_types or parse_hybrid_gnn_types(None, num_gnn_heads)
        if len(types) != num_gnn_heads:
            raise ValueError(
                f'gnn_types length {len(types)} != num_gnn_heads {num_gnn_heads}'
            )

        self.mp_heads = nn.ModuleList([
            _make_mp_head(
                t,
                d_model,
                d_h,
                gate_mode,
                gnn_dropout=self._mp_gnn_dropout,
            )
            for t in types
        ])

        total_heads = num_attn_heads + num_gnn_heads
        self.out_proj: nn.Module
        self.out_proj_attn: Optional[nn.Linear]
        self.out_proj_mp: Optional[nn.Linear]
        self.norm1_attn: Optional[nn.BatchNorm1d]
        self.norm1_local: Optional[nn.BatchNorm1d]
        self.dropout_attn: Optional[nn.Dropout]
        self.dropout_local: Optional[nn.Dropout]
        if self.block_bn:
            self.out_proj = nn.Identity()
            self.out_proj_attn = (
                nn.Linear(num_attn_heads * d_h, d_model)
                if num_attn_heads > 0 else None
            )
            self.out_proj_mp = (
                nn.Linear(num_gnn_heads * d_h, d_model)
                if num_gnn_heads > 0 else None
            )
            self.norm1_attn = (
                nn.BatchNorm1d(d_model) if num_attn_heads > 0 else None
            )
            self.norm1_local = (
                nn.BatchNorm1d(d_model) if num_gnn_heads > 0 else None
            )
            self.dropout_attn = (
                nn.Dropout(self._block_dropout) if num_attn_heads > 0 else None
            )
            self.dropout_local = (
                nn.Dropout(self._block_dropout) if num_gnn_heads > 0 else None
            )
        else:
            self.out_proj = nn.Linear(total_heads * d_h, d_model)
            self.out_proj_attn = None
            self.out_proj_mp = None
            self.norm1_attn = None
            self.norm1_local = None
            self.dropout_attn = None
            self.dropout_local = None
        self._scale = 1.0 / math.sqrt(float(d_h))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        attn_source: Optional[Tensor] = None,
        mp_source: Optional[Tensor] = None,
        return_gate_stats: bool = False,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Apply the hybrid block."""
        out, aux = self._forward_core(
            x,
            edge_index,
            batch,
            edge_attr=edge_attr,
            attn_source=attn_source,
            mp_source=mp_source,
            return_gate_stats=return_gate_stats,
            return_attn_weights=return_attn_weights,
        )
        if not return_gate_stats and not return_attn_weights:
            return out
        return out, aux

    def _forward_core(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        attn_source: Optional[Tensor],
        mp_source: Optional[Tensor],
        return_gate_stats: bool,
        return_attn_weights: bool,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        n = x.size(0)
        src: Tensor = x if self.block_bn else self.norm(x)
        src_attn: Tensor = src if attn_source is None else attn_source
        src_mp: Tensor = src if mp_source is None else mp_source

        if self.attn_mask_type == 'graph_restricted':
            allowed = _graph_adjacency_mask(edge_index, batch, n)
        else:
            allowed = _same_graph_mask(batch, n)

        neg_inf = torch.finfo(x.dtype).min / 4

        attn_outputs: List[Tensor] = []
        attn_gate_vals: List[Tensor] = []
        attn_weights_list: List[Tensor] = []

        for m in range(self.num_attn_heads):
            qg = self.qg_linears[m](src_attn)
            if self.gate_mode == 'headwise':
                q, g = torch.split(qg, [self.d_h, 1], dim=-1)
            else:
                q, g = torch.split(qg, [self.d_h, self.d_h], dim=-1)

            k = self.k_linears[m](src_attn)
            v = self.v_linears[m](src_attn)

            scores = (q @ k.transpose(0, 1)) * self._scale
            scores = scores.masked_fill(~allowed, neg_inf)
            weights = F.softmax(scores, dim=-1)
            if return_attn_weights:
                attn_weights_list.append(weights.detach().clone())
            weights = F.dropout(weights, p=self.attn_dropout, training=self.training)

            raw = weights @ v
            gamma = torch.sigmoid(g)
            attn_outputs.append(raw * gamma)
            attn_gate_vals.append(gamma)

        mp_outputs: List[Tensor] = []
        mp_gate_vals: List[Tensor] = []

        for mp_head in self.mp_heads:
            out_h, gamma = cast(
                Tuple[Tensor, Tensor], mp_head(src_mp, edge_index, edge_attr)
            )
            mp_outputs.append(out_h)
            mp_gate_vals.append(gamma)

        gate_stats: Dict[str, float] = {}
        if return_gate_stats:
            for m, gamma in enumerate(attn_gate_vals):
                gate_stats[f'attn_{m}_gate_mean'] = gamma.detach().mean().item()
            for m, gamma in enumerate(mp_gate_vals):
                gate_stats[f'gnn_{m}_gate_mean'] = gamma.detach().mean().item()

        if self.block_bn:
            branch_outs: List[Tensor] = []
            if self.num_attn_heads > 0:
                assert self.out_proj_attn is not None
                assert self.dropout_attn is not None
                assert self.norm1_attn is not None
                h_attn = self.out_proj_attn(torch.cat(attn_outputs, dim=-1))
                h_attn = self.dropout_attn(h_attn)
                h_attn = x + h_attn
                h_attn = self.norm1_attn(h_attn)
                branch_outs.append(h_attn)
            if self.num_gnn_heads > 0:
                assert self.out_proj_mp is not None
                assert self.dropout_local is not None
                assert self.norm1_local is not None
                h_mp = self.out_proj_mp(torch.cat(mp_outputs, dim=-1))
                h_mp = self.dropout_local(h_mp)
                h_mp = x + h_mp
                h_mp = self.norm1_local(h_mp)
                branch_outs.append(h_mp)
            if len(branch_outs) == 1:
                out = branch_outs[0]
            else:
                out = branch_outs[0]
                for extra in branch_outs[1:]:
                    out = out + extra
        else:
            fused_inputs = attn_outputs + mp_outputs
            if len(fused_inputs) == 0:
                raise RuntimeError('gated hybrid layer has no head outputs')
            fused = cast(Tensor, self.out_proj(torch.cat(fused_inputs, dim=-1)))
            out = x + fused

        aux: Dict[str, Any] = {}
        if return_gate_stats:
            aux['gate_stats'] = gate_stats
        if return_attn_weights:
            aux['attn_weights'] = attn_weights_list

        return out, aux
