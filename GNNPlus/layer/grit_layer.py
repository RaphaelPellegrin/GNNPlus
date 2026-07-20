"""GRIT transformer layer (Ma et al., ICML 2023).

Adapted from https://github.com/LiamMa/GRIT (MIT license).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_layer
from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from yacs.config import CfgNode as CN

try:
    import opt_einsum as oe
except ImportError:  # pragma: no cover - optional dependency
    oe = None  # type: ignore[assignment]


def _contract_einsum(
    equation: str,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> torch.Tensor:
    """Einstein contraction with ``opt_einsum`` when available."""
    if oe is not None:
        return oe.contract(equation, tensor_a, tensor_b, backend="torch")
    return torch.einsum(equation, tensor_a, tensor_b)


def pyg_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    """Sparse softmax grouped by ``index`` (PyG-style)."""
    num_nodes = maybe_num_nodes(index, num_nodes)
    max_val = scatter(src, index, dim=0, dim_size=num_nodes, reduce="max")
    out = src - max_val[index]
    out = out.exp()
    out = out / (scatter(out, index, dim=0, dim_size=num_nodes, reduce="sum")[index] + 1e-16)
    return out


def grit_paper_defaults() -> CN:
    """Return GRIT hyperparameters matching the official PATTERN config."""
    cfg = CN(new_allowed=True)
    cfg.update_e = True
    cfg.bn_momentum = 0.1
    cfg.bn_no_runner = False
    cfg.rezero = False
    cfg.attn = CN(new_allowed=True)
    cfg.attn.use = True
    cfg.attn.use_bias = False
    cfg.attn.clamp = 5.0
    cfg.attn.act = "relu"
    cfg.attn.edge_enhance = True
    cfg.attn.sqrt_relu = False
    cfg.attn.signed_sqrt = False
    cfg.attn.scaled_attn = False
    cfg.attn.no_qk = False
    cfg.attn.graphormer_attn = False
    cfg.attn.deg_scaler = True
    return cfg


def resolve_grit_num_heads(dim: int, requested: int) -> int:
    """Pick a valid head count dividing ``dim`` (prefer ``requested``)."""
    if requested < 1:
        raise ValueError(f"GRIT n_heads must be positive, got {requested}")
    if dim % requested == 0:
        return requested
    for h in range(min(requested, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


def build_grit_layer_cfg(
    *,
    dropout: float = 0.0,
    attn_dropout: float = 0.2,
    layer_norm: bool = False,
    batch_norm: bool = True,
    residual: bool = True,
    norm_e: bool = True,
    update_e: bool = True,
    rezero: bool = False,
) -> CN:
    """Build a GRIT layer config node with paper-style defaults."""
    cfg = grit_paper_defaults()
    cfg.dropout = float(dropout)
    cfg.attn_dropout = float(attn_dropout)
    cfg.layer_norm = bool(layer_norm)
    cfg.batch_norm = bool(batch_norm)
    cfg.residual = bool(residual)
    cfg.norm_e = bool(norm_e)
    cfg.update_e = bool(update_e)
    cfg.rezero = bool(rezero)
    return cfg


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """Sparse attention used by GRIT (edge-centric inductive bias)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        use_bias: bool,
        clamp: float = 5.0,
        dropout: float = 0.0,
        act: Optional[str] = "relu",
        edge_enhance: bool = True,
        cfg: Optional[CN] = None,
    ) -> None:
        super().__init__()
        del cfg
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act: nn.Module = nn.Identity()
        else:
            self.act = register.act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch: Any) -> None:
        """Compute sparse attention messages on ``batch.edge_index``."""
        src = batch.K_h[batch.edge_index[0]]
        dest = batch.Q_h[batch.edge_index[1]]
        score = src + dest

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            e_w, e_b = batch.E[:, :, : self.out_dim], batch.E[:, :, self.out_dim :]
            score = score * e_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + e_b

        score = self.act(score)
        e_t = score

        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        score = _contract_einsum("ehd, dhc->ehc", score, self.Aw)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])
        score = self.dropout(score)
        batch.attn = score

        msg = batch.V_h[batch.edge_index[0]] * score
        batch.wV = scatter(
            msg,
            batch.edge_index[1],
            dim=0,
            dim_size=batch.V_h.size(0),
            reduce="sum",
        )

        if self.edge_enhance and batch.E is not None:
            row_v = scatter(
                e_t * score,
                batch.edge_index[1],
                dim=0,
                dim_size=batch.V_h.size(0),
                reduce="sum",
            )
            row_v = _contract_einsum("nhd, dhc -> nhc", row_v, self.VeRow)
            batch.wV = batch.wV + row_v

    def forward(self, batch: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return node and optional edge attention outputs."""
        q_h = self.Q(batch.x)
        k_h = self.K(batch.x)
        v_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = k_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = v_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get("wE", None)
        return h_out, e_out


@torch.no_grad()
def get_log_deg(batch: Any) -> torch.Tensor:
    """Log-degree features for GRIT degree scaler."""
    if hasattr(batch, "log_deg") and batch.log_deg is not None:
        return batch.log_deg
    if hasattr(batch, "deg") and batch.deg is not None:
        deg = batch.deg
        return torch.log(deg + 1).unsqueeze(-1)
    warnings.warn(
        "Computing GRIT log-degree on the fly; precompute via RRWP transform when possible."
    )
    deg = pyg.utils.degree(
        batch.edge_index[1],
        num_nodes=batch.num_nodes,
        dtype=torch.float,
    )
    return torch.log(deg + 1).view(batch.num_nodes, 1)


class GritTransformerLayer(nn.Module):
    """One GRIT layer (attention + FFN + optional edge update)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        residual: bool = True,
        act: str = "relu",
        norm_e: bool = True,
        update_e: bool = True,
        cfg: Optional[CN] = None,
    ) -> None:
        super().__init__()
        layer_cfg = cfg if cfg is not None else build_grit_layer_cfg()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.layer_norm = bool(layer_norm)
        self.batch_norm = bool(batch_norm)
        self.update_e = bool(update_e)
        self.bn_momentum = float(layer_cfg.bn_momentum)
        self.bn_no_runner = bool(layer_cfg.bn_no_runner)
        self.rezero = bool(layer_cfg.rezero)
        self.deg_scaler = bool(layer_cfg.attn.deg_scaler)
        self.norm_e = bool(norm_e)

        self.act = register.act_dict[act]() if act is not None else nn.Identity()
        attn_cfg = layer_cfg.attn

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=bool(attn_cfg.use_bias),
            dropout=attn_dropout,
            clamp=float(attn_cfg.clamp),
            act=str(attn_cfg.act),
            edge_enhance=bool(attn_cfg.edge_enhance),
        )

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        self.layer_norm1_h = nn.LayerNorm(out_dim) if self.layer_norm else nn.Identity()
        self.layer_norm1_e = nn.LayerNorm(out_dim) if self.layer_norm and self.norm_e else nn.Identity()
        self.batch_norm1_h = nn.BatchNorm1d(
            out_dim,
            track_running_stats=not self.bn_no_runner,
            eps=1e-5,
            momentum=self.bn_momentum,
        ) if self.batch_norm else nn.Identity()
        self.batch_norm1_e = nn.BatchNorm1d(
            out_dim,
            track_running_stats=not self.bn_no_runner,
            eps=1e-5,
            momentum=self.bn_momentum,
        ) if self.batch_norm and self.norm_e else nn.Identity()

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.layer_norm2_h = nn.LayerNorm(out_dim) if self.layer_norm else nn.Identity()
        self.batch_norm2_h = nn.BatchNorm1d(
            out_dim,
            track_running_stats=not self.bn_no_runner,
            eps=1e-5,
            momentum=self.bn_momentum,
        ) if self.batch_norm else nn.Identity()

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch: Any) -> Any:
        """Update ``batch.x`` (and optionally ``batch.edge_attr``)."""
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h
        e_in1 = batch.get("edge_attr", None)

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        e: Optional[torch.Tensor] = None
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h
            if e is not None and e_in1 is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        h = self.layer_norm1_h(h)
        if e is not None:
            e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h

        h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1
        return batch


@register_layer("GritTransformer")
class GritTransformerLayerRegistered(GritTransformerLayer):
    """GraphGym-registered GRIT layer (used by :class:`GritTransformer` network)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        residual: bool = True,
        act: str = "relu",
        norm_e: bool = True,
        O_e: bool = True,
        cfg: Optional[CN] = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        layer_cfg = cfg if cfg is not None else build_grit_layer_cfg()
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            residual=residual,
            act=act,
            norm_e=norm_e,
            update_e=bool(layer_cfg.update_e),
            cfg=layer_cfg,
        )
        if not O_e:
            self.O_e = nn.Identity()
