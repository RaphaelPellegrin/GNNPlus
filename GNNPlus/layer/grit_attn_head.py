"""GRIT sparse attention as a SiGMA hybrid attention head."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from GNNPlus.layer.grit_hybrid_mp_head import _GritHybridBatch
from GNNPlus.layer.grit_layer import MultiHeadAttentionLayerGritSparse

GateMode = Literal["elementwise", "headwise", "none", "off"]


def _normalize_gate_mode(gate_mode: str) -> GateMode:
    """Map config aliases to a canonical gate mode."""
    mode = str(gate_mode).strip().lower()
    if mode in ("none", "off", "disabled", "ungated", "identity"):
        return "none"
    if mode in ("elementwise", "headwise"):
        return cast(GateMode, mode)
    raise ValueError(
        f"Unknown gate mode: {gate_mode!r} "
        "(expected elementwise, headwise, or none)"
    )


class _GRITAttnHead(nn.Module):
    """One GRIT sparse attention unit at head width ``d_h``, with SiGMA gating.

    Uses :class:`MultiHeadAttentionLayerGritSparse` with ``num_heads=1`` and
    ``out_dim=d_h``. Edge features are projected to ``d_model`` so they match
    the GRIT ``E`` linear (``in_dim=d_model``).
    """

    def __init__(
        self,
        d_model: int,
        d_h: int,
        gate_mode: str = "elementwise",
        *,
        edge_dim: int,
        attn_dropout: float = 0.0,
        clamp: float = 5.0,
        edge_enhance: bool = True,
        act: str = "relu",
        use_bias: bool = False,
    ) -> None:
        """Build a gated GRIT attention head.

        Args:
            d_model: Node feature width entering the head.
            d_h: Per-head output width.
            gate_mode: SiGMA gate style (``elementwise`` / ``headwise`` / ``none``).
            edge_dim: Incoming edge-feature width before projection.
            attn_dropout: Dropout inside GRIT attention.
            clamp: Score clamp magnitude (GRIT default 5.0).
            edge_enhance: Whether to use GRIT edge enhancement.
            act: Activation name registered in GraphGym.
            use_bias: Bias on GRIT ``K`` / ``V`` linears.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.d_h = int(d_h)
        self.gate_mode: GateMode = _normalize_gate_mode(gate_mode)
        self.edge_dim = int(edge_dim)

        self.edge_proj: nn.Module
        if self.edge_dim == self.d_model:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(self.edge_dim, self.d_model)

        self.attn = MultiHeadAttentionLayerGritSparse(
            in_dim=self.d_model,
            out_dim=self.d_h,
            num_heads=1,
            use_bias=bool(use_bias),
            clamp=float(clamp),
            dropout=float(attn_dropout),
            act=str(act),
            edge_enhance=bool(edge_enhance),
        )

        if self.gate_mode == "none":
            self.gate_proj: Optional[nn.Linear] = None
        elif self.gate_mode == "headwise":
            self.gate_proj = nn.Linear(self.d_model, 1)
        else:
            self.gate_proj = nn.Linear(self.d_model, self.d_h)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run GRIT attention and apply the SiGMA gate.

        Args:
            x: Node features ``[N, d_model]``.
            edge_index: Sparse (or full-graph) edges ``[2, E]``.
            edge_attr: Optional edge features ``[E, edge_dim]``.

        Returns:
            ``(gated_out, gamma)`` with shapes ``[N, d_h]`` and
            ``[N, 1]`` or ``[N, d_h]``.
        """
        num_e = int(edge_index.size(1))
        dev, dt = x.device, x.dtype
        if edge_attr is None:
            eh = torch.zeros((num_e, self.d_model), device=dev, dtype=dt)
        else:
            feat = edge_attr.float()
            if feat.size(-1) == self.d_model:
                eh = feat.to(dtype=dt)
            else:
                eh = self.edge_proj(feat).to(dtype=dt)

        batch = _GritHybridBatch(x, edge_index, eh)
        h_out, _e_out = self.attn(batch)
        raw = h_out.reshape(x.size(0), self.d_h)

        if self.gate_proj is None:
            gamma = torch.ones(raw.size(0), 1, device=raw.device, dtype=raw.dtype)
            return raw, gamma

        g = self.gate_proj(x)
        gamma = torch.sigmoid(g)
        return raw * gamma, gamma
