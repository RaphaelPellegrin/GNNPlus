"""Transolver++ Physics-Attention for SiGMA hybrid heads.

Ported from ``thuml/Transolver_plus`` ``Physics_Attention_1D_Eidetic`` without
distributed ``all_reduce``. Each call operates on a single mesh
(``B×N×C`` with ``B=1`` per graph, or a true batch of equal-sized meshes).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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


def gumbel_softmax(logits: Tensor, tau: Tensor | float, hard: bool = False) -> Tensor:
    """Differentiable categorical sample (Transolver++ Rep-Slice).

    Args:
        logits: Slice logits ``[..., G]``.
        tau: Temperature broadcastable to ``logits`` (local adaptive in ++).
        hard: If True, use straight-through one-hot.

    Returns:
        Soft (or ST-hard) weights over the last dimension.
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / tau
    y = F.softmax(y, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y


class PhysicsAttention1DEidetic(nn.Module):
    """Slice → attention among M tokens → deslice (Transolver++).

    Uses a single ``in_project_x`` (no duplicate ``fx`` path), Gumbel-Softmax
    slice weights, and a learned per-token temperature.
    """

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 32,
        use_gumbel: bool = True,
        temperature_bias: float = 0.5,
    ) -> None:
        """Build Physics-Attention.

        Args:
            dim: Channel width of the input / output tokens.
            heads: Number of attention heads inside the module.
            dim_head: Per-head channel width.
            dropout: Dropout on the desliced output projection.
            slice_num: Number of physics slices ``M``.
            use_gumbel: If False, use temperature-scaled softmax (original
                Transolver) instead of Gumbel-Softmax.
            temperature_bias: Additive bias on the adaptive temperature.
        """
        super().__init__()
        if heads < 1 or dim_head < 1 or slice_num < 1:
            raise ValueError("heads, dim_head, and slice_num must be positive")
        inner_dim = dim_head * heads
        self.dim = int(dim)
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        self.slice_num = int(slice_num)
        self.use_gumbel = bool(use_gumbel)
        self.dropout = nn.Dropout(float(dropout))
        self.bias = nn.Parameter(
            torch.ones(1, heads, 1, 1) * float(temperature_bias)
        )
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply Physics-Attention.

        Args:
            x: Token features ``[B, N, C]`` with ``C == dim``.

        Returns:
            Updated tokens ``[B, N, C]``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B,N,C], got shape {tuple(x.shape)}")
        bsz, n_nodes, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"channel dim {channels} != PhysicsAttention dim {self.dim}"
            )

        x_mid = (
            self.in_project_x(x)
            .reshape(bsz, n_nodes, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N D
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_logits = self.in_project_slice(x_mid)  # B H N G
        if self.use_gumbel:
            slice_weights = gumbel_softmax(slice_logits, temperature)
        else:
            slice_weights = F.softmax(slice_logits / temperature, dim=-1)

        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnd,bhng->bhgd", x_mid, slice_weights)
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0 if not self.training else 0.0
        )

        out_x = torch.einsum("bhgd,bhng->bhnd", out_slice, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).contiguous().reshape(bsz, n_nodes, -1)
        return self.to_out(out_x)


class _PhysicsAttnHead(nn.Module):
    """One SiGMA attention head using Transolver++ Physics-Attention.

    Projects ``d_model → d_h``, runs Physics-Attn in ``d_h`` space (1 internal
    head), then applies the SiGMA gate. Batched PyG graphs are handled by
    splitting on ``batch`` and running within each graph.
    """

    def __init__(
        self,
        d_model: int,
        d_h: int,
        gate_mode: str = "elementwise",
        *,
        slice_num: int = 32,
        use_gumbel: bool = True,
        temperature_bias: float = 0.5,
        attn_dropout: float = 0.0,
    ) -> None:
        """Build a gated Physics-Attention head.

        Args:
            d_model: Incoming node feature width.
            d_h: Per-head output width.
            gate_mode: SiGMA gate style.
            slice_num: Number of physics slices.
            use_gumbel: Transolver++ Gumbel-Softmax when True.
            temperature_bias: Temperature bias init.
            attn_dropout: Dropout inside Physics-Attention ``to_out``.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.d_h = int(d_h)
        self.gate_mode: GateMode = _normalize_gate_mode(gate_mode)
        self.input_proj = nn.Linear(self.d_model, self.d_h)
        self.physics = PhysicsAttention1DEidetic(
            dim=self.d_h,
            heads=1,
            dim_head=self.d_h,
            dropout=float(attn_dropout),
            slice_num=int(slice_num),
            use_gumbel=bool(use_gumbel),
            temperature_bias=float(temperature_bias),
        )
        if self.gate_mode == "none":
            self.gate_proj: Optional[nn.Linear] = None
        elif self.gate_mode == "headwise":
            self.gate_proj = nn.Linear(self.d_model, 1)
        else:
            self.gate_proj = nn.Linear(self.d_model, self.d_h)

    def _forward_single_graph(self, x_g: Tensor) -> Tensor:
        """Run Physics-Attn on one graph's nodes ``[N, d_model]`` → ``[N, d_h]``."""
        h = self.input_proj(x_g).unsqueeze(0)  # 1 N d_h
        return self.physics(h).squeeze(0)

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply Physics-Attn within each graph and gate.

        Args:
            x: Node features ``[N, d_model]``.
            batch: Graph assignment ``[N]``.

        Returns:
            ``(gated_out, gamma)`` with shapes ``[N, d_h]`` and gate tensor.
        """
        if batch.numel() != x.size(0):
            raise ValueError("batch length must match number of nodes")

        raw = torch.zeros(x.size(0), self.d_h, device=x.device, dtype=x.dtype)
        if batch.numel() == 0:
            pass
        else:
            num_graphs = int(batch.max().item()) + 1
            for g in range(num_graphs):
                mask = batch == g
                if not bool(mask.any()):
                    continue
                raw[mask] = self._forward_single_graph(x[mask])

        if self.gate_proj is None:
            gamma = torch.ones(raw.size(0), 1, device=raw.device, dtype=raw.dtype)
            return raw, gamma
        g = self.gate_proj(x)
        gamma = torch.sigmoid(g)
        return raw * gamma, gamma
