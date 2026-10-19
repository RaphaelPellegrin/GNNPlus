"""SiGMA-lite layer: one gate on top of the MMA block (Gated-GPS style).

SiGMA gates every head separately before concatenation. SiGMA-lite keeps the
heads ungated (the MMA block: any number of attention and MP heads, either
possibly zero), mixes them with ``out_proj``, and gates the mixed update once:

    x' = Norm(x)
    m  = W_O [A_1(x') | ... | A_a(x') | M_1(x') | ... | M_g(x')]
    x  <- x + W_1 (m * act(W_g x'))

with ``act`` = sigmoid (default) or GELU (the Gated-GPS gate, Gao et al.,
Briefings in Bioinformatics 2025, eq. 6). Gating the concatenation before
``out_proj`` elementwise would reproduce SiGMA ``gate: elementwise`` exactly,
so the gate sits after the heads are mixed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from GNNPlus.layer.gate_override import GateOverrideMode, apply_gate_override
from GNNPlus.layer.gated_hybrid_layer import (
    AttnMaskType,
    AttnType,
    GatedHybridGraphLayer,
    NormType,
)

SigmaLiteGateAct = Literal["sigmoid", "gelu"]


def _normalize_gate_act(gate_act: str) -> SigmaLiteGateAct:
    """Validate and canonicalize the SiGMA-lite gate nonlinearity."""
    act = str(gate_act).strip().lower()
    if act not in ("sigmoid", "gelu"):
        raise ValueError(f"Unknown SiGMA-lite gate_act: {gate_act!r} (expected sigmoid|gelu)")
    return cast(SigmaLiteGateAct, act)


class SigmaLiteLayer(nn.Module):
    """SiGMA-lite block: ungated MMA heads, one gate on the mixed output.

    Args:
        d_model: Node feature width ``d``.
        num_attn_heads: Number of attention heads (may be 0).
        num_gnn_heads: Number of message-passing heads (may be 0).
        d_h: Per-head width.
        attn_mask_type: ``full`` or ``graph_restricted`` dense attention mask.
        norm_type: Pre-norm applied to ``x`` before heads and gate.
        gnn_types: MP head kinds (e.g. ``["GCN", "GIN"]``).
        attn_dropout: Dropout on attention weights.
        mp_gnn_dropout: Dropout inside MP heads.
        residual: Add ``x`` back after the gated update.
        attn_type: ``vanilla`` (dense QK) or ``grit`` attention heads.
        edge_dim: Edge feature width for GRIT heads.
        grit_clamp: GRIT score clamp.
        grit_edge_enhance: GRIT edge enhancement flag.
        grit_act: GRIT activation.
        grit_use_bias: GRIT bias flag.
        gate_act: ``sigmoid`` (SiGMA-lite) or ``gelu`` (Gated-GPS gate).
    """

    def __init__(
        self,
        d_model: int,
        num_attn_heads: int,
        num_gnn_heads: int,
        d_h: int,
        attn_mask_type: AttnMaskType = "full",
        norm_type: NormType = "layernorm",
        gnn_types: Optional[List[str]] = None,
        attn_dropout: float = 0.0,
        mp_gnn_dropout: float = 0.0,
        residual: bool = True,
        attn_type: AttnType = "vanilla",
        edge_dim: Optional[int] = None,
        grit_clamp: float = 5.0,
        grit_edge_enhance: bool = True,
        grit_act: str = "relu",
        grit_use_bias: bool = False,
        gate_act: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.mma = GatedHybridGraphLayer(
            d_model=d_model,
            num_attn_heads=num_attn_heads,
            num_gnn_heads=num_gnn_heads,
            d_h=d_h,
            attn_mask_type=attn_mask_type,
            gate_mode="none",
            mp_gate_mode="none",
            norm_type=norm_type,
            gnn_types=gnn_types,
            attn_dropout=attn_dropout,
            mp_gnn_dropout=mp_gnn_dropout,
            block_bn=False,
            residual=False,
            identity_proj=False,
            attn_type=attn_type,
            edge_dim=edge_dim,
            grit_clamp=grit_clamp,
            grit_edge_enhance=grit_edge_enhance,
            grit_act=grit_act,
            grit_use_bias=grit_use_bias,
        )
        self.gate_act: SigmaLiteGateAct = _normalize_gate_act(gate_act)
        self.residual = bool(residual)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.gated_proj = nn.Linear(d_model, d_model)

    def _gate(self, src: Tensor) -> Tensor:
        """Return the per-node, per-feature gate ``act(W_g src)``."""
        g = self.gate_proj(src)
        if self.gate_act == "gelu":
            return F.gelu(g)
        return torch.sigmoid(g)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        attn_source: Optional[Tensor] = None,
        mp_source: Optional[Tensor] = None,
        edge_index_attn: Optional[Tensor] = None,
        edge_attr_attn: Optional[Tensor] = None,
        edge_index_mp: Optional[Tensor] = None,
        edge_attr_mp: Optional[Tensor] = None,
        return_gate_stats: bool = False,
        return_attn_weights: bool = False,
        mp_head_mask: Optional[Sequence[bool]] = None,
        gate_override: Optional[GateOverrideMode] = None,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Apply the SiGMA-lite block.

        Arguments match :meth:`GatedHybridGraphLayer.forward`, so the layer is
        a drop-in replacement inside :class:`HybridGNN`. ``gate_override``
        (inference only) clamps the single SiGMA-lite gate.

        Returns:
            Updated node features ``[N, d_model]``, plus an aux dict when
            ``return_gate_stats`` or ``return_attn_weights`` is set. The aux
            dict reports the gate under ``gate_stats['lite_gate_mean']`` and
            ``gate_values['lite']``.
        """
        mma_out = self.mma(
            x,
            edge_index,
            batch,
            edge_attr=edge_attr,
            attn_source=attn_source,
            mp_source=mp_source,
            edge_index_attn=edge_index_attn,
            edge_attr_attn=edge_attr_attn,
            edge_index_mp=edge_index_mp,
            edge_attr_mp=edge_attr_mp,
            return_gate_stats=False,
            return_attn_weights=return_attn_weights,
            mp_head_mask=mp_head_mask,
        )
        aux: Dict[str, Any] = {}
        if isinstance(mma_out, tuple):
            mixed, aux = mma_out
        else:
            mixed = mma_out

        gamma = apply_gate_override(self._gate(self.mma.norm(x)), gate_override, batch)
        update = cast(Tensor, self.gated_proj(mixed * gamma))
        out = x + update if self.residual else update

        if not return_gate_stats and not return_attn_weights:
            return out
        if return_gate_stats:
            aux["gate_stats"] = {"lite_gate_mean": gamma.detach().mean().item()}
            aux["gate_values"] = {"attn": [], "gnn": [], "lite": [gamma.detach()]}
        return out, aux
