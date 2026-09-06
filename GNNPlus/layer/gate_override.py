"""Inference-time SiGMA gate overrides (clamp / mean)."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

GateOverrideMode = Literal["ones", "mean"]


def apply_gate_override(
    gamma: Tensor,
    mode: Optional[GateOverrideMode],
    batch_ids: Optional[Tensor] = None,
) -> Tensor:
    """Replace learned sigmoid gates at inference (eval clamp).

    Args:
        gamma: Per-node gate tensor ``[N, 1]`` (headwise) or ``[N, d_h]``.
        mode: ``None`` keeps ``gamma``. ``ones`` forces full head scale.
            ``mean`` replaces each node gate by the mean gate within its
            graph (or the global mean if ``batch_ids`` is ``None``).
        batch_ids: Graph id per node ``[N]`` (PyG ``batch.batch``).

    Returns:
        Gate tensor with the same shape as ``gamma``.
    """
    if mode is None:
        return gamma
    if mode == "ones":
        return torch.ones_like(gamma)
    if mode == "mean":
        if batch_ids is None:
            return gamma.mean(dim=0, keepdim=True).expand_as(gamma)
        num_graphs = int(batch_ids.max().item()) + 1 if batch_ids.numel() else 0
        if num_graphs <= 0:
            return gamma.mean(dim=0, keepdim=True).expand_as(gamma)
        sums = torch.zeros(
            num_graphs,
            *gamma.shape[1:],
            device=gamma.device,
            dtype=gamma.dtype,
        )
        counts = torch.zeros(
            num_graphs,
            *([1] * (gamma.dim() - 1)),
            device=gamma.device,
            dtype=gamma.dtype,
        )
        sums.index_add_(0, batch_ids, gamma)
        ones = torch.ones(
            gamma.size(0),
            *([1] * (gamma.dim() - 1)),
            device=gamma.device,
            dtype=gamma.dtype,
        )
        counts.index_add_(0, batch_ids, ones)
        gmean = sums / counts.clamp_min(1.0)
        return gmean[batch_ids]
    raise ValueError(f"Unknown gate_override mode: {mode!r} (expected ones|mean)")
