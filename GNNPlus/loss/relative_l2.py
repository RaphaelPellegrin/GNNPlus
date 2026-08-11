"""Relative L2 loss for PDE / Transolver-style node regression."""

from __future__ import annotations

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


def _relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean relative L2 (local copy to avoid import cycles at package init)."""
    pred_f = pred.reshape(pred.size(0), -1).float()
    true_f = true.reshape(true.size(0), -1).float()
    diff = torch.norm(pred_f - true_f, p=2, dim=1)
    denom = torch.norm(true_f, p=2, dim=1).clamp_min(eps)
    return (diff / denom).mean()


@register_loss("pde_losses")
def pde_losses(pred: torch.Tensor, true: torch.Tensor):
    """Register relative-L2 loss for PDE heads.

    Returns:
        ``(loss, pred)`` when ``cfg.model.loss_fun == 'relative_l2'``; else ``None``.
    """
    if cfg.model.loss_fun == "relative_l2":
        if pred.dim() == 1:
            loss = _relative_l2(pred.unsqueeze(0), true.unsqueeze(0))
        else:
            loss = _relative_l2(pred.unsqueeze(0), true.unsqueeze(0))
        return loss, pred
    return None
