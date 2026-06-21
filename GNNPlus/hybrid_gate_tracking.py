"""Hybrid GNN headwise gate statistics for W&B (Heterogeneity_Profile parity)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, MutableMapping, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg


def hybrid_gate_logging_enabled() -> bool:
    """Return whether gate means should be logged this run."""
    if not bool(getattr(cfg.wandb, 'use', False)):
        return False
    if str(getattr(cfg.model, 'type', '')) != 'hybrid_gnn':
        return False
    return bool(getattr(cfg.gnn.hybrid, 'log_gate_stats', True))


def collect_hybrid_gate_stats(model: nn.Module, batch: Batch) -> Dict[str, float]:
    """Run one forward through hybrid blocks; return ``layer*/attn_*`` gate means.

    Uses the first mini-batch only (same as Heterogeneity_Profile). Does not run
    the prediction head.
    """
    if not hasattr(model, 'collect_gate_stats'):
        return {}
    return model.collect_gate_stats(batch)


def build_hybrid_gate_wandb_log(
    model: nn.Module,
    train_loader: Iterable[Any],
) -> Optional[Dict[str, float]]:
    """Grab one train batch and build ``gates/layer*/...`` scalars for W&B."""
    if not hybrid_gate_logging_enabled():
        return None

    device = torch.device(cfg.accelerator)
    was_training = model.training
    model.eval()
    gate_log: Dict[str, float] = {}
    try:
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            if not isinstance(sample_batch, Batch):
                return None
            sample_batch = sample_batch.to(device)
            stats = collect_hybrid_gate_stats(model, sample_batch)
            gate_log = {f'gates/{key}': float(val) for key, val in stats.items()}
    finally:
        if was_training:
            model.train()
    return gate_log or None


def log_hybrid_gate_stats(
    model: nn.Module,
    train_loader: Iterable[Any],
    epoch_log: MutableMapping[str, Any],
    train_loader_iter: Optional[Iterable[Any]] = None,
) -> None:
    """Merge gate scalars into ``epoch_log`` (for ``wandb.log``).

    Args:
        model: ``HybridGNN`` when gate logging is enabled.
        train_loader: Training loader (first batch used).
        epoch_log: Dict passed to ``wandb.log`` for this step.
        train_loader_iter: Unused; kept for API compatibility.
    """
    del train_loader_iter
    gate_log = build_hybrid_gate_wandb_log(model, train_loader)
    if gate_log:
        epoch_log.update(gate_log)
