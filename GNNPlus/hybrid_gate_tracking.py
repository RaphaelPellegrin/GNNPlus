"""Hybrid GNN gate statistics for W&B (Heterogeneity_Profile parity)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, MutableMapping, Optional

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg

_GATE_DIAG_LOGGED = False


def hybrid_gate_logging_enabled() -> bool:
    """Return whether gate means should be logged this run."""
    if not bool(getattr(cfg.wandb, 'use', False)):
        return False
    model_type = str(getattr(cfg.model, 'type', ''))
    if model_type == 'hybrid_gnn':
        hybrid_cfg = getattr(cfg.gnn, 'hybrid', None)
        if hybrid_cfg is None:
            return False
        return bool(getattr(hybrid_cfg, 'log_gate_stats', True))
    if model_type == 'custom_gnn':
        if str(getattr(cfg.gnn, 'layer_type', '')) != 'gcne':
            return False
        gate = str(getattr(cfg.gnn, 'gate', '') or '').lower()
        if gate not in ('headwise', 'elementwise'):
            return False
        return bool(getattr(cfg.gnn, 'log_gate_stats', False))
    return False


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner ``HybridGNN`` through GraphGym / DDP wrappers."""
    cur: nn.Module = model
    for _ in range(8):
        if hasattr(cur, 'collect_gate_stats'):
            return cur
        nxt: Optional[nn.Module] = None
        inner = getattr(cur, 'module', None)
        if isinstance(inner, nn.Module):
            nxt = inner
        else:
            wrapped = getattr(cur, 'model', None)
            if isinstance(wrapped, nn.Module):
                nxt = wrapped
        if nxt is None:
            break
        cur = nxt
    return cur


def _is_graph_batch(batch: Any) -> bool:
    """Return whether ``batch`` looks like a PyG graph mini-batch."""
    return (
        hasattr(batch, 'to')
        and hasattr(batch, 'x')
        and hasattr(batch, 'edge_index')
    )


def collect_hybrid_gate_stats(model: nn.Module, batch: Any) -> Dict[str, float]:
    """Run one forward through hybrid blocks; return ``layer*/attn_*`` gate means.

    Uses the first mini-batch only (same as Heterogeneity_Profile). Does not run
    the prediction head.
    """
    core = _unwrap_model(model)
    if not hasattr(core, 'collect_gate_stats'):
        return {}
    return core.collect_gate_stats(batch)


def build_hybrid_gate_wandb_log(
    model: nn.Module,
    train_loader: Iterable[Any],
) -> Dict[str, float]:
    """Grab one train batch and build ``gates/layer*/...`` scalars for W&B."""
    global _GATE_DIAG_LOGGED

    if not hybrid_gate_logging_enabled():
        return {}

    device = torch.device(cfg.accelerator)
    core = _unwrap_model(model)
    was_training = model.training
    model.eval()
    gate_log: Dict[str, float] = {}
    try:
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            if not _is_graph_batch(sample_batch):
                logging.warning(
                    'Hybrid gate stats: train loader batch is not a PyG graph '
                    '(type=%s); skipping gate logging.',
                    type(sample_batch).__name__,
                )
                return {}
            sample_batch = sample_batch.to(device)
            stats = collect_hybrid_gate_stats(model, sample_batch)
            if not stats:
                gate_hint = getattr(cfg.gnn.hybrid, 'gate', '?') if hasattr(cfg.gnn, 'hybrid') else getattr(cfg.gnn, 'gate', '?')
                logging.warning(
                    'Hybrid gate stats: collect_gate_stats returned no values '
                    '(model=%s, gate=%s).',
                    type(core).__name__,
                    gate_hint,
                )
                return {}
            gate_log = {f'gates/{key}': float(val) for key, val in stats.items()}
            n_gates = len(gate_log)
            gate_log['gates/_num_metrics'] = float(n_gates)
    except StopIteration:
        logging.warning('Hybrid gate stats: train loader is empty; skipping.')
    except Exception:
        logging.exception('Hybrid gate stats: failed to collect gate means.')
    finally:
        if was_training:
            model.train()

    if gate_log and not _GATE_DIAG_LOGGED:
        sample_key = next(
            k for k in gate_log if k != 'gates/_num_metrics'
        )
        logging.info(
            'Hybrid gate stats: logging %d W&B metrics (e.g. %s=%.4f).',
            int(gate_log['gates/_num_metrics']),
            sample_key,
            gate_log[sample_key],
        )
        _GATE_DIAG_LOGGED = True
    return gate_log


def log_hybrid_gate_stats(
    model: nn.Module,
    train_loader: Iterable[Any],
    epoch_log: MutableMapping[str, Any],
    train_loader_iter: Optional[Iterable[Any]] = None,
) -> Dict[str, float]:
    """Merge gate scalars into ``epoch_log`` and return them for W&B summary.

    Args:
        model: ``HybridGNN`` when gate logging is enabled.
        train_loader: Training loader (first batch used).
        epoch_log: Dict passed to ``wandb.log`` for this step.
        train_loader_iter: Unused; kept for API compatibility.

    Returns:
        Gate metrics keyed as ``gates/layer*/...`` (empty when disabled).
    """
    del train_loader_iter
    gate_log = build_hybrid_gate_wandb_log(model, train_loader)
    if gate_log:
        epoch_log.update(gate_log)
    return gate_log


def publish_gate_stats_to_wandb(
    run: Any,
    gate_log: Dict[str, float],
    step: int,
) -> None:
    """Push gate metrics to W&B history and run summary.

    Args:
        run: Active ``wandb.Run``.
        gate_log: Metrics from :func:`build_hybrid_gate_wandb_log`.
        step: Training epoch index used as the W&B step.
    """
    if not gate_log:
        return
    # Include ``train/epoch`` so ``define_metric(..., step_metric='train/epoch')``
    # links gate series to the training epoch axis.
    payload = dict(gate_log)
    payload['train/epoch'] = float(step)
    run.log(payload, step=step)
    run.summary.update(gate_log)
