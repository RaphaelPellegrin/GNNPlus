"""Per-τ / per-layer root gate W&B logging for GinDepthRouting.

Logs mean root MP gate γ at each hybrid layer on τ=0 (1-GIN / shallow) vs
τ=1 (2-GIN / deep) graphs — evidence that a 2-layer SiGMA opens layer-1 more
on deep-label graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg

from GNNPlus.hybrid_gate_tracking import (
    _is_graph_batch,
    _unwrap_model,
    hybrid_gate_logging_enabled,
)


def gin_depth_routing_gate_logging_enabled() -> bool:
    """Return whether per-τ depth gate metrics should be logged to W&B."""
    if not hybrid_gate_logging_enabled():
        return False
    if str(getattr(cfg.dataset, "format", "")) != "PyG-GinDepthRouting":
        return False
    hybrid = getattr(cfg.gnn, "hybrid", None)
    if hybrid is None:
        return False
    gate_mode = str(getattr(hybrid, "gate", "none")).lower()
    return gate_mode not in ("none", "off")


def _root_indices(batch: Any) -> torch.Tensor:
    """Global node indices of the root (first node) per graph in a batch."""
    if hasattr(batch, "ptr") and batch.ptr is not None:
        return batch.ptr[:-1].long()
    batch_ids = batch.batch
    num_graphs = int(batch_ids.max().item()) + 1 if batch_ids.numel() else 0
    roots = torch.zeros(num_graphs, dtype=torch.long, device=batch_ids.device)
    for graph_idx in range(num_graphs):
        mask = batch_ids == graph_idx
        roots[graph_idx] = int(torch.nonzero(mask, as_tuple=False)[0].item())
    return roots


def _safe_mean(vals: List[float]) -> float:
    """Mean of floats or NaN when empty."""
    return float(mean(vals)) if vals else float("nan")


@dataclass
class DepthGateAccumulator:
    """Root MP gates split by (layer, τ)."""

    # layer -> (tau0 list, tau1 list)
    by_layer_tau0: Dict[int, List[float]] = field(default_factory=dict)
    by_layer_tau1: Dict[int, List[float]] = field(default_factory=dict)
    n_ok_batches: int = 0
    n_fail_batches: int = 0

    @property
    def has_samples(self) -> bool:
        """True when at least one root gate was recorded."""
        return any(self.by_layer_tau0.values()) or any(self.by_layer_tau1.values())


def accumulate_depth_root_gates_from_batch(
    core: nn.Module,
    batch: Any,
    tau: torch.Tensor,
    accum: DepthGateAccumulator,
    *,
    head_idx: int = 0,
) -> bool:
    """Append root γ for all layers of one mini-batch; return True on success."""
    if not hasattr(core, "collect_per_graph_gates"):
        return False
    try:
        gate_out = core.collect_per_graph_gates(batch.clone())
        gnn_node = gate_out["gnn_node"]
        if gnn_node.ndim != 3:
            raise ValueError(f"expected gnn_node [N,L,Ng], got {tuple(gnn_node.shape)}")
        num_layers = int(gnn_node.shape[1])
        num_heads = int(gnn_node.shape[-1])
        if num_heads <= head_idx:
            raise ValueError(f"gnn_node has {num_heads} heads; need head_idx={head_idx}")
        roots = _root_indices(batch).to(gnn_node.device)
        tau_cpu = tau.detach().view(-1).long().cpu()
        for layer_idx in range(num_layers):
            root_g = gnn_node[roots, layer_idx, head_idx].detach().float().cpu()
            if int(root_g.numel()) != int(tau_cpu.numel()):
                raise ValueError(
                    f"tau graphs={int(tau_cpu.numel())} vs root gates={int(root_g.numel())}",
                )
            accum.by_layer_tau0.setdefault(layer_idx, []).extend(
                root_g[tau_cpu == 0].tolist(),
            )
            accum.by_layer_tau1.setdefault(layer_idx, []).extend(
                root_g[tau_cpu == 1].tolist(),
            )
        accum.n_ok_batches += 1
        return True
    except Exception:
        accum.n_fail_batches += 1
        if accum.n_fail_batches <= 1:
            logging.exception("Depth root gate collection failed on first bad batch")
        return False


def build_per_tau_depth_gate_wandb_log(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    split_name: str = "val",
    max_batches: int = 0,
    head_idx: int = 0,
) -> Dict[str, float]:
    """Aggregate root γ per layer on τ=0 vs τ=1 for W&B.

    Keys under ``gates_by_tau_depth/{split}/``:

    - ``layer{k}/tau0/mean_gamma``, ``layer{k}/tau1/mean_gamma``
    - ``routing/delta_layer{k}`` = mean(γ|τ=1) − mean(γ|τ=0)
      (for k≥1, want > 0 if deep graphs open deeper layers)
    """
    if not gin_depth_routing_gate_logging_enabled():
        return {}

    core = _unwrap_model(model)
    if not hasattr(core, "collect_per_graph_gates"):
        return {}

    device = torch.device(cfg.accelerator)
    was_training = model.training
    model.eval()
    accum = DepthGateAccumulator()

    try:
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                if max_batches > 0 and batch_i >= max_batches:
                    break
                if not _is_graph_batch(batch):
                    continue
                if not hasattr(batch, "tau") or batch.tau is None:
                    logging.warning(
                        "Depth gate logging: batch missing tau on %s split",
                        split_name,
                    )
                    return {}
                batch = batch.to(device)
                tau = batch.tau.view(-1).long()
                accumulate_depth_root_gates_from_batch(
                    core,
                    batch,
                    tau,
                    accum,
                    head_idx=head_idx,
                )
    except Exception:
        logging.exception("Depth gate logging failed on %s split", split_name)
        return {}
    finally:
        if was_training:
            model.train()

    if not accum.has_samples:
        return {}

    prefix = f"gates_by_tau_depth/{split_name}"
    out: Dict[str, float] = {}
    layer_ids = sorted(
        set(accum.by_layer_tau0.keys()) | set(accum.by_layer_tau1.keys()),
    )
    for layer_idx in layer_ids:
        mean_t0 = _safe_mean(accum.by_layer_tau0.get(layer_idx, []))
        mean_t1 = _safe_mean(accum.by_layer_tau1.get(layer_idx, []))
        out[f"{prefix}/layer{layer_idx}/tau0/mean_gamma"] = mean_t0
        out[f"{prefix}/layer{layer_idx}/tau1/mean_gamma"] = mean_t1
        out[f"{prefix}/routing/delta_layer{layer_idx}"] = mean_t1 - mean_t0
        out[f"{prefix}/layer{layer_idx}/_n_graphs_tau0"] = float(
            len(accum.by_layer_tau0.get(layer_idx, [])),
        )
        out[f"{prefix}/layer{layer_idx}/_n_graphs_tau1"] = float(
            len(accum.by_layer_tau1.get(layer_idx, [])),
        )
    return out


def publish_per_tau_depth_gate_stats_to_wandb(
    run: Any,
    gate_log: Dict[str, float],
    step: int,
) -> None:
    """Push per-τ depth gate metrics to W&B."""
    if not gate_log:
        return
    payload = dict(gate_log)
    payload["train/epoch"] = float(step)
    run.log(payload, step=step)
    run.summary.update(gate_log)
