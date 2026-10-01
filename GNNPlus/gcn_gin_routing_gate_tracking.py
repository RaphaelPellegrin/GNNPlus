"""Per-τ root gate W&B logging for the GcnGinRouting synthetic benchmark.

Logs mean root MP gate γ for GIN vs GCN heads on τ=0 (GCN-type) vs τ=1
(GIN-type) graphs — the routing evidence that pooled W&B gate curves hide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg

from GNNPlus.hybrid_gate_tracking import (
    _is_graph_batch,
    _unwrap_model,
    hybrid_gate_logging_enabled,
)


def gcn_gin_routing_gate_logging_enabled() -> bool:
    """Return whether per-τ root gate metrics should be logged to W&B."""
    if not hybrid_gate_logging_enabled():
        return False
    if str(getattr(cfg.dataset, "format", "")) != "PyG-GcnGinRouting":
        return False
    hybrid = getattr(cfg.gnn, "hybrid", None)
    if hybrid is None:
        return False
    gate_mode = str(getattr(hybrid, "gate", "none")).lower()
    if gate_mode in ("none", "off"):
        return False
    _, _, two_head = hybrid_head_indices(str(getattr(hybrid, "gnn_types", "")))
    return two_head


def hybrid_head_indices(gnn_types: str) -> Tuple[int, int, bool]:
    """Return ``(gin_head_idx, gcn_head_idx, is_two_head)`` from ``gnn_types``."""
    parts = [p.strip() for p in str(gnn_types).split(",") if p.strip()]
    if len(parts) != 2:
        return 0, 1, False
    gin_idx = 0
    gcn_idx = 1
    for idx, kind in enumerate(parts):
        upper = kind.upper()
        if "GIN" in upper or "ROUTING_SUM" in upper:
            gin_idx = idx
        if "GCN" in upper or "ROUTING_NORM" in upper or "NORMGCN" in upper:
            gcn_idx = idx
    return gin_idx, gcn_idx, True


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
class RootGateAccumulator:
    """Running lists of root MP gate γ split by graph type τ."""

    gin_t0: List[float] = field(default_factory=list)
    gin_t1: List[float] = field(default_factory=list)
    gcn_t0: List[float] = field(default_factory=list)
    gcn_t1: List[float] = field(default_factory=list)
    n_ok_batches: int = 0
    n_fail_batches: int = 0

    @property
    def has_samples(self) -> bool:
        """True when at least one root gate was recorded."""
        return bool(self.gin_t0 or self.gin_t1)

    def means(self) -> Dict[str, float]:
        """Per-τ mean root gates and routing deltas."""
        mean_gin_t0 = _safe_mean(self.gin_t0)
        mean_gin_t1 = _safe_mean(self.gin_t1)
        mean_gcn_t0 = _safe_mean(self.gcn_t0)
        mean_gcn_t1 = _safe_mean(self.gcn_t1)
        return {
            "gin_gate_tau0": mean_gin_t0,
            "gin_gate_tau1": mean_gin_t1,
            "gcn_gate_tau0": mean_gcn_t0,
            "gcn_gate_tau1": mean_gcn_t1,
            "delta_gcn": mean_gcn_t0 - mean_gcn_t1,
            "delta_gin": mean_gin_t1 - mean_gin_t0,
        }


def accumulate_root_gates_from_batch(
    core: nn.Module,
    batch: Any,
    tau: torch.Tensor,
    gin_idx: int,
    gcn_idx: int,
    accum: RootGateAccumulator,
    *,
    layer_idx: int = 0,
) -> bool:
    """Append root γ for one mini-batch; return True on success."""
    if not hasattr(core, "collect_per_graph_gates"):
        return False
    try:
        gate_out = core.collect_per_graph_gates(batch.clone())
        gnn_node = gate_out["gnn_node"]
        if gnn_node.ndim != 3:
            raise ValueError(f"expected gnn_node [N,L,Ng], got {tuple(gnn_node.shape)}")
        num_heads = int(gnn_node.shape[-1])
        if num_heads <= max(gin_idx, gcn_idx):
            raise ValueError(
                f"gnn_node has {num_heads} heads; need gin={gin_idx}, gcn={gcn_idx}",
            )
        if int(gnn_node.shape[1]) <= layer_idx:
            raise ValueError(
                f"layer_idx={layer_idx} out of range for L={int(gnn_node.shape[1])}",
            )
        roots = _root_indices(batch).to(gnn_node.device)
        gin_root = gnn_node[roots, layer_idx, gin_idx].detach().float().cpu()
        gcn_root = gnn_node[roots, layer_idx, gcn_idx].detach().float().cpu()
        tau_cpu = tau.detach().view(-1).long().cpu()
        if int(gin_root.numel()) != int(tau_cpu.numel()):
            raise ValueError(
                f"tau graphs={int(tau_cpu.numel())} vs root gates={int(gin_root.numel())}",
            )
        accum.gin_t0.extend(gin_root[tau_cpu == 0].tolist())
        accum.gin_t1.extend(gin_root[tau_cpu == 1].tolist())
        accum.gcn_t0.extend(gcn_root[tau_cpu == 0].tolist())
        accum.gcn_t1.extend(gcn_root[tau_cpu == 1].tolist())
        accum.n_ok_batches += 1
        return True
    except Exception:
        accum.n_fail_batches += 1
        if accum.n_fail_batches <= 1:
            logging.exception("Root gate collection failed on first bad batch")
        return False


def build_per_tau_root_gate_wandb_log(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    split_name: str = "val",
    max_batches: int = 0,
    layer_idx: int = 0,
) -> Dict[str, float]:
    """Aggregate root γ_GIN / γ_GCN on τ=0 vs τ=1 graphs for W&B.

    Keys are prefixed with ``gates_by_tau/{split_name}/``:

    - ``tau0/mean_gin_gamma``, ``tau0/mean_gcn_gamma`` — GCN-type graphs
    - ``tau1/mean_gin_gamma``, ``tau1/mean_gcn_gamma`` — GIN-type graphs
    - ``routing/delta_gcn`` = mean(γ_GCN|τ=0) − mean(γ_GCN|τ=1)  (want > 0)
    - ``routing/delta_gin`` = mean(γ_GIN|τ=1) − mean(γ_GIN|τ=0)  (want > 0)

    Args:
        model: Hybrid GNN with ``collect_per_graph_gates``.
        loader: PyG loader (val or test); batches must expose ``batch.tau``.
        split_name: W&B path segment (e.g. ``val``, ``test``).
        max_batches: Cap batches scanned (0 = all).
        layer_idx: Hybrid MP layer for root gates (default 0).

    Returns:
        Flat dict of scalars for ``wandb.log`` (empty when disabled / no τ).
    """
    if not gcn_gin_routing_gate_logging_enabled():
        return {}

    core = _unwrap_model(model)
    if not hasattr(core, "collect_per_graph_gates"):
        return {}

    hybrid = getattr(cfg.gnn, "hybrid", None)
    gnn_types = str(getattr(hybrid, "gnn_types", "")) if hybrid is not None else ""
    gin_idx, gcn_idx, two_head = hybrid_head_indices(gnn_types)
    if not two_head:
        return {}

    device = torch.device(cfg.accelerator)
    was_training = model.training
    model.eval()

    accum = RootGateAccumulator()

    try:
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                if max_batches > 0 and batch_i >= max_batches:
                    break
                if not _is_graph_batch(batch):
                    continue
                if not hasattr(batch, "tau") or batch.tau is None:
                    logging.warning(
                        "Per-τ gate logging: batch missing tau on %s split",
                        split_name,
                    )
                    return {}
                batch = batch.to(device)
                tau = batch.tau.view(-1).long()
                accumulate_root_gates_from_batch(
                    core,
                    batch,
                    tau,
                    gin_idx,
                    gcn_idx,
                    accum,
                    layer_idx=layer_idx,
                )
    except Exception:
        logging.exception(
            "Per-τ gate logging failed on %s split",
            split_name,
        )
        return {}
    finally:
        if was_training:
            model.train()

    if not accum.has_samples:
        return {}

    prefix = f"gates_by_tau/{split_name}"
    stats = accum.means()
    mean_gin_t0 = stats["gin_gate_tau0"]
    mean_gin_t1 = stats["gin_gate_tau1"]
    mean_gcn_t0 = stats["gcn_gate_tau0"]
    mean_gcn_t1 = stats["gcn_gate_tau1"]
    delta_gcn = stats["delta_gcn"]
    delta_gin = stats["delta_gin"]

    out: Dict[str, float] = {
        f"{prefix}/tau0/mean_gin_gamma": mean_gin_t0,
        f"{prefix}/tau0/mean_gcn_gamma": mean_gcn_t0,
        f"{prefix}/tau1/mean_gin_gamma": mean_gin_t1,
        f"{prefix}/tau1/mean_gcn_gamma": mean_gcn_t1,
        f"{prefix}/routing/delta_gcn": delta_gcn,
        f"{prefix}/routing/delta_gin": delta_gin,
        f"{prefix}/routing/gcn_loving_tau0": mean_gcn_t0,
        f"{prefix}/routing/gin_loving_tau1": mean_gin_t1,
        f"{prefix}/_n_graphs_tau0": float(len(accum.gin_t0)),
        f"{prefix}/_n_graphs_tau1": float(len(accum.gin_t1)),
    }
    return out


def publish_per_tau_gate_stats_to_wandb(
    run: Any,
    gate_log: Dict[str, float],
    step: int,
) -> None:
    """Push per-τ gate metrics to W&B (same step axis as training)."""
    if not gate_log:
        return
    payload = dict(gate_log)
    payload["train/epoch"] = float(step)
    run.log(payload, step=step)
    run.summary.update(gate_log)
