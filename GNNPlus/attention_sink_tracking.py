"""Periodic attention-sink diagnostics for W&B (Fesser-style panels).

During training, every ``cfg.gnn.hybrid.attention_sink_every`` epochs (plus
epoch 0 and the final epoch), grab one small train batch, dump dense
within-graph attention, and log:

* per-(layer, head) attention heatmap (first graph, degree-sorted)
* per-layer mean-over-heads heatmap
* L×H sink-rate / max-α / sink value-norm heatmaps
* scalar summaries under ``attn_sinks/*``

PNG files are also written under ``<run_dir>/attention_sinks/epXXXX/``.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg

from GNNPlus.hybrid_gate_tracking import _is_graph_batch, _unwrap_model

_AS_DIAG_LOGGED = False


def attention_sink_logging_enabled() -> bool:
    """Return whether attention-sink W&B panels are enabled for this run."""
    if not bool(getattr(cfg.wandb, "use", False)):
        return False
    if str(getattr(cfg.model, "type", "")) != "hybrid_gnn":
        return False
    hybrid_cfg = getattr(cfg.gnn, "hybrid", None)
    if hybrid_cfg is None:
        return False
    if int(getattr(hybrid_cfg, "num_attn_heads", 0)) <= 0:
        return False
    return bool(getattr(hybrid_cfg, "log_attention_sinks", False))


def should_log_attention_sinks(epoch: int, max_epoch: int) -> bool:
    """Sparse epoch schedule: 0, every ``every``, and last epoch."""
    if not attention_sink_logging_enabled():
        return False
    every = max(1, int(getattr(cfg.gnn.hybrid, "attention_sink_every", 50)))
    if epoch <= 0:
        return True
    if epoch == max_epoch - 1:
        return True
    return epoch % every == 0


def _parse_layer_head(name: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse ``layer{L}_attn{H}`` (optional underscore before H)."""
    m = re.match(r"layer(\d+)_attn_?(\d+)$", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _first_graph_slice(
    batch_ids: torch.Tensor,
) -> Tuple[int, int]:
    """Return ``[start, end)`` node indices for graph 0 in a PyG batch."""
    mask = batch_ids == int(batch_ids.min().item())
    idx = torch.where(mask)[0]
    start = int(idx[0].item())
    end = int(idx[-1].item()) + 1
    return start, end


def _degree_order(
    edge_index: torch.Tensor,
    n: int,
    node_offset: int,
) -> np.ndarray:
    """Descending degree permutation for a single graph's local nodes."""
    deg = np.zeros(n, dtype=np.float64)
    if edge_index.numel() == 0:
        return np.arange(n)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for a, b in zip(src, dst):
        if node_offset <= a < node_offset + n and node_offset <= b < node_offset + n:
            deg[a - node_offset] += 1.0
            deg[b - node_offset] += 1.0
    return np.argsort(-deg)


def _sink_strength(A: np.ndarray) -> np.ndarray:
    """Column mean α_j = mean_i A[i, j] (row-stochastic → receiver mass / n)."""
    return A.mean(axis=0)


def _tau_sink_mask(A: np.ndarray, tau: float) -> np.ndarray:
    """τ·μ rule on column totals Â_j = sum_i A[i, j]."""
    a_hat = A.sum(axis=0)
    mu = float(a_hat.mean()) if a_hat.size else 0.0
    if mu <= 0:
        return np.zeros_like(a_hat, dtype=bool)
    return a_hat > (tau * mu)


def _stride_imshow(arr: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Downsample a square matrix for display."""
    h, w = arr.shape
    step = max(1, int(math.ceil(max(h, w) / max_side)))
    return arr[::step, ::step]


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    """Save a matplotlib figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _build_panels_for_batch(
    attention: Mapping[str, torch.Tensor],
    value_norms: Mapping[str, torch.Tensor],
    edge_index: torch.Tensor,
    batch_ids: torch.Tensor,
    *,
    tau: float,
    epsilon: float,
    out_dir: Path,
    epoch: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Create PNG panels + scalar summaries for one mini-batch.

    Returns:
        ``(wandb_images_dict, scalar_metrics)``. Image values are file paths
        (caller wraps as ``wandb.Image``).
    """
    start, end = _first_graph_slice(batch_ids)
    n_g = end - start
    order = _degree_order(edge_index, n_g, start)

    # Group keys by layer.
    by_layer: Dict[int, Dict[int, str]] = {}
    for key in attention:
        layer, head = _parse_layer_head(key)
        if layer is None or head is None:
            continue
        by_layer.setdefault(layer, {})[head] = key

    layers = sorted(by_layer)
    if not layers:
        return {}, {}

    n_heads = max(max(by_layer[L]) for L in layers) + 1
    sink_rate = np.full((len(layers), n_heads), np.nan, dtype=np.float64)
    max_alpha = np.full_like(sink_rate, np.nan)
    sink_vnorm_ratio = np.full_like(sink_rate, np.nan)
    mean_gate = np.full_like(sink_rate, np.nan)

    # --- per head heatmaps (first graph, degree-sorted) ---
    fig_h, axes = plt.subplots(
        len(layers),
        n_heads,
        figsize=(2.2 * n_heads, 2.0 * len(layers)),
        squeeze=False,
    )
    for li, layer in enumerate(layers):
        for head in range(n_heads):
            ax = axes[li][head]
            key = by_layer[layer].get(head)
            if key is None:
                ax.axis("off")
                continue
            A_full = attention[key].detach().cpu().float().numpy()
            A = A_full[start:end, start:end][np.ix_(order, order)]
            alpha = _sink_strength(A)
            sink_m = _tau_sink_mask(A, tau)
            sink_rate[li, head] = float(sink_m.any())
            max_alpha[li, head] = float(alpha.max()) if alpha.size else float("nan")
            sink_j = int(alpha.argmax()) if alpha.size else 0
            if key in value_norms:
                vn = value_norms[key].detach().cpu().float().numpy()[start:end][order]
                mean_vn = float(vn.mean()) + 1e-8
                sink_vnorm_ratio[li, head] = float(vn[sink_j] / mean_vn)
            vis = _stride_imshow(A)
            ax.imshow(vis, cmap="viridis", aspect="auto", interpolation="nearest")
            ax.axvline(sink_j / max(A.shape[1] / vis.shape[1], 1e-6), color="r", ls="--", lw=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if li == 0:
                ax.set_title(f"h{head}", fontsize=8)
            if head == 0:
                ax.set_ylabel(f"L{layer}", fontsize=8)
    fig_h.suptitle(
        f"Attn maps (graph0, deg↓) · ep={epoch} · red=argmax α",
        fontsize=10,
    )
    fig_h.tight_layout()
    path_heads = out_dir / f"ep{epoch:05d}_attn_by_layer_head.png"
    _save_fig(fig_h, path_heads)

    # --- mean over heads per layer ---
    fig_m, axes_m = plt.subplots(
        1,
        len(layers),
        figsize=(2.4 * len(layers), 2.4),
        squeeze=False,
    )
    for li, layer in enumerate(layers):
        mats: List[np.ndarray] = []
        for head, key in sorted(by_layer[layer].items()):
            A_full = attention[key].detach().cpu().float().numpy()
            mats.append(A_full[start:end, start:end][np.ix_(order, order)])
        mean_A = np.mean(np.stack(mats, axis=0), axis=0)
        alpha = _sink_strength(mean_A)
        sink_j = int(alpha.argmax()) if alpha.size else 0
        ax = axes_m[0][li]
        vis = _stride_imshow(mean_A)
        ax.imshow(vis, cmap="viridis", aspect="auto", interpolation="nearest")
        if mean_A.shape[1] > 0 and vis.shape[1] > 0:
            ax.axvline(sink_j * (vis.shape[1] / mean_A.shape[1]), color="r", ls="--", lw=0.8)
        ax.set_title(f"L{layer} mean_h", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig_m.suptitle(f"Mean over heads · ep={epoch}", fontsize=10)
    fig_m.tight_layout()
    path_mean = out_dir / f"ep{epoch:05d}_attn_mean_over_heads.png"
    _save_fig(fig_m, path_mean)

    def _heatmap(data: np.ndarray, title: str, fname: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> Path:
        fig, ax = plt.subplots(figsize=(max(3.0, 0.55 * n_heads + 1.5), max(3.0, 0.35 * len(layers) + 1.2)))
        im = ax.imshow(data, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"h{h}" for h in range(n_heads)], fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{L}" for L in layers], fontsize=7)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = out_dir / fname
        _save_fig(fig, path)
        return path

    path_sr = _heatmap(
        sink_rate,
        f"τ·μ sink present (τ={tau}) · ep={epoch}",
        f"ep{epoch:05d}_sink_rate_LxH.png",
        vmin=0.0,
        vmax=1.0,
    )
    path_ma = _heatmap(
        max_alpha,
        f"max α (column mean) · ep={epoch}",
        f"ep{epoch:05d}_max_alpha_LxH.png",
        vmin=0.0,
        vmax=1.0,
    )
    path_vn = _heatmap(
        sink_vnorm_ratio,
        f"‖v_sink‖ / mean‖v‖ (NOP≪1, broadcast~1) · ep={epoch}",
        f"ep{epoch:05d}_sink_vnorm_ratio_LxH.png",
    )

    scalars: Dict[str, float] = {
        "attn_sinks/mean_sink_rate": float(np.nanmean(sink_rate)),
        "attn_sinks/mean_max_alpha": float(np.nanmean(max_alpha)),
        "attn_sinks/mean_sink_vnorm_ratio": float(np.nanmean(sink_vnorm_ratio)),
        "attn_sinks/frac_soft_eps_heads": float(
            np.nanmean((max_alpha > epsilon).astype(np.float64))
        ),
        "attn_sinks/n_graph0_nodes": float(n_g),
        "attn_sinks/tau": float(tau),
        "attn_sinks/epsilon": float(epsilon),
    }
    for li, layer in enumerate(layers):
        for head in range(n_heads):
            if np.isnan(sink_rate[li, head]):
                continue
            scalars[f"attn_sinks/L{layer}_h{head}/sink_rate"] = float(sink_rate[li, head])
            scalars[f"attn_sinks/L{layer}_h{head}/max_alpha"] = float(max_alpha[li, head])
            if not np.isnan(sink_vnorm_ratio[li, head]):
                scalars[f"attn_sinks/L{layer}_h{head}/sink_vnorm_ratio"] = float(
                    sink_vnorm_ratio[li, head]
                )

    images = {
        "attn_sinks/panel_by_layer_head": str(path_heads),
        "attn_sinks/panel_mean_over_heads": str(path_mean),
        "attn_sinks/panel_sink_rate_LxH": str(path_sr),
        "attn_sinks/panel_max_alpha_LxH": str(path_ma),
        "attn_sinks/panel_sink_vnorm_ratio_LxH": str(path_vn),
    }
    return images, scalars


def maybe_log_attention_sinks_to_wandb(
    run: Any,
    model: nn.Module,
    train_loader: Iterable[Any],
    epoch: int,
) -> None:
    """If scheduled, dump attention panels to disk and log them to W&B.

    Args:
        run: Active ``wandb.Run`` (or None).
        model: Training model (``hybrid_gnn``).
        train_loader: Train loader (first batch used).
        epoch: Current epoch index.
    """
    global _AS_DIAG_LOGGED
    max_epoch = int(getattr(cfg.optim, "max_epoch", 1))
    if not should_log_attention_sinks(epoch, max_epoch):
        return
    if run is None:
        return

    hybrid = cfg.gnn.hybrid
    tau = float(getattr(hybrid, "attention_sink_tau", 1.5))
    epsilon = float(getattr(hybrid, "attention_sink_epsilon", 0.3))
    max_nodes = int(getattr(hybrid, "attention_sink_max_nodes", 512))

    device = torch.device(cfg.accelerator)
    core = _unwrap_model(model)
    if not hasattr(core, "collect_attention_maps"):
        logging.warning("Attention sinks: model has no collect_attention_maps; skip.")
        return

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            if not _is_graph_batch(sample_batch):
                logging.warning("Attention sinks: train batch is not a PyG graph; skip.")
                return
            sample_batch = sample_batch.to(device)
            # Truncate oversized batches for dense N×N safety.
            n = int(sample_batch.num_nodes)
            if n > max_nodes:
                # Keep first graphs until under budget.
                b = sample_batch.batch
                keep_graphs = []
                cum = 0
                for g in range(int(b.max().item()) + 1):
                    ng = int((b == g).sum().item())
                    if cum + ng > max_nodes:
                        break
                    keep_graphs.append(g)
                    cum += ng
                if not keep_graphs:
                    logging.warning(
                        "Attention sinks: single graph exceeds max_nodes=%d; skip.",
                        max_nodes,
                    )
                    return
                from torch_geometric.data import Batch as PyGBatch

                data_list = sample_batch.to_data_list()
                sample_batch = PyGBatch.from_data_list(
                    [data_list[g] for g in keep_graphs]
                ).to(device)

            payload = core.collect_attention_maps(sample_batch)
            if not payload.get("attention"):
                logging.warning("Attention sinks: empty attention dict; skip.")
                return

            out_dir = Path(cfg.run_dir) / "attention_sinks" / f"ep{epoch:05d}"
            images, scalars = _build_panels_for_batch(
                payload["attention"],
                payload.get("value_norms", {}),
                payload["edge_index"],
                payload["batch"],
                tau=tau,
                epsilon=epsilon,
                out_dir=out_dir,
                epoch=epoch,
            )

            # Persist the raw batch bundle for offline aggregate plots.
            if bool(getattr(hybrid, "attention_sink_save_pt", True)):
                torch.save(
                    {
                        "epoch": epoch,
                        "attention": payload["attention"],
                        "value_norms": payload.get("value_norms", {}),
                        "gate_means": payload.get("gate_means", {}),
                        "edge_index": payload["edge_index"],
                        "batch": payload["batch"],
                        "num_nodes": payload["num_nodes"],
                        "meta": {
                            "tau": tau,
                            "epsilon": epsilon,
                            "dataset": str(cfg.dataset.name),
                            "gate": str(getattr(hybrid, "gate", "")),
                            "mp_gate": str(getattr(hybrid, "mp_gate", "") or ""),
                        },
                    },
                    out_dir / f"attention_batch_ep{epoch:05d}.pt",
                )

            try:
                import wandb
            except ImportError:
                logging.warning("Attention sinks: wandb not installed; PNGs only.")
                return

            log_payload: Dict[str, Any] = dict(scalars)
            log_payload["train/epoch"] = float(epoch)
            for key, path in images.items():
                log_payload[key] = wandb.Image(path)
            run.log(log_payload, step=epoch)
            run.summary.update(scalars)

            if not _AS_DIAG_LOGGED:
                logging.info(
                    "Attention sinks: logging W&B panels every %d epochs "
                    "(tau=%.2f, eps=%.2f) → %s",
                    int(getattr(hybrid, "attention_sink_every", 50)),
                    tau,
                    epsilon,
                    out_dir.parent,
                )
                _AS_DIAG_LOGGED = True
            logging.info(
                "Attention sinks: epoch %d → %d PNGs under %s",
                epoch,
                len(images),
                out_dir,
            )
    except StopIteration:
        logging.warning("Attention sinks: empty train loader; skip.")
    except Exception:
        logging.exception("Attention sinks: failed at epoch %d", epoch)
    finally:
        if was_training:
            model.train()
