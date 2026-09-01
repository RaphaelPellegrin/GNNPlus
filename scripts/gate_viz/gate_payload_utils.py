"""Shared helpers for loading SiGMA gate ``.pt`` dumps."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import torch

SplitName = Literal["train", "val", "test"]
ColorByField = Literal["none", "y", "tau", "tau_y"]

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")


def normalize_node_gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept TU dumps (``attn``/``gnn``) and routing dumps (``*_node`` keys)."""
    out = dict(payload)
    if "attn" not in out and "attn_node" in out:
        out["attn"] = out["attn_node"]
    if "gnn" not in out and "gnn_node" in out:
        out["gnn"] = out["gnn_node"]
    return out


def _split_mask(split_t: torch.Tensor, split: SplitName | int) -> torch.Tensor:
    """Boolean mask over graphs for one split."""
    if isinstance(split, str):
        split_id = SPLIT_NAMES.index(split.lower())  # type: ignore[arg-type]
    else:
        split_id = int(split)
    return split_t.view(-1).long() == split_id


def subset_payload_by_split(
    payload: dict[str, Any],
    split: SplitName | int,
) -> dict[str, Any]:
    """Keep only graphs belonging to ``split`` (requires ``ptr`` + graph-level fields)."""
    split_t = payload.get("split")
    if split_t is None:
        raise KeyError("Payload missing graph-level 'split' tensor.")
    mask = _split_mask(split_t, split)
    if not bool(mask.any()):
        raise ValueError(f"No graphs for split={split!r}.")

    ptr: torch.Tensor = payload["ptr"].long()
    graph_ids = torch.nonzero(mask, as_tuple=False).view(-1)
    node_slices: list[torch.Tensor] = []
    new_ptr = [0]
    node_offset = 0

    out: dict[str, Any] = {}
    for graph_idx in graph_ids.tolist():
        lo = int(ptr[graph_idx].item())
        hi = int(ptr[graph_idx + 1].item())
        node_slices.append(payload["gnn"][lo:hi])
        node_offset += hi - lo
        new_ptr.append(node_offset)

    out["gnn"] = torch.cat(node_slices, dim=0) if node_slices else payload["gnn"][:0]
    if "attn" in payload:
        attn_slices = []
        for graph_idx in graph_ids.tolist():
            lo = int(ptr[graph_idx].item())
            hi = int(ptr[graph_idx + 1].item())
            attn_slices.append(payload["attn"][lo:hi])
        out["attn"] = torch.cat(attn_slices, dim=0) if attn_slices else payload["attn"][:0]

    out["ptr"] = torch.tensor(new_ptr, dtype=torch.long)
    out["num_graphs"] = int(graph_ids.numel())
    out["num_nodes"] = int(out["gnn"].size(0))

    for key in ("tau", "y", "split"):
        if key in payload and payload[key] is not None:
            out[key] = payload[key][mask]

    if "batch" in payload and payload["batch"] is not None:
        batch_parts: list[torch.Tensor] = []
        for new_g, old_g in enumerate(graph_ids.tolist()):
            lo = int(ptr[old_g].item())
            hi = int(ptr[old_g + 1].item())
            batch_parts.append(
                torch.full((hi - lo,), new_g, dtype=payload["batch"].dtype),
            )
        out["batch"] = torch.cat(batch_parts, dim=0) if batch_parts else payload["batch"][:0]

    if "edge_index" in payload and "edge_ptr" in payload:
        edge_ptr: torch.Tensor = payload["edge_ptr"].long()
        edge_parts: list[torch.Tensor] = []
        edge_batch_parts: list[torch.Tensor] = []
        node_base = 0
        for new_g, old_g in enumerate(graph_ids.tolist()):
            n_lo = int(ptr[old_g].item())
            n_hi = int(ptr[old_g + 1].item())
            e_lo = int(edge_ptr[old_g].item())
            e_hi = int(edge_ptr[old_g + 1].item())
            if e_hi > e_lo:
                ei = payload["edge_index"][:, e_lo:e_hi].clone()
                ei = ei - int(n_lo) + node_base
                edge_parts.append(ei)
                edge_batch_parts.append(
                    torch.full((e_hi - e_lo,), new_g, dtype=torch.long),
                )
            node_base += n_hi - n_lo
        if edge_parts:
            out["edge_index"] = torch.cat(edge_parts, dim=1)
            edge_batch = torch.cat(edge_batch_parts, dim=0)
            out["edge_batch"] = edge_batch
            counts = torch.bincount(edge_batch, minlength=int(graph_ids.numel()))
            ep = [0]
            for c in counts.tolist():
                ep.append(ep[-1] + int(c))
            out["edge_ptr"] = torch.tensor(ep, dtype=torch.long)
            out["num_edges"] = int(out["edge_index"].size(1))
        else:
            out["edge_index"] = payload["edge_index"][:0]
            out["edge_batch"] = payload["edge_batch"][:0]
            out["edge_ptr"] = torch.zeros(int(graph_ids.numel()) + 1, dtype=torch.long)
            out["num_edges"] = 0

    for key in ("meta", "gin_head_idx", "gcn_head_idx", "layer_idx", "root_local_idx"):
        if key in payload:
            out[key] = payload[key]
    return out


def graph_means_from_nodes(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    n_graphs: int,
) -> np.ndarray:
    """Mean-pool node gates ``[N, L, H]`` → ``[G, L, H]``."""
    n_layers, n_heads = int(node_vals.shape[1]), int(node_vals.shape[2])
    out = np.zeros((n_graphs, n_layers, n_heads), dtype=np.float64)
    for g in range(n_graphs):
        lo, hi = int(ptr[g]), int(ptr[g + 1])
        if hi > lo:
            out[g] = node_vals[lo:hi].mean(axis=0)
    return out


def root_gates_from_nodes(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    n_graphs: int,
) -> np.ndarray:
    """Root node gates ``[N, L, H]`` → ``[G, L, H]`` (first node per graph)."""
    n_layers, n_heads = int(node_vals.shape[1]), int(node_vals.shape[2])
    out = np.zeros((n_graphs, n_layers, n_heads), dtype=np.float64)
    for g in range(n_graphs):
        lo = int(ptr[g])
        if int(ptr[g + 1]) > lo:
            out[g] = node_vals[lo]
    return out


def load_graph_level_gates(
    pt_path: str,
    *,
    split: Optional[SplitName] = None,
    aggregation: Literal["mean", "root"] = "mean",
) -> dict[str, Any]:
    """Load a node dump and return graph-level ``attn`` / ``gnn`` arrays."""
    payload = normalize_node_gate_payload(
        torch.load(pt_path, map_location="cpu", weights_only=False),
    )
    if split is not None:
        payload = subset_payload_by_split(payload, split)

    ptr = payload["ptr"].detach().cpu().long().numpy()
    n_graphs = int(payload.get("num_graphs", ptr.size - 1))
    gnn_n = payload["gnn"].detach().cpu().float().numpy()
    attn_n = payload["attn"].detach().cpu().float().numpy()

    pool = graph_means_from_nodes if aggregation == "mean" else root_gates_from_nodes
    gnn_g = pool(gnn_n, ptr, n_graphs)
    attn_g = pool(attn_n, ptr, n_graphs)

    result: dict[str, Any] = {
        "attn": attn_g,
        "gnn": gnn_g,
        "meta": dict(payload.get("meta") or {}),
        "num_graphs": n_graphs,
    }
    for key in ("y", "tau", "split"):
        if key in payload and payload[key] is not None:
            result[key] = payload[key].detach().cpu().numpy().astype(np.int64).reshape(-1)
    return result


def tau_y_color_map() -> dict[tuple[int, int], str]:
    """Four distinct colors for ``(tau, y)`` combinations."""
    return {
        (0, 0): "#4C72B0",
        (0, 1): "#55A868",
        (1, 0): "#DD8452",
        (1, 1): "#C44E52",
    }


def tau_y_labels() -> dict[tuple[int, int], str]:
    """Legend labels for ``(tau, y)`` combinations."""
    return {
        (0, 0): r"$\tau{=}0$, $y{=}0$",
        (0, 1): r"$\tau{=}0$, $y{=}1$",
        (1, 0): r"$\tau{=}1$, $y{=}0$",
        (1, 1): r"$\tau{=}1$, $y{=}1$",
    }
