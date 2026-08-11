"""Shared helpers for PDE / Transolver-style mesh datasets."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


def relative_l2(pred: Tensor, true: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean relative L2 over leading batch / graph examples.

    Args:
        pred: Predictions ``[N, ...]`` or ``[B, N, ...]``.
        true: Targets with the same shape.
        eps: Stability term on the denominator.

    Returns:
        Scalar relative L2 (mean over the leading dimension when batched).
    """
    pred_f = pred.reshape(pred.size(0), -1).float()
    true_f = true.reshape(true.size(0), -1).float()
    diff = torch.norm(pred_f - true_f, p=2, dim=1)
    denom = torch.norm(true_f, p=2, dim=1).clamp_min(eps)
    return (diff / denom).mean()


def build_knn_edges(
    pos: Tensor,
    k: int = 8,
    *,
    undirected: bool = True,
) -> Tensor:
    """Build a kNN edge index from node positions.

    Uses ``torch_cluster`` via PyG when available; otherwise a dense
    pairwise distance fallback (fine for moderate ``N``).

    Args:
        pos: Coordinates ``[N, D]``.
        k: Number of neighbors.
        undirected: If True, make the graph undirected.

    Returns:
        Edge index ``[2, E]``.
    """
    if pos.dim() != 2:
        raise ValueError(f"pos must be [N, D], got {tuple(pos.shape)}")
    n_nodes = int(pos.size(0))
    k_eff = max(1, min(int(k), max(1, n_nodes - 1)))
    try:
        edge_index = knn_graph(pos, k=k_eff, loop=False)
    except (AttributeError, ImportError, RuntimeError, ModuleNotFoundError):
        # Pure-torch fallback when torch_cluster is missing.
        dist = torch.cdist(pos.float(), pos.float())
        dist.fill_diagonal_(float("inf"))
        knn = dist.topk(k_eff, largest=False).indices  # [N, k]
        src = torch.arange(n_nodes, device=pos.device).unsqueeze(1).expand_as(knn)
        edge_index = torch.stack([src.reshape(-1), knn.reshape(-1)], dim=0)
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=n_nodes)
    return edge_index


def make_mesh_data(
    *,
    x: Tensor,
    y: Tensor,
    pos: Tensor,
    knn_k: int = 8,
    edge_index: Optional[Tensor] = None,
) -> Data:
    """Assemble a PyG ``Data`` object for a PDE mesh sample.

    Args:
        x: Node input features ``[N, F_in]``.
        y: Node targets ``[N, F_out]`` (or ``[N]``).
        pos: Coordinates ``[N, D]``.
        knn_k: Neighbors for kNN when ``edge_index`` is None.
        edge_index: Optional precomputed edges.

    Returns:
        PyG ``Data`` with ``x``, ``y``, ``pos``, ``edge_index``.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    ei = edge_index if edge_index is not None else build_knn_edges(pos, k=knn_k)
    return Data(x=x.float(), y=y.float(), pos=pos.float(), edge_index=ei)


class UnitGaussianNormalizer:
    """Per-feature mean/std normalizer (FNO / Transolver style)."""

    def __init__(self, x: Tensor, eps: float = 1e-8) -> None:
        """Fit mean/std on ``x`` with shape ``[..., C]`` or ``[...,]``."""
        flat = x.reshape(-1, x.size(-1) if x.dim() > 1 else 1).float()
        if x.dim() == 1:
            flat = x.reshape(-1, 1).float()
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp_min(eps)
        self.eps = float(eps)

    def encode(self, x: Tensor) -> Tensor:
        """Normalize ``x``."""
        if x.dim() == 1:
            return ((x.unsqueeze(-1) - self.mean) / self.std).squeeze(-1)
        return (x - self.mean) / self.std

    def decode(self, x: Tensor) -> Tensor:
        """Denormalize ``x``."""
        if x.dim() == 1:
            return (x.unsqueeze(-1) * self.std + self.mean).squeeze(-1)
        return x * self.std + self.mean

    def state_dict(self) -> dict[str, Tensor]:
        """Serialize normalizer tensors."""
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load normalizer tensors."""
        self.mean = state["mean"]
        self.std = state["std"]


def train_val_test_index_split(
    n_total: int,
    n_train: int,
    n_test: int,
    n_val: Optional[int] = None,
) -> Tuple[list[int], list[int], list[int]]:
    """Contiguous Transolver-style index splits.

    Train = ``[0, n_train)``, test = last ``n_test``, val = remainder
    (or a middle block of size ``n_val`` when provided).
    """
    if n_train + n_test > n_total:
        raise ValueError(
            f"n_train ({n_train}) + n_test ({n_test}) > n_total ({n_total})"
        )
    train_idx = list(range(n_train))
    test_idx = list(range(n_total - n_test, n_total))
    middle = [i for i in range(n_total) if i not in set(train_idx + test_idx)]
    if n_val is None:
        val_idx = middle if middle else train_idx[-max(1, n_train // 10) :]
    else:
        val_idx = middle[:n_val] if middle else train_idx[-n_val:]
    return train_idx, list(val_idx), test_idx
