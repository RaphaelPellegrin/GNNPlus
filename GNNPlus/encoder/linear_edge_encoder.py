from typing import Any

import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


def _rwse_times_in_dim() -> int:
    """Infer edge input dim from ``posenc_RWSE.kernel.times_func`` (GraphGym hack).

    Returns:
        Last value of the evaluated times sequence (used as ``in_dim``).

    Raises:
        ValueError: If ``times_func`` is missing/empty (``eval('')`` would
            raise ``SyntaxError``). Prefer ``dataset.edge_encoder: False``
            when the dataset has no edge features.
    """
    times_func = str(
        getattr(cfg.posenc_RWSE.kernel, "times_func", "") or ""
    ).strip()
    if not times_func:
        raise ValueError(
            "LinearEdgeEncoder requires non-empty "
            "posenc_RWSE.kernel.times_func to set in_dim. "
            "For datasets without edge features / RWSE, set "
            "dataset.edge_encoder: False."
        )
    return int(list(eval(times_func))[-1])


@register_edge_encoder("LinearEdge")
class LinearEdgeEncoder(torch.nn.Module):
    """Linear projection of raw edge features into ``emb_dim``."""

    def __init__(self, emb_dim: int) -> None:
        """Build a linear edge encoder.

        Args:
            emb_dim: Output embedding dimension.
        """
        super().__init__()
        if cfg.dataset.name in ["MNIST", "CIFAR10"]:
            if cfg.dataset.node_encoder_name == "LinearNode":
                self.in_dim = 1
            else:
                self.in_dim = _rwse_times_in_dim() + 1
        else:
            self.in_dim = _rwse_times_in_dim()
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch: Any) -> Any:
        """Encode ``batch.edge_attr`` in-place and return ``batch``."""
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
