"""Dummy edge encoder (optional RWSE edge features when dims match)."""

from __future__ import annotations

import logging

import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder

logger = logging.getLogger(__name__)


@register_edge_encoder("DummyEdge")
class DummyEdgeEncoder(torch.nn.Module):
    """Embed a constant edge type; optionally add RWSE edge features.

    When ``node_encoder_name`` is ``LinearNode+RWSE``, historical GPS-style
    pipelines store RWSE landing probs on ``edge_attr`` with width
    ``max(times)``. If a stale cache or another PE left a different width
    (e.g. RRWP ``ksteps+identity``), ``view(-1, in_dim)`` used to crash.
    In that case we fall back to the constant embedding only.
    """

    def __init__(self, emb_dim: int) -> None:
        """Build the constant embedding and optional RWSE linear map.

        Args:
            emb_dim: Output edge embedding width (usually ``cfg.gnn.dim_inner``).
        """
        super().__init__()
        self.encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
        self.in_dim: int | None = None
        self.encoder2: torch.nn.Linear | None = None
        if cfg.dataset.node_encoder_name == "LinearNode+RWSE":
            self.in_dim = int(list(eval(cfg.posenc_RWSE.kernel.times_func))[-1])
            self.encoder2 = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch: object) -> object:
        """Write ``batch.edge_attr`` as ``[E, emb_dim]``."""
        edge_index = batch.edge_index
        dummy_attr = edge_index.new_zeros(edge_index.shape[1])
        emb = self.encoder(dummy_attr)

        ea = getattr(batch, "edge_attr", None)
        if self.encoder2 is None or self.in_dim is None or ea is None:
            batch.edge_attr = emb
            return batch

        if ea.dim() == 1:
            ea = ea.unsqueeze(-1)

        n_edges = int(edge_index.size(1))
        if ea.size(0) == n_edges and int(ea.size(-1)) == int(self.in_dim):
            batch.edge_attr = emb + self.encoder2(ea)
            return batch

        if ea.numel() == n_edges * int(self.in_dim):
            batch.edge_attr = emb + self.encoder2(ea.view(n_edges, self.in_dim))
            return batch

        logger.warning(
            "DummyEdge: edge_attr shape %s incompatible with RWSE in_dim=%s "
            "(n_edges=%s); using constant embedding only.",
            tuple(ea.shape),
            self.in_dim,
            n_edges,
        )
        batch.edge_attr = emb
        return batch
