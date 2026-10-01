"""Graph readout MLP presets (Heterogeneity_Profile ``hybrid_readout_mlp``)."""

from __future__ import annotations

import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg


def build_readout_mlp(dim_in: int, dim_out: int, preset: str) -> nn.Sequential:
    """Build a pooled-graph readout MLP.

    Presets:
        - ``linear``: ``dim_in -> dim_out``
        - ``narrow2``: ``dim_in -> dim_in//2 -> dim_out`` (one hidden)
        - ``pyramid``: ``dim_in -> dim_in//2 -> dim_in//4 -> dim_out`` (default in MOE)
        - ``deep4``: ``dim_in -> dim_in//2 -> dim_in//4 -> dim_in//8 -> dim_out``

    Empty preset or ``mlp_graph`` falls back to legacy ``MLPGraphHead`` stacking.
    """
    def _h(x: int) -> int:
        return max(1, int(x))

    act = register.act_dict[cfg.gnn.act]()
    dropout = float(cfg.gnn.dropout)
    p = str(preset).lower().strip()

    if p in ('pyramid', 'default'):
        h1, h2 = _h(dim_in // 2), _h(dim_in // 4)
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, h1, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(h1, h2, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(h2, int(dim_out), bias=True),
        )
    if p in ('linear', '0h'):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, int(dim_out), bias=True),
        )
    if p in ('narrow2', '2l', '1h'):
        h1 = _h(dim_in // 2)
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, h1, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(h1, int(dim_out), bias=True),
        )
    if p in ('deep4', '4h'):
        h1, h2, h3 = _h(dim_in // 2), _h(dim_in // 4), _h(dim_in // 8)
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, h1, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(h1, h2, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(h2, h3, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(h3, int(dim_out), bias=True),
        )
    raise ValueError(
        'gnn.readout_mlp must be linear, narrow2, pyramid, deep4, or empty '
        f'(got {preset!r})'
    )
