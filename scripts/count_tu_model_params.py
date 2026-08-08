#!/usr/bin/env python3
"""Count trainable params for TU GCN / SiGMA / GPS-style configs (CPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from yacs.config import CfgNode

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import GNNPlus  # noqa: F401


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=str, required=True, help="YAML config path.")
    p.add_argument("--dim-in", type=int, default=3, help="Dummy node feat dim.")
    p.add_argument("--dim-out", type=int, default=6, help="Dummy num classes.")
    p.add_argument("--n-nodes", type=int, default=20, help="Dummy graph size.")
    return p.parse_args(argv)


def _dummy_batch(n: int, dim_in: int) -> Batch:
    """Build a tiny batched graph for a dry forward (optional)."""
    x = torch.randn(n, dim_in)
    # chain edges
    src = torch.arange(0, n - 1)
    dst = torch.arange(1, n)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    data = Data(x=x, edge_index=ei, y=torch.tensor([0]))
    return Batch.from_data_list([data])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Load cfg, build model, print param count."""
    args = _parse(argv)
    set_cfg(cfg)
    # GraphGym expects argv-style cfg load
    opt = CfgNode({"cfg_file": args.cfg, "opts": []})
    load_cfg(cfg, opt)
    cfg.dataset.node_encoder = True
    cfg.accelerator = "cpu"
    # Force dims for create_model
    model = create_model(dim_in=args.dim_in, dim_out=args.dim_out)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"cfg={args.cfg}")
    print(f"params={n_params}")
    print(
        f"hybrid={getattr(cfg.gnn, 'hybrid', None) and dict(cfg.gnn.hybrid)}"
        if hasattr(cfg.gnn, "hybrid")
        else "no hybrid"
    )
    print(f"dim_inner={cfg.gnn.dim_inner} layers_mp={cfg.gnn.layers_mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
