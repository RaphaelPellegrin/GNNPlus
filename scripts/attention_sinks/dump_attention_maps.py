#!/usr/bin/env python3
"""Dump dense attention maps from a SiGMA / GPS hybrid checkpoint.

Writes Heterogeneity_Profile-compatible ``.pt`` bundles::

    {
      "epoch": int,
      "attention": {"layer{L}_attn{H}": FloatTensor[N, N], ...},
      "value_norms": {"layer{L}_attn{H}": FloatTensor[N], ...},  # ‖v_j‖₂
      "gate_means": {"layer{L}_attn{H}": float, ...},
      "edge_index": LongTensor[2, E],
      "batch": LongTensor[N],
      "num_nodes": int,
      "meta": {...},
    }

Use small ``--batch_size`` (default 8) so dense ``N×N`` stays tractable.
MUTAG / ENZYMES are good; COLLAB / REDDIT are not.

Example (local, after rsync ckpt)::

  python scripts/attention_sinks/dump_attention_maps.py \\
    --run_dir results/tu_attention_sinks/mutag_SiGMA_hetero_ungated_attn_lr001_seed2 \\
    --cfg configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-matched-anchor.yaml \\
    --splits train,val,test --batch_size 8
"""

from __future__ import annotations

import argparse
import logging
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch_geometric.graphgym.checkpoint import get_ckpt_epochs, load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401
from GNNPlus.hybrid_gate_tracking import _unwrap_model


def _parse_dump_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse attention-dump CLI (GraphGym ``--cfg`` via remaining argv)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (default: <run_dir>/attention_matrices).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to dump.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Loader batch size (keep small for dense N×N).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Max batches per split (-1 = all).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Filename prefix (default: sanitized run dir name).",
    )
    args, remaining = parser.parse_known_args(argv)
    args.gg_argv = remaining
    return args


def _pick_epoch(run_dir: str, epoch: int) -> int:
    """Resolve checkpoint epoch (-1 → latest available)."""
    if epoch >= 0:
        return epoch
    epochs = get_ckpt_epochs(run_dir)
    if not epochs:
        raise FileNotFoundError(f"No checkpoints under {run_dir}/ckpt/")
    return int(max(epochs))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load ckpt and write attention ``.pt`` bundles."""
    dump_args = _parse_dump_args(argv)
    gg_argv: List[str] = list(dump_args.gg_argv)
    if "--cfg" not in gg_argv and "-cfg" not in gg_argv:
        raise SystemExit("Pass GraphGym --cfg <yaml> (same config as training).")

    old_argv = sys.argv
    sys.argv = [old_argv[0], *gg_argv]
    try:
        args = parse_args()
        set_cfg(cfg)
        load_cfg(cfg, args)
    finally:
        sys.argv = old_argv

    run_dir = osp.abspath(dump_args.run_dir)
    cfg.run_dir = run_dir
    cfg.out_dir = osp.dirname(run_dir) or cfg.out_dir
    cfg.train.batch_size = int(dump_args.batch_size)
    seed_everything(int(cfg.seed))
    auto_select_device()

    loaders = create_loader()
    model = create_model()
    core = _unwrap_model(model)
    if not hasattr(core, "collect_attention_maps"):
        raise SystemExit(
            f"Model {type(core).__name__} has no collect_attention_maps "
            "(need hybrid_gnn with vanilla attention)."
        )

    epoch = _pick_epoch(run_dir, int(dump_args.epoch))
    loaded = load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    logging.info("Loaded checkpoint epoch=%s (start_epoch=%s)", epoch, loaded)
    model.eval()

    out_dir = Path(dump_args.out_dir) if dump_args.out_dir else Path(run_dir) / "attention_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = dump_args.tag.strip() or Path(run_dir).name.replace("/", "_")

    name_to_id = {"train": 0, "val": 1, "test": 2}
    wanted = [s.strip().lower() for s in dump_args.splits.split(",") if s.strip()]
    loader_by_name = {
        "train": loaders[0] if len(loaders) > 0 else None,
        "val": loaders[1] if len(loaders) > 1 else None,
        "test": loaders[2] if len(loaders) > 2 else None,
    }

    gate = str(getattr(cfg.gnn.hybrid, "gate", "?"))
    mp_gate = str(getattr(cfg.gnn.hybrid, "mp_gate", "") or gate)
    attn_mask = str(getattr(cfg.gnn.hybrid, "attn_mask", "full"))
    meta_base: Dict[str, Any] = {
        "run_dir": run_dir,
        "epoch": epoch,
        "seed": int(cfg.seed),
        "dataset": str(cfg.dataset.name),
        "gate": gate,
        "mp_gate": mp_gate,
        "attn_mask": attn_mask,
        "num_attn_heads": int(cfg.gnn.hybrid.num_attn_heads),
        "num_gnn_heads": int(cfg.gnn.hybrid.num_gnn_heads),
        "layers_mp": int(cfg.gnn.layers_mp),
        "d_h": int(cfg.gnn.hybrid.d_h),
        "gnn_types": str(getattr(cfg.gnn.hybrid, "gnn_types", "")),
    }

    device = torch.device(cfg.accelerator)
    model.to(device)

    with torch.no_grad():
        for split in wanted:
            if split not in name_to_id:
                raise ValueError(f"Unknown split {split!r}")
            loader = loader_by_name[split]
            if loader is None:
                logging.warning("Skip missing split %s", split)
                continue
            n_saved = 0
            for bi, batch in enumerate(loader):
                if dump_args.max_batches >= 0 and bi >= dump_args.max_batches:
                    break
                batch = batch.to(device)
                payload = core.collect_attention_maps(batch)
                if not payload["attention"]:
                    raise RuntimeError(
                        "No attention maps returned — is attn_type grit / num_attn_heads=0?"
                    )
                out_path = out_dir / f"{tag}_{split}_batch{bi:04d}_epoch{epoch:05d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "split": split,
                        "batch_index": bi,
                        "attention": payload["attention"],
                        "value_norms": payload["value_norms"],
                        "gate_means": payload["gate_means"],
                        "edge_index": payload["edge_index"],
                        "batch": payload["batch"],
                        "num_nodes": payload["num_nodes"],
                        "y": payload["y"],
                        "meta": meta_base,
                    },
                    out_path,
                )
                n_saved += 1
                logging.info(
                    "Wrote %s (nodes=%d, heads=%d)",
                    out_path,
                    payload["num_nodes"],
                    len(payload["attention"]),
                )
            logging.info("Split %s: saved %d attention bundles → %s", split, n_saved, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
