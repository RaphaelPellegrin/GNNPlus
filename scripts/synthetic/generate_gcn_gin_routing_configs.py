#!/usr/bin/env python3
"""Generate GCN/GIN routing synthetic training YAML configs (8 models × 2 tracks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "configs" / "synthetic"

MODELS: tuple[dict[str, Any], ...] = (
    {
        "slug": "a0g2_gated",
        "num_gnn_heads": 2,
        "gnn_types_toy": "ROUTING_SUM,ROUTING_NORMGCN",
        "gnn_types_sigma": "GIN,GCN",
        "gate": "headwise",
        "mp_gate": "",
        "wandb_suffix": "a0g2_gated",
    },
    {
        "slug": "a0g2_ungated",
        "num_gnn_heads": 2,
        "gnn_types_toy": "ROUTING_SUM,ROUTING_NORMGCN",
        "gnn_types_sigma": "GIN,GCN",
        "gate": "none",
        "mp_gate": "none",
        "wandb_suffix": "a0g2_ungated",
    },
    {
        "slug": "a0g1_gcn",
        "num_gnn_heads": 1,
        "gnn_types_toy": "ROUTING_NORMGCN",
        "gnn_types_sigma": "GCN",
        "gate": "none",
        "mp_gate": "none",
        "wandb_suffix": "a0g1_gcn",
    },
    {
        "slug": "a0g1_gin",
        "num_gnn_heads": 1,
        "gnn_types_toy": "ROUTING_SUM",
        "gnn_types_sigma": "GIN",
        "gate": "none",
        "mp_gate": "none",
        "wandb_suffix": "a0g1_gin",
    },
)


def _build_cfg(
    track: str,
    model: dict[str, Any],
    *,
    node_encoder: bool = True,
    slug_suffix: str = "",
) -> dict[str, Any]:
    """Return one GraphGym config dict."""
    is_toy = track == "toy"
    d_h = 1 if is_toy else 4
    dim_inner = 2 if is_toy else 4
    gnn_types = model["gnn_types_toy"] if is_toy else model["gnn_types_sigma"]
    slug = f"{model['slug']}{slug_suffix}"
    hybrid: dict[str, Any] = {
        "num_attn_heads": 0,
        "num_gnn_heads": model["num_gnn_heads"],
        "d_h": d_h,
        "attn_mask": "full",
        "gate": model["gate"],
        "norm": "none",
        "gnn_types": gnn_types,
        "attn_dropout": 0.0,
        "mp_dropout": 0.0,
        "block_bn": False,
        "log_gate_stats": True,
        "residual": False,
        "identity_proj": False,
    }
    if model["mp_gate"]:
        hybrid["mp_gate"] = model["mp_gate"]

    return {
        "out_dir": "results",
        "metric_best": "accuracy",
        "wandb": {
            "use": False,
            "project": "GNNPlus",
            "tags": [
                "gnnplus",
                "hybrid_gnn",
                "gcn_gin_routing_synthetic",
                f"gcn_gin_routing_{track}",
                f"gcn_gin_routing_{track}_{slug}",
            ],
        },
        "dataset": {
            "format": "PyG-GcnGinRouting",
            "name": "v1",
            "task": "graph",
            "task_type": "classification",
            "transductive": False,
            "split_mode": "standard",
            "node_encoder": node_encoder,
            "node_encoder_name": "LinearNode",
            "node_encoder_bn": False,
            "edge_encoder": False,
        },
        "train": {
            "mode": "custom",
            "batch_size": 128,
            "eval_period": 1,
            "ckpt_period": 50,
        },
        "model": {
            "type": "hybrid_gnn",
            "loss_fun": "cross_entropy",
            "graph_pooling": "graph_token",
        },
        "gnn": {
            "head": "default",
            "ffn": False,
            "layers_pre_mp": 0,
            "layers_mp": 1,
            "layers_post_mp": 1,
            "dim_inner": dim_inner,
            "act": "relu",
            "residual": False,
            "dropout": 0.0,
            "hybrid": hybrid,
        },
        "optim": {
            "clip_grad_norm": False,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "base_lr": 0.001,
            "max_epoch": 200,
            "scheduler": "cosine_with_warmup",
            "num_warmup_epochs": 5,
        },
    }


def main() -> None:
    """Write YAML configs under ``configs/synthetic/``."""
    import sys

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_noxenc = "--noxenc" in sys.argv
    for track in ("toy", "sigma"):
        for model in MODELS:
            slug = model["slug"]
            path = OUT_DIR / f"gcn_gin_routing_{track}_{slug}.yaml"
            cfg = _build_cfg(track, model)
            with path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            print(f"Wrote {path}")

    if write_noxenc:
        for model in MODELS:
            slug = model["slug"]
            path = OUT_DIR / f"gcn_gin_routing_toy_{slug}_noxenc.yaml"
            cfg = _build_cfg("toy", model, node_encoder=False, slug_suffix="_noxenc")
            with path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
