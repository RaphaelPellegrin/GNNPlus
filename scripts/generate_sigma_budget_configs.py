#!/usr/bin/env python3
"""Generate SiGMA baby/tiny budget configs from paper anchors.

Only creates configs for (dataset, budget) where the main Table III/IV
SiGMA exceeds that budget. Existing multi-seed alts under budget are
documented in Paper_sigma_budget.md (not re-generated here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "gated_hybrid" / "budget"


def _load(rel: str) -> dict[str, Any]:
    """Load an anchor YAML as a dict."""
    path = ROOT / rel
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _dump(name: str, cfg: dict[str, Any], comment: str) -> Path:
    """Write a budget config with a header comment."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    path.write_text(f"# {comment}\n{body}")
    return path


def _retag(cfg: dict[str, Any], *tags: str) -> None:
    """Replace wandb tags/group for budget campaign."""
    wb = cfg.setdefault("wandb", {})
    wb["use"] = False
    wb["project"] = "GNNPlus"
    wb["group"] = tags[0]
    wb["tags"] = ["gnnplus", "hybrid_gnn", "sigma_budget", *tags]


def _set_hybrid(
    cfg: dict[str, Any],
    *,
    n_attn: int,
    n_gnn: int,
    d_h: int,
    gnn_types: str,
    dim_inner: int | None = None,
    layers_mp: int | None = None,
) -> None:
    """Apply head/width/depth overrides."""
    gnn = cfg.setdefault("gnn", {})
    hy = gnn.setdefault("hybrid", {})
    hy["num_attn_heads"] = n_attn
    hy["num_gnn_heads"] = n_gnn
    hy["d_h"] = d_h
    hy["gnn_types"] = gnn_types
    hy["log_gate_stats"] = True
    if dim_inner is not None:
        gnn["dim_inner"] = dim_inner
    if layers_mp is not None:
        gnn["layers_mp"] = layers_mp


def main() -> None:
    """Emit all budget YAMLs."""
    written: list[Path] = []

    # --- MNIST ≤500k (main 965k a2g2 → a1g1, smaller d_h) ---
    cfg = _load("configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml")
    _set_hybrid(cfg, n_attn=1, n_gnn=1, d_h=32, gnn_types="GATEDGCN", dim_inner=48)
    _retag(cfg, "paper_budget_mnist_b500k", "mnist", "b500k", "hybrid_a1g1")
    written.append(
        _dump(
            "mnist-b500k-a1g1.yaml",
            cfg,
            "MNIST baby ≤500k from lcvbyyss a2g2 → a1g1 GATEDGCN, H=48, d_h=32.",
        )
    )

    # --- CIFAR ≤500k / ≤1M / ≤2M (main 27.8M a8g4 d_h=256) ---
    # Widths recounted to fit budgets (prior H35/dh64, H48/dh96, H56/dh96 overshot).
    for budget, n_attn, n_gnn, d_h, h, l, tag in (
        ("b500k", 1, 1, 52, 66, 10, "a1g1"),  # ~498.8k
        ("b1m", 1, 1, 76, 86, 10, "a1g1"),  # ~998.9k
        ("b2m", 1, 2, 84, 82, 10, "a1g2"),  # ~1.998M
    ):
        cfg = _load("configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml")
        types = "GATEDGCN" if n_gnn == 1 else "GATEDGCN,GATEDGCN"
        _set_hybrid(
            cfg,
            n_attn=n_attn,
            n_gnn=n_gnn,
            d_h=d_h,
            gnn_types=types,
            dim_inner=h,
            layers_mp=l,
        )
        _retag(
            cfg,
            f"paper_budget_cifar10_{budget}_fit",
            "cifar10",
            budget,
            f"hybrid_{tag}",
            "params_fit",
        )
        written.append(
            _dump(
                f"cifar10-{budget}-{tag}.yaml",
                cfg,
                f"CIFAR10 {budget} from ulij45a2 a8g4 → {tag}, H={h}, d_h={d_h} (params fit).",
            )
        )

    # --- PATTERN ≤500k / ≤1M (main 1.99M a2g2 GRIT) ---
    for budget, n_attn, n_gnn, d_h, h, l, tag in (
        ("b500k", 1, 1, 48, 48, 10, "a1g1"),
        ("b1m", 1, 1, 64, 64, 12, "a1g1"),
    ):
        cfg = _load(
            "configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
        )
        _set_hybrid(
            cfg,
            n_attn=n_attn,
            n_gnn=n_gnn,
            d_h=d_h,
            gnn_types="GCNE",
            dim_inner=h,
            layers_mp=l,
        )
        _retag(cfg, f"paper_budget_pattern_{budget}", "pattern", budget, f"hybrid_{tag}")
        written.append(
            _dump(
                f"pattern-{budget}-{tag}-grit.yaml",
                cfg,
                f"PATTERN {budget} from ta9qtxb9 a2g2 → {tag} GCNE+GRIT, H={h}, d_h={d_h}.",
            )
        )

    # --- CLUSTER ≤500k / ≤1M (main 1.03M a1g1 — shrink width) ---
    for budget, d_h, h, l in (
        ("b500k", 32, 40, 12),
        ("b1m", 48, 48, 14),
    ):
        cfg = _load("configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml")
        _set_hybrid(
            cfg,
            n_attn=1,
            n_gnn=1,
            d_h=d_h,
            gnn_types="GATEDGCN",
            dim_inner=h,
            layers_mp=l,
        )
        _retag(cfg, f"paper_budget_cluster_{budget}", "cluster", budget, "hybrid_a1g1")
        written.append(
            _dump(
                f"cluster-{budget}-a1g1.yaml",
                cfg,
                f"CLUSTER {budget} from ht9bntg2 a1g1 → H={h}, d_h={d_h}, L={l}.",
            )
        )

    # --- Peptides-struct ≤500k only (1M/2M: reuse rholn782) ---
    cfg = _load(
        "configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
    )
    _set_hybrid(
        cfg,
        n_attn=1,
        n_gnn=1,
        d_h=64,
        gnn_types="GINE",
        dim_inner=64,
        layers_mp=4,
    )
    _retag(
        cfg,
        "paper_budget_peptides_struct_b500k",
        "peptides_struct",
        "b500k",
        "hybrid_a1g1",
    )
    written.append(
        _dump(
            "peptides-struct-b500k-a1g1.yaml",
            cfg,
            "Peptides-struct baby ≤500k from g3bsaq32 → a1g1 GINE H=64 d_h=64 L=4.",
        )
    )

    # --- VOC ≤500k / ≤1M / ≤2M (main 3.19M a2g2) ---
    for budget, n_attn, n_gnn, d_h, h, l, tag in (
        ("b500k", 1, 1, 32, 36, 8, "a1g1"),
        ("b1m", 1, 1, 40, 48, 10, "a1g1"),
        ("b2m", 1, 1, 48, 64, 12, "a1g1"),
    ):
        cfg = _load("configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml")
        _set_hybrid(
            cfg,
            n_attn=n_attn,
            n_gnn=n_gnn,
            d_h=d_h,
            gnn_types="GATEDGCN",
            dim_inner=h,
            layers_mp=l,
        )
        _retag(cfg, f"paper_budget_voc_{budget}", "voc", budget, f"hybrid_{tag}")
        written.append(
            _dump(
                f"voc-{budget}-{tag}.yaml",
                cfg,
                f"PascalVOC-SP {budget} from j7ukyzdm a2g2 → {tag}, H={h}, d_h={d_h}, L={l}.",
            )
        )

    # --- COCO ≤500k (main 898k a1g1 — shrink H/d_h) ---
    cfg = _load("configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml")
    _set_hybrid(
        cfg,
        n_attn=1,
        n_gnn=1,
        d_h=36,
        gnn_types="GATEDGCN",
        dim_inner=36,
        layers_mp=12,
    )
    _retag(cfg, "paper_budget_coco_b500k", "coco", "b500k", "hybrid_a1g1")
    written.append(
        _dump(
            "coco-b500k-a1g1.yaml",
            cfg,
            "COCO-SP baby ≤500k from 5b4z9l3u a1g1 → H=36 d_h=36 L=12.",
        )
    )

    # --- MalNet ≤500k (main 549k a1g1 vcb1cuql — slight shrink) ---
    # Paper run figmqani: node_encoder=False (LDP feats), layers_pre_mp=1,
    # batchnorm=True, a1g1 GCNE H=110 d_h=64, graph_restricted, ep=150.
    cfg = _load("configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml")
    cfg["dataset"]["node_encoder"] = False
    cfg["dataset"]["node_encoder_name"] = "Atom"
    gnn = cfg.setdefault("gnn", {})
    gnn["layers_pre_mp"] = 1
    gnn["batchnorm"] = True
    hy = gnn.setdefault("hybrid", {})
    hy["attn_mask"] = "graph_restricted"
    hy["gate"] = "elementwise"
    hy["norm"] = "layernorm"
    hy["attn_dropout"] = 0.0
    hy["mp_dropout"] = 0.0
    _set_hybrid(
        cfg,
        n_attn=1,
        n_gnn=1,
        d_h=56,
        gnn_types="GCNE",
        dim_inner=96,
        layers_mp=8,
    )
    cfg["optim"]["base_lr"] = 0.000371
    cfg["optim"]["max_epoch"] = 150
    _retag(cfg, "paper_budget_malnet_b500k", "malnet", "b500k", "hybrid_a1g1")
    written.append(
        _dump(
            "malnet-b500k-a1g1.yaml",
            cfg,
            "MalNet-Tiny baby ≤500k from vcb1cuql a1g1 → H=96 d_h=56 L=8 GCNE "
            "(node_encoder=False like paper).",
        )
    )

    print(f"Wrote {len(written)} configs under {OUT}:")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
