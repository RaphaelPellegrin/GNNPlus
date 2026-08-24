#!/usr/bin/env python3
"""Generate SiGMA d_h-matched configs for Tab. 3/4 budget shrinks.

Keep paper heads / depth / train recipe; shrink ``d_h`` (and ``H`` only when
``d_h`` alone cannot reach ≤500k, currently VOC). PATTERN / CLUSTER configs
are already hand-authored under ``configs/gated_hybrid/dh_matched/``.

Skip datasets whose main Tab. 3/4 SiGMA is already ≤500k (ZINC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "gated_hybrid" / "dh_matched"


def _load(rel: str) -> dict[str, Any]:
    """Load an anchor YAML as a dict."""
    path = ROOT / rel
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _dump(name: str, cfg: dict[str, Any], comment: str) -> Path:
    """Write a d_h-matched config with a header comment."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    path.write_text(f"# {comment}\n{body}")
    return path


def _retag(cfg: dict[str, Any], group: str, *extra_tags: str) -> None:
    """Replace wandb tags/group for the d_h-matched campaign."""
    wb = cfg.setdefault("wandb", {})
    wb["use"] = False
    wb["project"] = "GNNPlus"
    wb["group"] = group
    wb["tags"] = [
        "gnnplus",
        "hybrid_gnn",
        "sigma_dh_matched",
        group,
        *extra_tags,
    ]


def _set_dh(
    cfg: dict[str, Any],
    *,
    d_h: int,
    dim_inner: int | None = None,
) -> None:
    """Apply head-width (and optional model-width) override."""
    gnn = cfg.setdefault("gnn", {})
    hy = gnn.setdefault("hybrid", {})
    hy["d_h"] = int(d_h)
    hy["log_gate_stats"] = True
    if dim_inner is not None:
        gnn["dim_inner"] = int(dim_inner)


def main() -> None:
    """Emit d_h-matched YAMLs for over-500k Tab. 3/4 SiGMA anchors."""
    written: list[Path] = []

    # --- MNIST (main 965k a2g2) → ≤500k ---
    cfg = _load("configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml")
    _set_dh(cfg, d_h=37)
    _retag(cfg, "paper_sigma_dh_matched_mnist_dh37", "mnist", "dh37", "b500k", "hybrid_a2g2")
    written.append(
        _dump(
            "mnist-a2g2-dh37.yaml",
            cfg,
            "MNIST Tab.3 SiGMA a2g2 H60: d_h=37 (~488k ≤500k). Heads/L/H/LR frozen.",
        )
    )

    # --- CIFAR10 (main 27.8M a8g4) → ≤500k / ≤1M ---
    for d_h, tag, budget, approx in (
        (20, "cifar_dh20", "b500k", "~477k"),
        (34, "cifar_dh34", "b1m", "~978k"),
    ):
        cfg = _load("configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml")
        _set_dh(cfg, d_h=d_h)
        group = f"paper_sigma_dh_matched_{tag}"
        _retag(cfg, group, "cifar10", f"dh{d_h}", budget, "hybrid_a8g4")
        written.append(
            _dump(
                f"cifar10-a8g4-dh{d_h}.yaml",
                cfg,
                f"CIFAR10 Tab.3 SiGMA a8g4 H35: d_h={d_h} ({approx} {budget}). "
                "Heads/L/H/LR frozen.",
            )
        )

    # --- Peptides-func Tab.4 / Tab.12 a1g2 homog (main ~1.54M) ---
    for d_h, tag, budget, approx in (
        (23, "pepfunc_dh23", "b500k", "~491k"),
        (75, "pepfunc_dh75", "b1m", "~995k"),
    ):
        cfg = _load(
            "configs/gated_hybrid/peptides-func-hybrid-homog-a1g2-gcn-anchor.yaml"
        )
        _set_dh(cfg, d_h=d_h)
        group = f"paper_sigma_dh_matched_{tag}"
        _retag(cfg, group, "peptides_func", f"dh{d_h}", budget, "hybrid_a1g2")
        written.append(
            _dump(
                f"peptides-func-a1g2-dh{d_h}.yaml",
                cfg,
                f"Pep-func Tab.4 SiGMA a1g2 GCN×2 H275: d_h={d_h} ({approx} {budget}). "
                "Heads/L/H/LR frozen.",
            )
        )

    # --- Peptides-struct (main ~2.35M a1g1) ---
    for d_h, tag, budget, approx in (
        (43, "pepstruct_dh43", "b500k", "~498k"),
        (92, "pepstruct_dh92", "b1m", "~998k"),
    ):
        cfg = _load(
            "configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
        )
        _set_dh(cfg, d_h=d_h)
        group = f"paper_sigma_dh_matched_{tag}"
        _retag(cfg, group, "peptides_struct", f"dh{d_h}", budget, "hybrid_a1g1")
        written.append(
            _dump(
                f"peptides-struct-a1g1-dh{d_h}.yaml",
                cfg,
                f"Pep-struct Tab.4 SiGMA a1g1 GINE H200: d_h={d_h} ({approx} {budget}). "
                "Heads/L/H/LR frozen.",
            )
        )

    # --- VOC: ≤1M keeps H=95; ≤500k also shrinks H→64 (d_h alone floors ~623k) ---
    cfg = _load("configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml")
    _set_dh(cfg, d_h=15)
    _retag(cfg, "paper_sigma_dh_matched_voc_dh15", "voc", "dh15", "b1m", "hybrid_a2g2")
    written.append(
        _dump(
            "voc-a2g2-dh15.yaml",
            cfg,
            "VOC Tab.4 SiGMA a2g2 H95: d_h=15 (~995k ≤1M). Heads/L/H/LR frozen.",
        )
    )
    cfg = _load("configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml")
    _set_dh(cfg, d_h=12, dim_inner=64)
    _retag(
        cfg,
        "paper_sigma_dh_matched_voc_h64_dh12",
        "voc",
        "h64",
        "dh12",
        "b500k",
        "hybrid_a2g2",
    )
    written.append(
        _dump(
            "voc-a2g2-h64-dh12.yaml",
            cfg,
            "VOC Tab.4 SiGMA a2g2: H 95→64 + d_h=12 (~499k ≤500k). "
            "Heads/L/LR frozen; H shrink required (d_h=1 @ H95 still ~623k).",
        )
    )

    # --- COCO (main ~898k) → ≤500k ---
    cfg = _load("configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml")
    _set_dh(cfg, d_h=34)
    _retag(cfg, "paper_sigma_dh_matched_coco_dh34", "coco", "dh34", "b500k", "hybrid_a1g1")
    written.append(
        _dump(
            "coco-a1g1-dh34.yaml",
            cfg,
            "COCO Tab.4 SiGMA a1g1 H52: d_h=34 (~480k ≤500k). Heads/L/H/LR frozen.",
        )
    )

    # --- MalNet vcb1cuql (main ~549k) → ≤500k ---
    cfg = _load("configs/gated_hybrid/malnet-hybrid-vcb1cuql-anchor.yaml")
    _set_dh(cfg, d_h=57)
    _retag(
        cfg,
        "paper_sigma_dh_matched_malnet_dh57",
        "malnet",
        "dh57",
        "b500k",
        "hybrid_a1g1",
    )
    written.append(
        _dump(
            "malnet-a1g1-dh57.yaml",
            cfg,
            "MalNet Tab.4 SiGMA a1g1 GCNE H110: d_h=57 (~498k ≤500k). "
            "Heads/L/H/LR frozen.",
        )
    )

    print(f"Wrote {len(written)} configs under {OUT}:")
    for path in written:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
