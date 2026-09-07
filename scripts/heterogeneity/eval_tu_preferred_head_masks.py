#!/usr/bin/env python3
"""Preferred-head masking ablation on real TU SiGMA checkpoints (eval only).

Causal analogue of the synthetic GCN/GIN mask study
(``scripts/synthetic/eval_gcn_gin_routing_masks.py``):

**Design A (global mask, preference-stratified).** Zero one MP head on *all*
graphs; report accuracy within specialist-preference bins. Expectation if
routing is functional: on GCN-preferred graphs, ``mask_GCN`` hurts more than
``mask_SAGE`` (and symmetrically).

**Design B (adaptive per-graph mask).** For each graph, zero its *preferred*
vs *runner-up* specialist head (``mask_preferred`` / ``mask_anti``). Uses
``batch_size=1``. Expectation: ``mask_preferred`` hurts more than ``mask_anti``.

Preference comes from the same Xu hetero pickles as
``join_tu_gate_operator_preference.py``. Val/test graph indices are remapped
via GraphGym ``ShuffleSplit`` (same as the join script).

Masks apply to **all** MP layers by default (TU L=4; unlike the synthetic
layer-0-only ablation).

Example (cluster; ckpts live under ``$GNNPLUS_OUT_DIR``)::

  python scripts/heterogeneity/eval_tu_preferred_head_masks.py \\
    --datasets mutag,enzymes \\
    --hetero-root $GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge \\
    --gate-root $GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage \\
    --lr-tag a0g2_gated --operators GCN,SAGE --seeds 0,1,2,3,4 \\
    --adaptive --out-dir results/heterogeneity/tu_pref_mask_gcs_a0g2_gated
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torch_geometric.data import Batch
from torch_geometric.graphgym.checkpoint import get_ckpt_epochs, load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args as gg_parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

import GNNPlus  # noqa: F401

from GNNPlus.hybrid_gate_tracking import _unwrap_model
from scripts.heterogeneity.join_tu_gate_operator_preference import (
    OPERATOR_CFG_SUFFIX,
    TU_NAME,
    _load_operator_profiles,
    _load_tu_dataset,
    _preferred_operator,
    _tu_labels,
    reconstruct_graphgym_random_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PALETTE: Mapping[str, str] = {
    "none": "#4C72B0",
    "mask_preferred": "#E45756",
    "mask_anti": "#55A868",
    "mask_random": "#8C8C8C",
    "GCN": "#4C72B0",
    "GIN": "#55A868",
    "SAGE": "#DD8452",
    "GAT": "#937860",
}


@dataclass(frozen=True)
class PrefInfo:
    """Specialist preference for one global graph index."""

    preferred: str
    anti: str
    margin: float
    accuracies: Mapping[str, float]


@dataclass
class MaskEvalRow:
    """One (run × mask_mode) accuracy row, stratified by preference."""

    dataset: str
    lr_tag: str
    seed: int
    mask_mode: str
    run_dir: str
    epoch: int
    n_all: int
    acc_all: float
    n_by_pref: Dict[str, int] = field(default_factory=dict)
    acc_by_pref: Dict[str, float] = field(default_factory=dict)


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="mutag,enzymes",
        help="Comma-separated dataset slugs.",
    )
    parser.add_argument(
        "--hetero-root",
        type=str,
        required=True,
        help="Root with <ds>_{gcn,gin,sage,gat}/ preference pickles.",
    )
    parser.add_argument(
        "--gate-root",
        type=str,
        required=True,
        help="Root with <ds>_SiGMA_hetero_<lr-tag>_seed<s>/ run dirs.",
    )
    parser.add_argument(
        "--lr-tag",
        type=str,
        required=True,
        help="Run-dir tag after SiGMA_hetero_ (e.g. a0g2_gated or xu).",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default="GCN,SAGE",
        help="Specialist operators used for preference (must match pickles).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated SiGMA seeds.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated splits to evaluate (val and/or test).",
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=100,
        help="Min specialist test appearances for preference membership.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get("GNNPLUS_DATASET_DIR", "datasets"),
        help="GraphGym dataset.dir parent.",
    )
    parser.add_argument(
        "--tu-root",
        type=str,
        default="",
        help="TUDataset download root (default: <dataset-dir>/TUDataset).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for CSVs + figures.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Also run Design B (mask_preferred / mask_anti / mask_random).",
    )
    parser.add_argument(
        "--mask-all-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply MP head mask on every layer (default: True).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args(argv)


def _csv_list(raw: str) -> Tuple[str, ...]:
    """Split a comma-separated CLI list."""
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_run_config(run_dir: Path) -> Optional[Path]:
    """Return config yaml for a run dir, or None if missing."""
    for name in ("config_used.yaml", "config.yaml"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _pick_best_epoch(run_dir: Path) -> int:
    """Return latest checkpoint epoch under ``run_dir/ckpt``."""
    cfg.run_dir = str(run_dir)
    epochs = list(get_ckpt_epochs())
    if not epochs:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/ckpt")
    return int(max(epochs))


def _pred_labels_from_score(pred_score: torch.Tensor) -> torch.Tensor:
    """Convert model scores to integer labels (GraphGym Logger parity)."""
    if pred_score.ndim == 1 or (pred_score.ndim == 2 and pred_score.shape[1] == 1):
        thresh = float(getattr(cfg.model, "thresh", 0.5))
        return (pred_score.view(-1) > thresh).long()
    return pred_score.argmax(dim=-1).view(-1)


def _mp_head_names(hybrid: object) -> List[str]:
    """Parse MP head type names from hybrid cfg."""
    raw = str(getattr(hybrid, "gnn_types", "") or "")
    names = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not names:
        raise ValueError("hybrid.gnn_types is empty")
    return names


def _head_index_map(head_names: Sequence[str]) -> Dict[str, int]:
    """Map operator name → MP head index (first occurrence)."""
    out: Dict[str, int] = {}
    for i, name in enumerate(head_names):
        out.setdefault(name, i)
    return out


def _build_pref_map(
    profiles: Mapping[int, Mapping[str, float]],
    operators: Sequence[str],
) -> Dict[int, PrefInfo]:
    """Build preference info for graphs with full operator coverage."""
    pref_map: Dict[int, PrefInfo] = {}
    for gidx, acc in profiles.items():
        if any(op not in acc for op in operators):
            continue
        preferred, margin = _preferred_operator({op: acc[op] for op in operators})
        if preferred == "TIE":
            anti = "TIE"
        else:
            ordered = sorted(
                ((op, acc[op]) for op in operators if op != preferred),
                key=lambda kv: (-kv[1], kv[0]),
            )
            anti = ordered[0][0] if ordered else "TIE"
        pref_map[int(gidx)] = PrefInfo(
            preferred=preferred,
            anti=anti,
            margin=float(margin),
            accuracies=dict(acc),
        )
    return pref_map


def _load_cfg_for_run(
    run_dir: Path,
    *,
    dataset_dir: str,
    seed: int,
    batch_size: Optional[int] = None,
) -> None:
    """Load GraphGym cfg from the run's saved config yaml."""
    cfg_path_obj = _resolve_run_config(run_dir)
    if cfg_path_obj is None:
        raise FileNotFoundError(f"No config yaml in {run_dir}")
    overrides = [
        "--cfg",
        str(cfg_path_obj),
        "dataset.dir",
        dataset_dir,
        "seed",
        str(seed),
        "wandb.use",
        "False",
    ]
    if batch_size is not None:
        overrides.extend(["train.batch_size", str(int(batch_size))])
    old_argv = sys.argv
    sys.argv = [old_argv[0], *overrides]
    try:
        args = gg_parse_args()
        set_cfg(cfg)
        load_cfg(cfg, args)
    finally:
        sys.argv = old_argv
    cfg.run_dir = str(run_dir)
    cfg.out_dir = str(run_dir.parent)


def _global_mask(
    mask_mode: str,
    *,
    head_names: Sequence[str],
    head_index: Mapping[str, int],
) -> Optional[List[bool]]:
    """Build a batch-wide MP head mask for Design A, or None for ``none``."""
    n = len(head_names)
    if mask_mode == "none":
        return None
    if not mask_mode.startswith("mask_"):
        raise ValueError(f"Unknown global mask mode {mask_mode!r}")
    op = mask_mode[len("mask_") :].upper()
    if op not in head_index:
        raise KeyError(f"Mask target {op} not in heads {list(head_names)}")
    active = [True] * n
    active[head_index[op]] = False
    return active


def _adaptive_mask(
    mask_mode: str,
    pref: PrefInfo,
    *,
    head_names: Sequence[str],
    head_index: Mapping[str, int],
    rng: np.random.Generator,
) -> Optional[List[bool]]:
    """Build a per-graph MP head mask for Design B."""
    n = len(head_names)
    if mask_mode == "none":
        return None
    if mask_mode == "mask_preferred":
        if pref.preferred == "TIE" or pref.preferred not in head_index:
            return None
        active = [True] * n
        active[head_index[pref.preferred]] = False
        return active
    if mask_mode == "mask_anti":
        if pref.anti == "TIE" or pref.anti not in head_index:
            return None
        active = [True] * n
        active[head_index[pref.anti]] = False
        return active
    if mask_mode == "mask_random":
        idx = int(rng.integers(0, n))
        active = [True] * n
        active[idx] = False
        return active
    raise ValueError(f"Unknown adaptive mask mode {mask_mode!r}")


@torch.no_grad()
def _forward_masked(
    core: torch.nn.Module,
    batch: Batch,
    mp_head_mask: Optional[Sequence[bool]],
    *,
    mask_all_layers: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run hybrid core with optional MP head masking."""
    (
        x,
        batch_enc,
        edge_index_attn,
        edge_attr_attn,
        edge_index_mp,
        edge_attr_mp,
        _ei,
        _ea,
    ) = core._encode_batch(batch)

    for layer_i, layer in enumerate(core.layers):
        use_mask = mp_head_mask if (mask_all_layers or layer_i == 0) else None
        layer_out = layer(
            x,
            edge_index_mp,
            batch_enc.batch,
            edge_attr_mp,
            edge_index_attn=edge_index_attn,
            edge_attr_attn=edge_attr_attn,
            edge_index_mp=edge_index_mp,
            edge_attr_mp=edge_attr_mp,
            mp_head_mask=use_mask,
        )
        x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        if core.ffn_blocks is not None:
            x = core.ffn_blocks[layer_i](x)
        batch_enc.x = x

    pred, true = core.post_mp(batch_enc)
    return pred, true


def _acc(num: int, den: int) -> float:
    """Safe accuracy."""
    return float(num / den) if den > 0 else float("nan")


@torch.no_grad()
def evaluate_run_masks(
    *,
    run_dir: Path,
    dataset: str,
    lr_tag: str,
    seed: int,
    dataset_dir: str,
    tu_root: Path,
    pref_map: Mapping[int, PrefInfo],
    operators: Sequence[str],
    splits: Sequence[str],
    mask_modes: Sequence[str],
    adaptive: bool,
    mask_all_layers: bool,
    device: torch.device,
) -> List[MaskEvalRow]:
    """Evaluate one checkpoint under all requested mask modes."""
    batch_size = 1 if adaptive else None
    _load_cfg_for_run(
        run_dir,
        dataset_dir=dataset_dir,
        seed=seed,
        batch_size=batch_size,
    )
    seed_everything(int(seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    ds_obj = _load_tu_dataset(dataset, tu_root)
    labels = _tu_labels(ds_obj)
    _train_idx, val_idx, test_idx = reconstruct_graphgym_random_split(labels, seed=seed)
    split_indices = {
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }

    loaders = create_loader()
    loader_by_name = {
        "train": loaders[0] if len(loaders) > 0 else None,
        "val": loaders[1] if len(loaders) > 1 else None,
        "test": loaders[2] if len(loaders) > 2 else None,
    }

    model = create_model()
    epoch = _pick_best_epoch(run_dir)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    model.eval()
    model.to(device)
    core = _unwrap_model(model)

    hybrid = getattr(cfg.gnn, "hybrid", None)
    if hybrid is None:
        raise RuntimeError(f"{run_dir}: missing gnn.hybrid")
    head_names = _mp_head_names(hybrid)
    head_index = _head_index_map(head_names)
    for op in operators:
        if op not in head_index:
            logging.warning(
                "%s: preference operator %s not in MP heads %s",
                run_dir.name,
                op,
                head_names,
            )

    rows: List[MaskEvalRow] = []
    for mask_mode in mask_modes:
        correct_all = 0
        n_all = 0
        correct_by_pref: Dict[str, int] = defaultdict(int)
        n_by_pref: Dict[str, int] = defaultdict(int)
        rng = np.random.default_rng(seed + 17_029)

        for split_name in splits:
            loader = loader_by_name.get(split_name)
            if loader is None:
                raise RuntimeError(f"Missing loader for split {split_name}")
            indices = split_indices[split_name]
            cursor = 0
            for batch in loader:
                batch = batch.to(device)
                n_graphs = int(batch.num_graphs)
                gidxs = indices[cursor : cursor + n_graphs]
                if len(gidxs) != n_graphs:
                    raise RuntimeError(
                        f"{dataset} seed={seed} {split_name}: "
                        f"ran past split indices at cursor={cursor}"
                    )
                cursor += n_graphs

                # Design A: one mask for the whole batch.
                # Design B: batch_size=1 so adaptive mask is well-defined.
                if adaptive and mask_mode in (
                    "mask_preferred",
                    "mask_anti",
                    "mask_random",
                ):
                    if n_graphs != 1:
                        raise RuntimeError(
                            "Adaptive masks require batch_size=1 "
                            f"(got {n_graphs})"
                        )
                    gidx = int(gidxs[0])
                    pref = pref_map.get(gidx)
                    if pref is None:
                        # Graph lacked ≥min appearances under some specialist.
                        mp_mask = None
                        pref_key = "MISSING"
                    else:
                        mp_mask = _adaptive_mask(
                            mask_mode,
                            pref,
                            head_names=head_names,
                            head_index=head_index,
                            rng=rng,
                        )
                        pref_key = pref.preferred
                else:
                    mp_mask = _global_mask(
                        mask_mode,
                        head_names=head_names,
                        head_index=head_index,
                    )
                    pref_key = ""

                pred, true = _forward_masked(
                    core,
                    batch,
                    mp_mask,
                    mask_all_layers=mask_all_layers,
                )
                _loss, pred_score = compute_loss(pred, true)
                pred_label = _pred_labels_from_score(pred_score)
                true_label = true.view(-1).long()
                correct = pred_label == true_label

                correct_all += int(correct.sum().item())
                n_all += int(correct.numel())

                for i in range(n_graphs):
                    gidx = int(gidxs[i])
                    pref = pref_map.get(gidx)
                    key = pref.preferred if pref is not None else "MISSING"
                    if adaptive and mask_mode in (
                        "mask_preferred",
                        "mask_anti",
                        "mask_random",
                    ):
                        key = pref_key
                    n_by_pref[key] += 1
                    if bool(correct[i].item()):
                        correct_by_pref[key] += 1

            if cursor != len(indices):
                raise RuntimeError(
                    f"{dataset} seed={seed} {split_name}: "
                    f"consumed {cursor}/{len(indices)} graphs"
                )

        acc_by_pref = {
            k: _acc(correct_by_pref[k], n_by_pref[k]) for k in sorted(n_by_pref)
        }
        rows.append(
            MaskEvalRow(
                dataset=dataset,
                lr_tag=lr_tag,
                seed=seed,
                mask_mode=mask_mode,
                run_dir=str(run_dir),
                epoch=epoch,
                n_all=n_all,
                acc_all=_acc(correct_all, n_all),
                n_by_pref=dict(n_by_pref),
                acc_by_pref=acc_by_pref,
            )
        )
        logging.info(
            "%s seed=%d %s: acc_all=%.4f n=%d prefs=%s",
            dataset,
            seed,
            mask_mode,
            rows[-1].acc_all,
            n_all,
            {k: f"{acc_by_pref[k]:.3f}({n_by_pref[k]})" for k in sorted(n_by_pref)},
        )
    return rows


def _all_pref_keys(rows: Sequence[MaskEvalRow]) -> List[str]:
    """Sorted preference-bin keys appearing in rows."""
    keys: set[str] = set()
    for row in rows:
        keys.update(row.n_by_pref.keys())
    preferred_order = ["GCN", "GIN", "SAGE", "GAT", "TIE", "MISSING"]
    ordered = [k for k in preferred_order if k in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def _write_per_run_csv(rows: Sequence[MaskEvalRow], path: Path) -> None:
    """Write long-form per-run CSV (one row per mask × preference bin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pref_keys = _all_pref_keys(rows)
    fieldnames = [
        "dataset",
        "lr_tag",
        "seed",
        "mask_mode",
        "run_dir",
        "epoch",
        "n_all",
        "acc_all",
        "pref_bin",
        "n_pref",
        "acc_pref",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for pref in pref_keys:
                writer.writerow(
                    {
                        "dataset": row.dataset,
                        "lr_tag": row.lr_tag,
                        "seed": row.seed,
                        "mask_mode": row.mask_mode,
                        "run_dir": row.run_dir,
                        "epoch": row.epoch,
                        "n_all": row.n_all,
                        "acc_all": f"{row.acc_all:.6f}",
                        "pref_bin": pref,
                        "n_pref": int(row.n_by_pref.get(pref, 0)),
                        "acc_pref": (
                            f"{row.acc_by_pref[pref]:.6f}"
                            if pref in row.acc_by_pref
                            else ""
                        ),
                    }
                )


def _summarize(
    rows: Sequence[MaskEvalRow],
) -> List[Dict[str, object]]:
    """Mean ± std over seeds for acc_all and each preference bin."""
    pref_keys = _all_pref_keys(rows)
    summary: List[Dict[str, object]] = []
    keys = sorted({(r.dataset, r.lr_tag, r.mask_mode) for r in rows})
    for dataset, lr_tag, mask_mode in keys:
        subset = [
            r
            for r in rows
            if r.dataset == dataset and r.lr_tag == lr_tag and r.mask_mode == mask_mode
        ]
        if not subset:
            continue
        all_vals = [r.acc_all for r in subset]
        summary.append(
            {
                "dataset": dataset,
                "lr_tag": lr_tag,
                "mask_mode": mask_mode,
                "metric": "acc_all",
                "pref_bin": "ALL",
                "n_seeds": len(subset),
                "mean": float(mean(all_vals)),
                "std": float(pstdev(all_vals)) if len(all_vals) > 1 else 0.0,
            }
        )
        for pref in pref_keys:
            vals = [r.acc_by_pref[pref] for r in subset if pref in r.acc_by_pref]
            if not vals:
                continue
            summary.append(
                {
                    "dataset": dataset,
                    "lr_tag": lr_tag,
                    "mask_mode": mask_mode,
                    "metric": "acc_pref",
                    "pref_bin": pref,
                    "n_seeds": len(vals),
                    "mean": float(mean(vals)),
                    "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
                }
            )
    return summary


def _write_summary_csv(summary: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write aggregated summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "lr_tag",
        "mask_mode",
        "metric",
        "pref_bin",
        "n_seeds",
        "mean",
        "std",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def _plot_design_a(
    rows: Sequence[MaskEvalRow],
    *,
    operators: Sequence[str],
    out_path: Path,
    dpi: int,
) -> None:
    """Bars: for each preferred bin, accuracy under none / mask_each_op."""
    datasets = sorted({r.dataset for r in rows})
    global_modes = ["none"] + [f"mask_{op}" for op in operators]
    present_modes = [m for m in global_modes if any(r.mask_mode == m for r in rows)]
    if len(present_modes) < 2:
        logging.warning("Skipping Design A figure (need ≥2 global mask modes)")
        return

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.2 * len(datasets), 4.4),
        squeeze=False,
    )
    bar_w = 0.8 / max(len(present_modes), 1)

    for ax, dataset in zip(axes[0], datasets, strict=True):
        pref_bins = [op for op in operators if any(
            op in r.n_by_pref and r.dataset == dataset for r in rows
        )]
        x = np.arange(len(pref_bins))
        for i, mode in enumerate(present_modes):
            means: List[float] = []
            stds: List[float] = []
            for pref in pref_bins:
                vals = [
                    r.acc_by_pref[pref]
                    for r in rows
                    if r.dataset == dataset
                    and r.mask_mode == mode
                    and pref in r.acc_by_pref
                ]
                means.append(float(mean(vals)) if vals else float("nan"))
                stds.append(float(pstdev(vals)) if len(vals) > 1 else 0.0)
            color = PALETTE.get(mode.replace("mask_", ""), PALETTE.get(mode, "#888888"))
            ax.bar(
                x + i * bar_w,
                means,
                width=bar_w,
                yerr=stds,
                capsize=2.5,
                label=mode,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
            )
        ax.set_xticks(x + 0.5 * bar_w * (len(present_modes) - 1))
        ax.set_xticklabels([f"pref={p}" for p in pref_bins])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Accuracy (val+test)")
        ax.set_title(dataset.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Design A: global head mask × specialist preference", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def _plot_design_b(
    rows: Sequence[MaskEvalRow],
    *,
    out_path: Path,
    dpi: int,
) -> None:
    """Bars: overall accuracy under none / mask_preferred / mask_anti / mask_random."""
    modes = ["none", "mask_preferred", "mask_anti", "mask_random"]
    present = [m for m in modes if any(r.mask_mode == m for r in rows)]
    if "mask_preferred" not in present:
        logging.warning("Skipping Design B figure (no adaptive rows)")
        return
    datasets = sorted({r.dataset for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(datasets))
    bar_w = 0.8 / max(len(present), 1)
    for i, mode in enumerate(present):
        means: List[float] = []
        stds: List[float] = []
        for dataset in datasets:
            vals = [
                r.acc_all
                for r in rows
                if r.dataset == dataset and r.mask_mode == mode
            ]
            means.append(float(mean(vals)) if vals else float("nan"))
            stds.append(float(pstdev(vals)) if len(vals) > 1 else 0.0)
        ax.bar(
            x + i * bar_w,
            means,
            width=bar_w,
            yerr=stds,
            capsize=2.5,
            label=mode,
            color=PALETTE.get(mode, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
        )
    ax.set_xticks(x + 0.5 * bar_w * (len(present) - 1))
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy (val+test)")
    ax.set_title("Design B: per-graph preferred / anti / random mask")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_cli(argv)
    datasets = tuple(s.lower() for s in _csv_list(args.datasets))
    operators = tuple(s.upper() for s in _csv_list(args.operators))
    seeds = tuple(int(s) for s in _csv_list(args.seeds))
    splits = tuple(s.lower() for s in _csv_list(args.splits))
    for op in operators:
        if op not in OPERATOR_CFG_SUFFIX:
            raise ValueError(f"Unknown operator {op!r}")
    for split in splits:
        if split not in ("val", "test"):
            raise ValueError(f"Unsupported split {split!r}; use val,test")

    hetero_root = Path(args.hetero_root)
    gate_root = Path(args.gate_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = str(args.dataset_dir)
    tu_root = Path(args.tu_root) if args.tu_root else Path(dataset_dir) / "TUDataset"
    device = _select_device(args.device)

    global_modes = ["none"] + [f"mask_{op}" for op in operators]
    adaptive_modes = (
        ["mask_preferred", "mask_anti", "mask_random"] if args.adaptive else []
    )
    # Adaptive modes need batch_size=1; evaluate them in a second pass if needed.
    # Here we always use batch_size=1 when --adaptive so one pass covers both.
    mask_modes = global_modes + adaptive_modes

    all_rows: List[MaskEvalRow] = []
    for dataset in datasets:
        if dataset not in TU_NAME:
            logging.warning("Unusual dataset slug %s (TU_NAME fallback to upper)", dataset)
        profiles = _load_operator_profiles(
            hetero_root,
            dataset,
            int(args.min_appearances),
            operators,
        )
        pref_map = _build_pref_map(profiles, operators)
        n_untied = sum(1 for p in pref_map.values() if p.preferred != "TIE")
        logging.info(
            "%s preference map: %d graphs (%d untied) operators=%s",
            dataset,
            len(pref_map),
            n_untied,
            ",".join(operators),
        )

        for seed in seeds:
            run_name = f"{dataset}_SiGMA_hetero_{args.lr_tag}_seed{seed}"
            run_dir = gate_root / run_name
            if not run_dir.is_dir():
                logging.warning("Missing run dir: %s", run_dir)
                continue
            if _resolve_run_config(run_dir) is None:
                logging.warning("Missing config in %s", run_dir)
                continue
            if not (run_dir / "ckpt").is_dir():
                logging.warning("Missing ckpt/ in %s (mask eval needs checkpoints)", run_dir)
                continue
            rows = evaluate_run_masks(
                run_dir=run_dir,
                dataset=dataset,
                lr_tag=args.lr_tag,
                seed=seed,
                dataset_dir=dataset_dir,
                tu_root=tu_root,
                pref_map=pref_map,
                operators=operators,
                splits=splits,
                mask_modes=mask_modes,
                adaptive=bool(args.adaptive),
                mask_all_layers=bool(args.mask_all_layers),
                device=device,
            )
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No mask-eval rows produced (check gate-root ckpts).")

    per_run_path = out_dir / "mask_ablation_per_run.csv"
    summary_path = out_dir / "mask_ablation_summary.csv"
    _write_per_run_csv(all_rows, per_run_path)
    summary = _summarize(all_rows)
    _write_summary_csv(summary, summary_path)
    logging.info("Wrote %s", per_run_path)
    logging.info("Wrote %s", summary_path)

    paper = out_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    _plot_design_a(
        all_rows,
        operators=operators,
        out_path=paper / "fig_mask_design_a_by_pref.png",
        dpi=int(args.dpi),
    )
    if args.adaptive:
        _plot_design_b(
            all_rows,
            out_path=paper / "fig_mask_design_b_adaptive.png",
            dpi=int(args.dpi),
        )


if __name__ == "__main__":
    main()
