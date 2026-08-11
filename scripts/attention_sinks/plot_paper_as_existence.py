#!/usr/bin/env python3
"""Paper figures: attention-sink existence on TU graphs.

Builds publication plots from ``*_mech.csv`` dumps under
``results/tu_attention_sinks/analysis/``:

1. Bio GPS ungated: layer×head heatmaps of mean ×uniform and τ-sink rate
2. Strongest-head summary bars (bio AS vs flat social GPS)
3. SiGMA vs GPS on IMDB / COLLAB (AS where GPS is flat)

Example::

  python scripts/attention_sinks/plot_paper_as_existence.py \\
    --analysis-dir results/tu_attention_sinks/analysis \\
    --out-dir results/tu_attention_sinks/paper_figures
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Paper-friendly palette (avoid purple / neon defaults).
_CMAP_RATIO = "YlOrRd"
_CMAP_RATE = "Blues"
_BAR_BIO = "#1f4e79"
_BAR_FLAT = "#9aa5b1"
_BAR_SIGMA = "#c45c26"
_BAR_GPS = "#4a6670"


@dataclass(frozen=True)
class RunSpec:
    """One mech CSV run used in a paper panel."""

    label: str
    filename: str
    dataset: str
    arch: str  # GPS | SiGMA
    gate: str  # ungated | gated


BIO_GPS_UNGATED: Tuple[RunSpec, ...] = (
    RunSpec("MUTAG", "mutag_GPS_ungated_attn_lr001_seed2_mech.csv", "MUTAG", "GPS", "ungated"),
    RunSpec(
        "ENZYMES",
        "enzymes_GPS_ungated_attn_lr001_seed2_mech.csv",
        "ENZYMES",
        "GPS",
        "ungated",
    ),
    RunSpec(
        "PROTEINS",
        "proteins_GPS_ungated_attn_lr001_seed2_mech.csv",
        "PROTEINS",
        "GPS",
        "ungated",
    ),
)

SOCIAL_COMPARE: Tuple[RunSpec, ...] = (
    RunSpec(
        "IMDB · GPS",
        "imdb_binary_GPS_ungated_attn_lr001_seed2_mech.csv",
        "IMDB",
        "GPS",
        "ungated",
    ),
    RunSpec(
        "IMDB · SiGMA",
        "imdb_binary_SiGMA_hetero_ungated_attn_lr001_seed2_mech.csv",
        "IMDB",
        "SiGMA",
        "ungated",
    ),
    RunSpec(
        "COLLAB · GPS",
        "collab_GPS_ungated_attn_lr001_seed2_mech.csv",
        "COLLAB",
        "GPS",
        "ungated",
    ),
    RunSpec(
        "COLLAB · SiGMA",
        "collab_SiGMA_hetero_ungated_attn_lr01_seed2_mech.csv",
        "COLLAB",
        "SiGMA",
        "ungated",
    ),
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for paper AS existence figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/tu_attention_sinks/analysis"),
        help="Directory with *_mech.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tu_attention_sinks/paper_figures"),
        help="Output directory for PNG/PDF figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    return parser.parse_args(argv)


def _parse_layer_head(key: str) -> Tuple[int, int]:
    """Parse ``layer{L}_attn{H}`` into ``(L, H)``."""
    # Expected: layer0_attn0
    parts = key.split("_")
    layer = int(parts[0].replace("layer", ""))
    head = int(parts[1].replace("attn", ""))
    return layer, head


def _style_paper() -> None:
    """Apply restrained matplotlib defaults for print."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def load_head_summary(csv_path: Path) -> pd.DataFrame:
    """Aggregate mech rows to per-(layer, head) mean ratio and sink rate.

    Args:
        csv_path: Path to a ``*_mech.csv`` file.

    Returns:
        DataFrame with columns ``layer``, ``head``, ``layer_head``,
        ``mean_ratio``, ``sink_rate``, ``n_graphs``.
    """
    df = pd.read_csv(csv_path)
    required = {"layer_head", "ratio_vs_uniform", "tau_sink"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    grouped = (
        df.groupby("layer_head", as_index=False)
        .agg(
            mean_ratio=("ratio_vs_uniform", "mean"),
            sink_rate=("tau_sink", "mean"),
            n_graphs=("tau_sink", "size"),
        )
        .sort_values("mean_ratio", ascending=False)
        .reset_index(drop=True)
    )
    layers_heads = grouped["layer_head"].map(_parse_layer_head)
    grouped["layer"] = [lh[0] for lh in layers_heads]
    grouped["head"] = [lh[1] for lh in layers_heads]
    return grouped


def load_run(analysis_dir: Path, spec: RunSpec) -> pd.DataFrame:
    """Load and tag a run summary.

    Args:
        analysis_dir: Directory containing mech CSVs.
        spec: Run metadata.

    Returns:
        Head-level summary with run metadata columns.
    """
    path = analysis_dir / spec.filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing mech CSV: {path}")
    summary = load_head_summary(path)
    summary["label"] = spec.label
    summary["dataset"] = spec.dataset
    summary["arch"] = spec.arch
    summary["gate"] = spec.gate
    return summary


def matrix_from_heads(
    summary: pd.DataFrame,
    value_col: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Build a layer×head matrix from a head summary.

    Args:
        summary: Output of :func:`load_head_summary`.
        value_col: Column to place in the matrix.

    Returns:
        ``(matrix, layer_ids, head_ids)``.
    """
    layers = sorted(summary["layer"].unique().tolist())
    heads = sorted(summary["head"].unique().tolist())
    mat = np.full((len(layers), len(heads)), np.nan, dtype=np.float64)
    layer_to_i = {layer: i for i, layer in enumerate(layers)}
    head_to_j = {head: j for j, head in enumerate(heads)}
    for _, row in summary.iterrows():
        i = layer_to_i[int(row["layer"])]
        j = head_to_j[int(row["head"])]
        mat[i, j] = float(row[value_col])
    return mat, layers, heads


def _annotate_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    fmt: str = ".1f",
    text_thresh: Optional[float] = None,
) -> None:
    """Write numeric annotations on a heatmap."""
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return
    if text_thresh is None:
        text_thresh = float(np.nanpercentile(finite, 60))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val >= text_thresh else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=color, fontsize=7)


def _save(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> List[Path]:
    """Save PNG and PDF; return written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_bio_heatmaps(
    analysis_dir: Path,
    out_dir: Path,
    dpi: int,
) -> List[Path]:
    """Layer×head heatmaps of ×uniform and sink-rate for bio GPS ungated.

    Args:
        analysis_dir: Mech CSV directory.
        out_dir: Figure output directory.
        dpi: PNG DPI.

    Returns:
        Paths of written figure files.
    """
    runs = [load_run(analysis_dir, spec) for spec in BIO_GPS_UNGATED]
    written: List[Path] = []

    # Ratio heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)
    vmax = max(float(r["mean_ratio"].max()) for r in runs)
    norm = Normalize(vmin=1.0, vmax=max(vmax, 1.05))
    for ax, summary, spec in zip(axes, runs, BIO_GPS_UNGATED):
        mat, layers, heads = matrix_from_heads(summary, "mean_ratio")
        im = ax.imshow(mat, aspect="auto", cmap=_CMAP_RATIO, norm=norm, origin="lower")
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([str(h) for h in heads])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers])
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        top = summary.iloc[0]
        ax.set_title(
            f"{spec.dataset}\n"
            f"top L{int(top['layer'])}H{int(top['head'])}: "
            f"{float(top['mean_ratio']):.1f}× · {100 * float(top['sink_rate']):.0f}% sinks"
        )
        _annotate_heatmap(ax, mat, fmt=".1f", text_thresh=norm.vmax * 0.55)
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
        label=r"mean $\max\alpha$ / $(1/n)$",
    )
    fig.suptitle("GPS ungated — attention concentration on bio graphs", y=1.05, fontsize=12)
    written.extend(_save(fig, out_dir, "fig_bio_gps_ratio_heatmaps", dpi))

    # Sink-rate heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)
    for ax, summary, spec in zip(axes, runs, BIO_GPS_UNGATED):
        mat, layers, heads = matrix_from_heads(summary, "sink_rate")
        im = ax.imshow(mat, aspect="auto", cmap=_CMAP_RATE, vmin=0.0, vmax=1.0, origin="lower")
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([str(h) for h in heads])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers])
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        top = summary.iloc[0]
        ax.set_title(
            f"{spec.dataset}\n"
            f"top L{int(top['layer'])}H{int(top['head'])}: "
            f"{100 * float(top['sink_rate']):.0f}% τ-sinks"
        )
        _annotate_heatmap(ax, 100.0 * mat, fmt=".0f", text_thresh=55.0)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label=r"τ·μ sink rate (%)")
    fig.suptitle("GPS ungated — τ-sink rate on bio graphs", y=1.05, fontsize=12)
    written.extend(_save(fig, out_dir, "fig_bio_gps_sinkrate_heatmaps", dpi))
    return written


def plot_bio_vs_flat_bars(
    analysis_dir: Path,
    out_dir: Path,
    dpi: int,
) -> List[Path]:
    """Strongest-head bars: bio AS vs flat social GPS.

    Args:
        analysis_dir: Mech CSV directory.
        out_dir: Figure output directory.
        dpi: PNG DPI.

    Returns:
        Paths of written figure files.
    """
    specs = list(BIO_GPS_UNGATED) + [
        RunSpec(
            "IMDB",
            "imdb_binary_GPS_ungated_attn_lr001_seed2_mech.csv",
            "IMDB",
            "GPS",
            "ungated",
        ),
        RunSpec(
            "COLLAB",
            "collab_GPS_ungated_attn_lr001_seed2_mech.csv",
            "COLLAB",
            "GPS",
            "ungated",
        ),
    ]
    tops: List[Dict[str, float | str]] = []
    for spec in specs:
        summary = load_run(analysis_dir, spec)
        top = summary.iloc[0]
        tops.append(
            {
                "dataset": spec.dataset,
                "mean_ratio": float(top["mean_ratio"]),
                "sink_rate": float(top["sink_rate"]),
                "layer": int(top["layer"]),
                "head": int(top["head"]),
                "is_bio": spec.dataset in {"MUTAG", "ENZYMES", "PROTEINS"},
            }
        )

    labels = [
        f"{t['dataset']}\nL{t['layer']}H{t['head']}" for t in tops
    ]
    ratios = [float(t["mean_ratio"]) for t in tops]
    rates = [100.0 * float(t["sink_rate"]) for t in tops]
    colors = [_BAR_BIO if t["is_bio"] else _BAR_FLAT for t in tops]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    x = np.arange(len(labels))

    axes[0].bar(x, ratios, color=colors, width=0.72, edgecolor="none")
    axes[0].axhline(1.0, color="#555555", lw=0.8, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(r"mean $\max\alpha$ / $(1/n)$")
    axes[0].set_title("Strongest head · concentration")
    axes[0].legend(
        handles=[
            Patch(facecolor=_BAR_BIO, label="Bio (GPS ungated)"),
            Patch(facecolor=_BAR_FLAT, label="Social (GPS ungated)"),
            plt.Line2D([0], [0], color="#555555", lw=0.8, ls="--", label="uniform (1×)"),
        ],
        frameon=False,
        loc="upper left",
    )

    axes[1].bar(x, rates, color=colors, width=0.72, edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(r"τ·μ sink rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Strongest head · sink prevalence")

    fig.suptitle("Clear AS on bio graphs; GPS ungated flat on IMDB / COLLAB", fontsize=12)
    return _save(fig, out_dir, "fig_bio_as_vs_flat_gps_bars", dpi)


def plot_sigma_vs_gps_social(
    analysis_dir: Path,
    out_dir: Path,
    dpi: int,
) -> List[Path]:
    """SiGMA shows AS on IMDB/COLLAB where GPS does not.

    Args:
        analysis_dir: Mech CSV directory.
        out_dir: Figure output directory.
        dpi: PNG DPI.

    Returns:
        Paths of written figure files.
    """
    rows: List[Dict[str, float | str]] = []
    for spec in SOCIAL_COMPARE:
        summary = load_run(analysis_dir, spec)
        top = summary.iloc[0]
        rows.append(
            {
                "dataset": spec.dataset,
                "arch": spec.arch,
                "mean_ratio": float(top["mean_ratio"]),
                "sink_rate": float(top["sink_rate"]),
                "layer": int(top["layer"]),
                "head": int(top["head"]),
            }
        )
    frame = pd.DataFrame(rows)

    datasets = ["IMDB", "COLLAB"]
    x = np.arange(len(datasets))
    width = 0.36

    def _vals(arch: str, col: str) -> List[float]:
        out: List[float] = []
        for dataset in datasets:
            match = frame[(frame["dataset"] == dataset) & (frame["arch"] == arch)]
            if match.empty:
                out.append(0.0)
            else:
                out.append(float(match.iloc[0][col]))
        return out

    gps_ratio = _vals("GPS", "mean_ratio")
    sigma_ratio = _vals("SiGMA", "mean_ratio")
    gps_rate = [100.0 * v for v in _vals("GPS", "sink_rate")]
    sigma_rate = [100.0 * v for v in _vals("SiGMA", "sink_rate")]

    # Annotation strings with top head
    def _head_label(dataset: str, arch: str) -> str:
        match = frame[(frame["dataset"] == dataset) & (frame["arch"] == arch)].iloc[0]
        return f"L{int(match['layer'])}H{int(match['head'])}"

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), constrained_layout=True)

    b0 = axes[0].bar(
        x - width / 2,
        gps_ratio,
        width,
        color=_BAR_GPS,
        label="GPS ungated",
        edgecolor="none",
    )
    b1 = axes[0].bar(
        x + width / 2,
        sigma_ratio,
        width,
        color=_BAR_SIGMA,
        label="SiGMA ungated",
        edgecolor="none",
    )
    axes[0].axhline(1.0, color="#555555", lw=0.8, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel(r"mean $\max\alpha$ / $(1/n)$")
    axes[0].set_title("Strongest head · concentration")
    axes[0].legend(frameon=False, loc="upper left")
    for bars, arch in ((b0, "GPS"), (b1, "SiGMA")):
        for rect, dataset in zip(bars, datasets):
            height = rect.get_height()
            axes[0].annotate(
                f"{height:.1f}×\n{_head_label(dataset, arch)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    # Headroom for COLLAB ~60× annotations
    axes[0].set_ylim(0, max(sigma_ratio) * 1.18)

    axes[1].bar(
        x - width / 2,
        gps_rate,
        width,
        color=_BAR_GPS,
        label="GPS ungated",
        edgecolor="none",
    )
    axes[1].bar(
        x + width / 2,
        sigma_rate,
        width,
        color=_BAR_SIGMA,
        label="SiGMA ungated",
        edgecolor="none",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel(r"τ·μ sink rate (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Strongest head · sink prevalence")
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle(
        "SiGMA exhibits AS on IMDB / COLLAB where GPS ungated is flat",
        fontsize=12,
    )
    written = _save(fig, out_dir, "fig_sigma_vs_gps_imdb_collab", dpi)

    # Companion heatmaps: ratio for GPS vs SiGMA on each social dataset
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.6), constrained_layout=True)
    panel_specs = [
        (SOCIAL_COMPARE[0], axes[0, 0]),
        (SOCIAL_COMPARE[1], axes[0, 1]),
        (SOCIAL_COMPARE[2], axes[1, 0]),
        (SOCIAL_COMPARE[3], axes[1, 1]),
    ]
    # Shared color scale would squash COLLAB SiGMA; use per-row shared scale
    for row_i, datasets_row in enumerate((("IMDB",), ("COLLAB",))):
        row_specs = [s for s, _ in panel_specs if s.dataset in datasets_row]
        summaries = [load_run(analysis_dir, s) for s in row_specs]
        vmax = max(float(s["mean_ratio"].max()) for s in summaries)
        vmax = max(vmax, 1.05)
        norm = Normalize(vmin=1.0, vmax=vmax)
        for spec, ax in panel_specs:
            if spec.dataset not in datasets_row:
                continue
            summary = load_run(analysis_dir, spec)
            mat, layers, heads = matrix_from_heads(summary, "mean_ratio")
            im = ax.imshow(mat, aspect="auto", cmap=_CMAP_RATIO, norm=norm, origin="lower")
            ax.set_xticks(range(len(heads)))
            ax.set_xticklabels([str(h) for h in heads])
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([str(layer) for layer in layers])
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            top = summary.iloc[0]
            ax.set_title(
                f"{spec.dataset} · {spec.arch}\n"
                f"top {float(top['mean_ratio']):.1f}× · "
                f"{100 * float(top['sink_rate']):.0f}% sinks"
            )
            # Annotate only if matrix is not huge numerically (IMDB ok; COLLAB SiGMA still annotate)
            _annotate_heatmap(
                ax,
                mat,
                fmt=".1f" if vmax < 20 else ".0f",
                text_thresh=norm.vmin + 0.55 * (norm.vmax - norm.vmin),
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(im, cax=cax)

    fig.suptitle(
        r"Layer×head mean $\max\alpha/(1/n)$ — GPS flat, SiGMA concentrated",
        fontsize=12,
    )
    written.extend(_save(fig, out_dir, "fig_sigma_vs_gps_social_heatmaps", dpi))
    return written


def write_summary_table(
    analysis_dir: Path,
    out_dir: Path,
) -> Path:
    """Write a CSV of strongest-head stats used in the figures.

    Args:
        analysis_dir: Mech CSV directory.
        out_dir: Output directory.

    Returns:
        Path to the summary CSV.
    """
    specs: List[RunSpec] = list(BIO_GPS_UNGATED) + list(SOCIAL_COMPARE)
    # Deduplicate by filename while preserving order
    seen: set[str] = set()
    unique: List[RunSpec] = []
    for spec in specs:
        if spec.filename in seen:
            continue
        seen.add(spec.filename)
        unique.append(spec)

    records: List[Dict[str, object]] = []
    for spec in unique:
        summary = load_run(analysis_dir, spec)
        top = summary.iloc[0]
        records.append(
            {
                "dataset": spec.dataset,
                "arch": spec.arch,
                "gate": spec.gate,
                "top_layer_head": top["layer_head"],
                "mean_ratio": float(top["mean_ratio"]),
                "sink_rate": float(top["sink_rate"]),
                "n_graphs_top_head": int(top["n_graphs"]),
                "n_heads": int(len(summary)),
                "source_csv": spec.filename,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper_as_existence_summary.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate paper AS existence figures."""
    args = _parse_args(argv)
    _style_paper()
    analysis_dir: Path = args.analysis_dir
    out_dir: Path = args.out_dir
    dpi: int = int(args.dpi)

    written: List[Path] = []
    written.extend(plot_bio_heatmaps(analysis_dir, out_dir, dpi))
    written.extend(plot_bio_vs_flat_bars(analysis_dir, out_dir, dpi))
    written.extend(plot_sigma_vs_gps_social(analysis_dir, out_dir, dpi))
    summary_csv = write_summary_table(analysis_dir, out_dir)

    print("Wrote summary:", summary_csv)
    print("Wrote figures:")
    for path in written:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
