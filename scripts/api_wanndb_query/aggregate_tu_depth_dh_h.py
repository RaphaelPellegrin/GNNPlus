#!/usr/bin/env python3
"""Aggregate TU L×d_h×H gated vs ungated from W&B; write CSV + heatmaps.

As each dataset finishes on the cluster, re-run for that slug (and the H values
you care about). Outputs land in ``results/tu_sigma_depth_dh_h/analysis/``.

Examples:
    # MUTAG paper-width trunks (already done)
    python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py --datasets mutag --H 64 8

    # MUTAG narrow trunks (after H∈{4,2} MUTAG job finishes)
    python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py --datasets mutag --H 4 2

    # ENZYMES whenever 801-1600 is done
    python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py --datasets enzymes --H 64 8

    # several datasets / all H at once
    python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py \\
      --datasets mutag enzymes proteins --H 64 8 4 2

    # replot from cached CSV only
    python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py \\
      --datasets mutag --H 64 8 --from-csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ENTITY = "weber-geoml-harvard-university"
PROJECT = "GNNPlus"
LAYERS = [1, 2, 4, 8, 16]
DHS = [1, 2, 4, 16]
DEFAULT_HS = [64, 8]
OUT_DIR = Path("results/tu_sigma_depth_dh_h/analysis")

DATASET_LABELS: dict[str, str] = {
    "mutag": "MUTAG",
    "enzymes": "ENZYMES",
    "proteins": "PROTEINS",
    "collab": "COLLAB",
    "imdb_binary": "IMDB-BINARY",
    "reddit_binary": "REDDIT-BINARY",
}


def to_pct(raw: float) -> float:
    """Convert W&B accuracy to percent if logged in [0, 1]."""
    return raw * 100.0 if raw <= 1.5 else raw


def mean_std(xs: list[float]) -> tuple[float, float]:
    """Return sample mean and std (ddof=1)."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def seed_of(run: object) -> int | None:
    """Extract seed from run config or name."""
    cfg = getattr(run, "config", {}) or {}
    for k in ("seed", "run_seed"):
        if k in cfg:
            try:
                return int(cfg[k])
            except (TypeError, ValueError):
                pass
    name = getattr(run, "name", "") or ""
    for part in name.replace("-", "_").split("_"):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def group_by_seed(api: wandb.Api, group: str) -> dict[int, float]:
    """Map seed → best_test_perf (%) for finished runs in a W&B group."""
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group, "state": "finished"})
    out: dict[int, float] = {}
    for run in runs:
        raw = run.summary.get("best_test_perf")
        if raw is None:
            continue
        s = seed_of(run)
        if s is None:
            continue
        out[s] = to_pct(float(raw))
    return out


def sig_label(p: float) -> str:
    """Map paired p-value to significance marker."""
    if math.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_heatmaps(df: pd.DataFrame, out: Path, ds_label: str, hs: list[int]) -> None:
    """Write Δ heatmaps with mean±std in cells and a color/significance legend."""
    slug = ds_label.lower().replace("-", "_")
    for H in hs:
        sub = df[df["H"] == H].copy()
        if sub.empty:
            print(f"skip {ds_label} H={H}: no rows")
            continue

        delta = np.full((len(LAYERS), len(DHS)), np.nan)
        g_mean = np.full_like(delta, np.nan)
        g_std = np.full_like(delta, np.nan)
        u_mean = np.full_like(delta, np.nan)
        u_std = np.full_like(delta, np.nan)
        pmat = np.full_like(delta, np.nan)

        for i, L in enumerate(LAYERS):
            for j, dh in enumerate(DHS):
                r = sub[(sub["L"] == L) & (sub["d_h"] == dh)]
                if len(r) != 1:
                    continue
                row = r.iloc[0]
                delta[i, j] = float(row["delta"])
                g_mean[i, j] = float(row["gated_mean"])
                g_std[i, j] = float(row["gated_std"])
                u_mean[i, j] = float(row["ungated_mean"])
                u_std[i, j] = float(row["ungated_std"])
                pmat[i, j] = float(row["p_paired"])

        n_cells = int(np.isfinite(delta).sum())
        expect = len(LAYERS) * len(DHS)
        if n_cells < expect:
            print(
                f"warning {ds_label} H={H}: {n_cells}/{expect} cells "
                f"(plotting partial grid)"
            )

        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        finite = delta[np.isfinite(delta)]
        vmax = max(2.0, float(np.nanmax(np.abs(finite)))) if finite.size else 2.0
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DHS)))
        ax.set_xticklabels([str(d) for d in DHS])
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([str(L) for L in LAYERS])
        ax.set_xlabel(r"$d_h$")
        ax.set_ylabel(r"$L$ (layers)")
        ax.set_title(
            f"{ds_label}: SiGMA gated vs ungated (best LR), H={H}\n"
            r"Cell color = $\Delta$ = gated $-$ ungated (pp); "
            r"red $\Rightarrow$ gated better"
        )

        for i in range(len(LAYERS)):
            for j in range(len(DHS)):
                if np.isnan(delta[i, j]):
                    continue
                star = sig_label(pmat[i, j])
                text = (
                    f"Δ {delta[i, j]:+.1f}  {star}\n"
                    f"g {g_mean[i, j]:.1f}±{g_std[i, j]:.1f}\n"
                    f"u {u_mean[i, j]:.1f}±{u_std[i, j]:.1f}"
                )
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    linespacing=1.25,
                    color="white" if abs(delta[i, j]) > 0.55 * vmax else "black",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(r"$\Delta$ (pp): gated $-$ ungated")

        legend_handles = [
            Patch(
                facecolor="#b2182b",
                edgecolor="k",
                label="Red: SiGMA gated > SiGMA ungated (Δ>0)",
            ),
            Patch(
                facecolor="#2166ac",
                edgecolor="k",
                label="Blue: SiGMA ungated > SiGMA gated (Δ<0)",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="",
                label="g / u = gated / ungated mean±std (%)",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="",
                label="* p<0.05   ** p<0.01   *** p<0.001",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="",
                label="n.s.  paired t-test not significant (p≥0.05)",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=1,
            frameon=True,
            fontsize=8.5,
        )
        fig.tight_layout()
        path = out / f"fig_{slug}_delta_heatmap_H{H}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


def fetch_from_wandb(ds: str, hs: list[int]) -> pd.DataFrame:
    """Query W&B and return best-LR summary table for one dataset."""
    api = wandb.Api(timeout=180)
    rows: list[dict[str, object]] = []

    for L in LAYERS:
        for dh in DHS:
            for H in hs:
                print(f"  {ds} L={L} dh={dh} H={H}", flush=True)
                best_g: tuple[float, float, str, dict[int, float]] | None = None
                best_u: tuple[float, float, str, dict[int, float]] | None = None
                for lr in ("lr001", "lr01"):
                    g = group_by_seed(
                        api, f"tu_L{L}_dh{dh}_H{H}_{ds}_SiGMA_hetero_{lr}"
                    )
                    u = group_by_seed(
                        api, f"tu_L{L}_dh{dh}_H{H}_{ds}_SiGMA_ungated_{lr}"
                    )
                    if len(g) >= 5:
                        gm, gs = mean_std(list(g.values()))
                        if best_g is None or gm > best_g[0]:
                            best_g = (gm, gs, lr, g)
                    if len(u) >= 5:
                        um, us = mean_std(list(u.values()))
                        if best_u is None or um > best_u[0]:
                            best_u = (um, us, lr, u)
                if best_g is None or best_u is None:
                    print("    incomplete")
                    continue
                gm, gs, glr, gby = best_g
                um, us, ulr, uby = best_u
                seeds = sorted(set(gby) & set(uby))
                gv = [gby[s] for s in seeds]
                uv = [uby[s] for s in seeds]
                p = (
                    float(stats.ttest_rel(gv, uv).pvalue)
                    if len(seeds) >= 2
                    else float("nan")
                )
                rows.append(
                    {
                        "dataset": ds,
                        "L": L,
                        "d_h": dh,
                        "H": H,
                        "gated_mean": gm,
                        "gated_std": gs,
                        "gated_lr": glr,
                        "ungated_mean": um,
                        "ungated_std": us,
                        "ungated_lr": ulr,
                        "delta": gm - um,
                        "p_paired": p,
                        "n_paired": len(seeds),
                    }
                )
                print(f"    Δ={gm - um:+.2f} p={p:.3f}")
    return pd.DataFrame(rows)


def process_dataset(
    ds: str,
    hs: list[int],
    *,
    from_csv: bool,
) -> None:
    """Fetch/merge one dataset and write its heatmaps."""
    if ds not in DATASET_LABELS:
        raise ValueError(f"unknown dataset slug {ds!r}; choose from {sorted(DATASET_LABELS)}")
    ds_label = DATASET_LABELS[ds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{ds}_best_lr.csv"

    if from_csv:
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        if "lr" in df.columns and (df["lr"] == "best").any():
            df = df[df["lr"] == "best"].copy()
    else:
        print(f"=== fetch {ds_label} H={hs} ===")
        df_new = fetch_from_wandb(ds, hs)
        if csv_path.is_file():
            old = pd.read_csv(csv_path)
            if "lr" in old.columns and (old["lr"] == "best").any():
                old = old[old["lr"] == "best"].copy()
            old = old[~old["H"].isin(hs)]
            df = pd.concat([old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")

    plot_heatmaps(df[df["H"].isin(hs)], OUT_DIR, ds_label, hs)


def main() -> None:
    """CLI entry point for multi-dataset L×d_h×H heatmaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip W&B; plot from results/.../<ds>_best_lr.csv",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mutag"],
        help="W&B dataset slugs (default: mutag). "
        f"Choices: {', '.join(DATASET_LABELS)}",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Alias for a single --datasets entry (back-compat).",
    )
    parser.add_argument(
        "--H",
        type=int,
        nargs="+",
        default=None,
        help="H values to include (default: 64 8).",
    )
    args = parser.parse_args()
    hs = list(args.H) if args.H is not None else list(DEFAULT_HS)
    datasets = list(args.datasets)
    if args.dataset is not None:
        datasets = [args.dataset]

    for ds in datasets:
        process_dataset(ds, hs, from_csv=args.from_csv)


if __name__ == "__main__":
    main()
