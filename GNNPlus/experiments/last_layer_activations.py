"""Per-graph multi-layer activation diagnostics (all layers × graph index)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import scatter


def per_graph_mean_node_l2(
    x: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """Mean over nodes of ``||x_v||_2`` for each graph in a batch.

    Args:
        x: Node features ``[N, F]``.
        batch_index: Graph id per node ``[N]``.

    Returns:
        Tensor ``[B]`` of mean node L2 norms.
    """
    node_l2 = torch.linalg.vector_norm(x, ord=2, dim=-1)
    return scatter(node_l2, batch_index, dim=0, reduce='mean')


def per_graph_pooled_l2(
    x: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """L2 norm of mean-pooled graph embedding for each graph.

    Args:
        x: Node features ``[N, F]``.
        batch_index: Graph id per node ``[N]``.

    Returns:
        Tensor ``[B]`` of pooled-embedding L2 norms.
    """
    pooled = scatter(x, batch_index, dim=0, reduce='mean')
    return torch.linalg.vector_norm(pooled, ord=2, dim=-1)


def plot_activations_by_index(
    graph_indices: np.ndarray,
    activations: np.ndarray,
    *,
    dataset_name: str,
    model_tag: str,
    ylabel: str,
    output_path: Path,
    title_suffix: str = '',
    title_kind: str = 'activation',
) -> Path:
    """Scatter/line plot: activation vs graph index.

    Args:
        graph_indices: Graph storage indices.
        activations: Per-graph activation scalars.
        dataset_name: Dataset name for title.
        model_tag: Model label for title/filename context.
        ylabel: Y-axis label.
        output_path: Destination PNG path.
        title_suffix: Optional extra title text.
        title_kind: Short kind string in the title.

    Returns:
        The written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(graph_indices, activations, marker='o', markersize=2, linewidth=0.8, alpha=0.85)
    ax.set_xlabel('Graph index')
    ax.set_ylabel(ylabel)
    title = f'{dataset_name} · {model_tag} · {title_kind}'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_activations_sorted(
    activations: np.ndarray,
    *,
    dataset_name: str,
    model_tag: str,
    ylabel: str,
    output_path: Path,
    title_kind: str = 'activation',
) -> Path:
    """Plot activations sorted ascending (heterogeneity shape).

    Args:
        activations: Per-graph activation scalars.
        dataset_name: Dataset name.
        model_tag: Model label.
        ylabel: Y-axis label.
        output_path: Destination PNG.
        title_kind: Short kind string in the title.

    Returns:
        The written path.
    """
    order = np.argsort(activations)
    sorted_acts = activations[order]
    xs = np.arange(len(sorted_acts))
    return plot_activations_by_index(
        xs,
        sorted_acts,
        dataset_name=dataset_name,
        model_tag=model_tag,
        ylabel=ylabel,
        output_path=output_path,
        title_suffix='sorted by activation',
        title_kind=title_kind,
    )


def plot_all_layers_by_index(
    graph_indices: np.ndarray,
    layer_activations: np.ndarray,
    *,
    dataset_name: str,
    model_tag: str,
    output_path: Path,
    title_suffix: str = '',
    ylabel: str = 'Mean node ‖h‖₂',
) -> Path:
    """Overlay one curve per layer: activation vs graph index.

    Args:
        graph_indices: Graph indices ``[G]``.
        layer_activations: Array ``[L, G]`` of per-layer per-graph activations.
        dataset_name: Dataset name.
        model_tag: Model label.
        output_path: Destination PNG.
        title_suffix: Optional title text (e.g. epoch tag).
        ylabel: Y-axis label.

    Returns:
        Written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_layers = int(layer_activations.shape[0])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = plt.get_cmap('viridis')
    for li in range(n_layers):
        color = cmap(li / max(n_layers - 1, 1))
        ax.plot(
            graph_indices,
            layer_activations[li],
            marker='o',
            markersize=1.5,
            linewidth=0.7,
            alpha=0.75,
            color=color,
            label=f'L{li}',
        )
    ax.set_xlabel('Graph index')
    ax.set_ylabel(ylabel)
    title = f'{dataset_name} · {model_tag} · all-layer activations'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if n_layers <= 16:
        ax.legend(ncol=min(n_layers, 8), fontsize=8, frameon=False)
    else:
        ax.legend(
            ncol=8,
            fontsize=7,
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_layers_heatmap(
    layer_activations: np.ndarray,
    *,
    dataset_name: str,
    model_tag: str,
    output_path: Path,
    title_suffix: str = '',
) -> Path:
    """Heatmap of activations with layers on y and graph index on x.

    Args:
        layer_activations: Array ``[L, G]``.
        dataset_name: Dataset name.
        model_tag: Model label.
        output_path: Destination PNG.
        title_suffix: Optional title text.

    Returns:
        Written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(3.0, 0.35 * layer_activations.shape[0] + 1.5)))
    im = ax.imshow(layer_activations, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_xlabel('Graph index')
    ax.set_ylabel('Layer')
    ax.set_yticks(np.arange(layer_activations.shape[0]))
    ax.set_yticklabels([f'L{i}' for i in range(layer_activations.shape[0])])
    title = f'{dataset_name} · {model_tag} · activation heatmap'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='Mean node ‖h‖₂')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_activations_csv(
    path: Path,
    graph_indices: Sequence[int],
    mean_node_l2: Sequence[float],
    pooled_l2: Sequence[float],
    labels: Optional[Sequence[int]] = None,
) -> Path:
    """Write per-graph single-layer activation CSV (legacy last-layer format).

    Args:
        path: Destination CSV.
        graph_indices: Global graph indices.
        mean_node_l2: Mean node L2 per graph.
        pooled_l2: Pooled embedding L2 per graph.
        labels: Optional graph labels.

    Returns:
        Written path.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['graph_idx', 'mean_node_l2', 'pooled_emb_l2']
    if labels is not None:
        fields.append('label')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, gidx in enumerate(graph_indices):
            row: Dict[str, object] = {
                'graph_idx': int(gidx),
                'mean_node_l2': float(mean_node_l2[i]),
                'pooled_emb_l2': float(pooled_l2[i]),
            }
            if labels is not None:
                row['label'] = int(labels[i])
            writer.writerow(row)
    return path


def write_all_layers_csv(
    path: Path,
    graph_indices: Sequence[int],
    layer_activations: np.ndarray,
    labels: Optional[Sequence[int]] = None,
) -> Path:
    """Write per-graph, per-layer mean node L2 CSV.

    Args:
        path: Destination CSV.
        graph_indices: Global graph indices ``[G]``.
        layer_activations: Array ``[L, G]``.
        labels: Optional graph labels ``[G]``.

    Returns:
        Written path.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_layers = int(layer_activations.shape[0])
    fields = ['graph_idx'] + [f'layer{i}_mean_node_l2' for i in range(n_layers)]
    if labels is not None:
        fields.append('label')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gi, gidx in enumerate(graph_indices):
            row: Dict[str, object] = {'graph_idx': int(gidx)}
            for li in range(n_layers):
                row[f'layer{li}_mean_node_l2'] = float(layer_activations[li, gi])
            if labels is not None:
                row['label'] = int(labels[gi])
            writer.writerow(row)
    return path


def dump_all_layer_plots(
    out_dir: Path,
    *,
    dataset_name: str,
    model_tag: str,
    graph_indices: np.ndarray,
    layer_activations: np.ndarray,
    labels: Optional[np.ndarray],
    epoch_tag: str,
) -> Dict[str, str]:
    """Write CSV + overlay/heatmap/last-layer plots for one snapshot.

    Args:
        out_dir: Output directory (created if needed).
        dataset_name: Dataset name.
        model_tag: Model tag.
        graph_indices: ``[G]``.
        layer_activations: ``[L, G]``.
        labels: Optional ``[G]``.
        epoch_tag: Snapshot label (e.g. ``last``, ``mid``, ``best``).

    Returns:
        Dict of artifact paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'{dataset_name}_{model_tag}_{epoch_tag}'
    paths: Dict[str, str] = {}

    csv_path = out_dir / f'{prefix}_all_layers.csv'
    write_all_layers_csv(
        csv_path,
        graph_indices.tolist(),
        layer_activations,
        labels.tolist() if labels is not None else None,
    )
    paths['csv'] = str(csv_path)

    paths['overlay'] = str(
        plot_all_layers_by_index(
            graph_indices,
            layer_activations,
            dataset_name=dataset_name,
            model_tag=model_tag,
            output_path=out_dir / f'{prefix}_all_layers_by_index.png',
            title_suffix=epoch_tag,
        )
    )
    paths['heatmap'] = str(
        plot_layers_heatmap(
            layer_activations,
            dataset_name=dataset_name,
            model_tag=model_tag,
            output_path=out_dir / f'{prefix}_all_layers_heatmap.png',
            title_suffix=epoch_tag,
        )
    )
    # Last layer alone (matches original single-layer figure).
    last = layer_activations[-1]
    paths['last_by_index'] = str(
        plot_activations_by_index(
            graph_indices,
            last,
            dataset_name=dataset_name,
            model_tag=model_tag,
            ylabel='Mean node ‖h‖₂ (last layer)',
            output_path=out_dir / f'{prefix}_last_layer_by_index.png',
            title_suffix=epoch_tag,
            title_kind='last-layer activation',
        )
    )
    paths['last_sorted'] = str(
        plot_activations_sorted(
            last,
            dataset_name=dataset_name,
            model_tag=model_tag,
            ylabel='Mean node ‖h‖₂ (last layer)',
            output_path=out_dir / f'{prefix}_last_layer_sorted.png',
            title_kind='last-layer activation',
        )
    )
    return paths
