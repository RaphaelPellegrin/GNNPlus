"""Utilities for tracking and plotting average accuracy/error per graph across test appearances."""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def compute_average_per_graph(
    graph_dict: Dict[int, List[Union[int, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average accuracy (classification) or average error (regression) per graph.

    Args:
        graph_dict: ``{graph_idx: [value1, value2, ...]}`` where values are 0/1
            (classification) or per-graph errors (regression).

    Returns:
        Tuple of (graph_indices, average_values) as numpy arrays.
    """
    graph_indices: List[int] = []
    average_values: List[float] = []

    for graph_idx in sorted(graph_dict.keys()):
        values = graph_dict[graph_idx]
        if len(values) > 0:
            average_values.append(float(np.mean(values)))
            graph_indices.append(int(graph_idx))

    return np.array(graph_indices), np.array(average_values)


def get_detailed_model_name(
    layer_type: str,
    layer_types: Optional[list] = None,
    router_type: str = "MLP",
    is_encoding_moe: bool = False,
    num_layers: Optional[int] = None,
) -> str:
    """Generate a detailed model name (Heterogeneity_Profile-compatible)."""
    depth_suffix = f"_L{num_layers}" if num_layers is not None else ""

    if is_encoding_moe:
        return f"{layer_type}_router_{router_type}{depth_suffix}"
    if layer_types is not None:
        expert_combo = "_".join(layer_types)
        return f"{layer_type}_router_{router_type}_{expert_combo}{depth_suffix}"
    return f"{layer_type}{depth_suffix}" if num_layers is not None else layer_type


def get_gnnplus_model_slug(cfg: Any) -> str:
    """
    Short filesystem-safe model identifier for pickles and plots.

    Examples:
        ``gcne`` (custom_gnn), ``hybrid_a2g2_GCN-GINE_dh16`` (hybrid_gnn).
    """
    if str(cfg.model.type) == "hybrid_gnn":
        hybrid = cfg.gnn.hybrid
        gnn_types = str(hybrid.gnn_types).replace(",", "-").replace(" ", "")
        return (
            f"hybrid_a{int(hybrid.num_attn_heads)}g{int(hybrid.num_gnn_heads)}"
            f"_{gnn_types}_dh{int(hybrid.d_h)}"
        )
    layer_type = str(getattr(cfg.gnn, "layer_type", cfg.model.type))
    return layer_type


def infer_plot_task_type(dataset_task_type: str) -> str:
    """Map GNNPlus dataset task type to plot task type (classification vs regression)."""
    if dataset_task_type == "regression":
        return "regression"
    return "classification"


def plot_average_per_graph(
    graph_indices: np.ndarray,
    average_values: np.ndarray,
    dataset_name: str,
    layer_type: str,
    encoding: Optional[str],
    num_layers: int,
    task_type: str = "classification",
    output_dir: str = "results",
    save_filename: Optional[str] = None,
    layer_types: Optional[list] = None,
    router_type: str = "MLP",
    is_encoding_moe: bool = False,
) -> str:
    """Plot average accuracy (classification) or average error (regression) per graph."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if task_type == "classification":
        ylabel = "Average Accuracy (%)"
        title_suffix = "Accuracy"
        y_values_plot: np.ndarray = average_values * 100
        ax.set_ylim(-5, 105)
    else:
        ylabel = "Average Error (MAE)"
        title_suffix = "Error"
        y_values_plot = average_values

    ax.scatter(graph_indices, y_values_plot, alpha=0.6, s=20)
    ax.set_xlabel("Graph Index", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    title_parts = [
        f"Average {title_suffix} per Graph",
        f"Dataset: {dataset_name}",
        f"Model: {layer_type} ({num_layers} layers)",
    ]
    if encoding:
        title_parts.append(f"Encoding: {encoding}")
    ax.set_title(" | ".join(title_parts), fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    mean_value = float(np.mean(y_values_plot).item())
    std_value = float(np.std(y_values_plot).item())
    min_value = float(np.min(y_values_plot).item())
    max_value = float(np.max(y_values_plot).item())
    stats_text = (
        f"Mean: {mean_value:.2f} | Std: {std_value:.2f} | "
        f"Min: {min_value:.2f} | Max: {max_value:.2f} | "
        f"Graphs: {len(graph_indices)}"
    )
    ax.text(
        0.5,
        -0.1,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    layers_dir = os.path.join(output_dir, f"{num_layers}_layers")
    os.makedirs(layers_dir, exist_ok=True)
    if save_filename is None:
        encoding_str = f"_{encoding}" if encoding else ""
        detailed_model_name = get_detailed_model_name(
            layer_type, layer_types, router_type, is_encoding_moe, num_layers
        )
        save_filename = os.path.join(
            layers_dir,
            f"{dataset_name}_{detailed_model_name}{encoding_str}"
            f"_avg_{task_type}_per_graph.png",
        )
    elif not os.path.isabs(save_filename):
        save_filename = os.path.join(output_dir, save_filename)

    fig.savefig(save_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_filename


def load_and_plot_average_per_graph(
    pickle_filepath: str,
    dataset_name: str,
    layer_type: str,
    encoding: Union[str, None],
    num_layers: int,
    task_type: str = "classification",
    output_dir: str = "results",
    layer_types: Optional[list] = None,
    router_type: str = "MLP",
    skip_connection: bool = False,
    normalize_features: bool = False,
    is_encoding_moe: bool = False,
    model_slug: Optional[str] = None,
) -> Tuple[str, str]:
    """Load graph_dict pickle and create by_index and by_accuracy heterogeneity plots."""
    with open(pickle_filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "graph_dict" in data:
        graph_dict = data["graph_dict"]
        if model_slug is None and "model_slug" in data:
            model_slug = str(data["model_slug"])
        if "dataset_name" in data:
            dataset_name = str(data["dataset_name"])
        if "num_layers" in data:
            num_layers = int(data["num_layers"])
        if "task_type" in data:
            task_type = infer_plot_task_type(str(data["task_type"]))
    else:
        graph_dict = data

    graph_indices, average_values = compute_average_per_graph(graph_dict)
    if len(graph_indices) == 0:
        print(f"No data found in {pickle_filepath}, skipping plot generation")
        return "", ""

    detailed_model_name = model_slug or get_detailed_model_name(
        layer_type, layer_types, router_type, is_encoding_moe, num_layers
    )
    skip_str = "skip_true" if skip_connection else "skip_false"
    norm_str = "norm_true" if normalize_features else "norm_false"
    encoding_str = encoding if encoding else "none"
    encoding_suffix = f"_encodings_{encoding_str}"

    original_plot_path = plot_average_per_graph(
        graph_indices,
        average_values,
        dataset_name,
        detailed_model_name,
        encoding,
        num_layers,
        task_type,
        output_dir,
        save_filename=(
            f"{dataset_name}_{detailed_model_name}_{skip_str}_{norm_str}"
            f"{encoding_suffix}_by_index.png"
        ),
    )

    sort_indices = np.argsort(average_values)[::-1]
    sorted_plot_path = plot_average_per_graph(
        graph_indices[sort_indices],
        average_values[sort_indices],
        dataset_name,
        detailed_model_name,
        encoding,
        num_layers,
        task_type,
        output_dir,
        save_filename=(
            f"{dataset_name}_{detailed_model_name}_{skip_str}_{norm_str}"
            f"{encoding_suffix}_by_accuracy.png"
        ),
    )
    return original_plot_path, sorted_plot_path


def load_and_plot_gnnplus_pickle(
    pickle_filepath: str,
    output_dir: str = "results",
) -> Tuple[str, str]:
    """Load a GNNPlus heterogeneity pickle (with metadata) and write both plots."""
    with open(pickle_filepath, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)

    return load_and_plot_average_per_graph(
        pickle_filepath=pickle_filepath,
        dataset_name=str(data.get("dataset_name", "dataset")),
        layer_type=str(data.get("model_slug", "model")),
        encoding=data.get("encoding"),
        num_layers=int(data.get("num_layers", 1)),
        task_type=infer_plot_task_type(str(data.get("task_type", "classification"))),
        output_dir=output_dir,
        model_slug=str(data.get("model_slug", "model")),
    )
