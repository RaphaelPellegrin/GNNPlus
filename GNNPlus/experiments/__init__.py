"""Heterogeneity-profile experiments (per-graph performance across test appearances)."""

from GNNPlus.experiments.track_avg_accuracy import (
    compute_average_per_graph,
    load_and_plot_average_per_graph,
    plot_average_per_graph,
)

__all__ = [
    'compute_average_per_graph',
    'load_and_plot_average_per_graph',
    'plot_average_per_graph',
]
