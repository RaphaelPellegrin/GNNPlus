"""Train mode: heterogeneity profile (multi-trial per-graph test tracking)."""

from __future__ import annotations

import logging

from torch_geometric.graphgym.register import register_train


@register_train("heterogeneity")
def heterogeneity_train(loggers, loaders, model, optimizer, scheduler) -> None:
    """
    Run heterogeneity-profile experiment.

    Ignores the ``loggers`` / ``model`` passed from ``main.py`` (fresh per trial).
    Use ``--repeat 1`` on the CLI; trials are controlled by ``cfg.heterogeneity``.
    """
    from GNNPlus.experiments.heterogeneity_profile import run_heterogeneity_profile

    del loggers, loaders, model, optimizer, scheduler
    logging.info("Starting heterogeneity-profile training mode")
    run_heterogeneity_profile()
    logging.info("Heterogeneity-profile training complete")
