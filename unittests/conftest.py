"""Shared pytest fixtures for GNNPlus unit tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from torch_geometric.graphgym.config import cfg, set_cfg

import GNNPlus  # noqa: F401  (registers the custom GraphGym config groups)


@pytest.fixture(autouse=True)
def _reset_graphgym_cfg() -> Iterator[None]:
    """Reset the global GraphGym ``cfg`` to registered defaults around each test.

    Layers read the module-level ``cfg`` imported at load time, so tests must
    mutate that object in place rather than build a separate ``CfgNode``.
    """
    set_cfg(cfg)
    yield
    set_cfg(cfg)
