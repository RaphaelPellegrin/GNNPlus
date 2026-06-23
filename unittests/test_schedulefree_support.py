"""Tests for ScheduleFree optimizer integration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from GNNPlus.optimizer.schedulefree_support import (
    ScheduleFreeScheduler,
    get_optimizer_logged_lr,
    uses_schedulefree_scheduler,
)


def test_uses_schedulefree_scheduler() -> None:
    """ScheduleFree scheduler name is recognized."""
    assert uses_schedulefree_scheduler("schedulefree")
    assert not uses_schedulefree_scheduler("cosine_with_warmup")


def test_schedulefree_scheduler_get_last_lr_from_optimizer() -> None:
    """Placeholder scheduler reads LR from the wrapped optimizer."""
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = ScheduleFreeScheduler(opt)
    assert sched.get_last_lr() == [0.01]


def test_get_optimizer_logged_lr_prefers_scheduled_lr() -> None:
    """Logged LR uses scheduled_lr when present (ScheduleFree warmup)."""
    opt = SimpleNamespace(param_groups=[{"lr": 0.01, "scheduled_lr": 0.002}])
    assert get_optimizer_logged_lr(opt) == 0.002  # type: ignore[arg-type]
