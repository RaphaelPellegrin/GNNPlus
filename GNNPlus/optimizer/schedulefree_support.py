"""ScheduleFree optimizer helpers (ported from Heterogeneity_Profile)."""

from __future__ import annotations

from typing import Any, Optional, cast

from torch.optim import Optimizer

try:
    from schedulefree import AdamWScheduleFree
except ImportError:  # pragma: no cover - installed on cluster via requirements
    AdamWScheduleFree = None


class ScheduleFreeScheduler:
    """No-op scheduler; :class:`AdamWScheduleFree` owns LR scheduling."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def step(self, *_args: object, **_kwargs: object) -> None:
        """ScheduleFree does not use an external per-epoch scheduler."""

    def get_last_lr(self) -> list[float]:
        """Return the effective LR for logging."""
        return [get_optimizer_logged_lr(self.optimizer)]

    def state_dict(self) -> dict[str, object]:
        """Checkpoint hook (no state)."""
        return {}

    def load_state_dict(self, _state_dict: dict[str, object]) -> None:
        """Checkpoint hook (no state)."""


def is_schedulefree_optimizer(optimizer: Optimizer) -> bool:
    """Return whether ``optimizer`` is a ScheduleFree variant."""
    if AdamWScheduleFree is not None and isinstance(optimizer, AdamWScheduleFree):
        return True
    return optimizer.__class__.__name__ == "AdamWScheduleFree"


def get_optimizer_logged_lr(optimizer: Optimizer) -> float:
    """Return the LR value to log (handles ScheduleFree ``scheduled_lr``)."""
    if not optimizer.param_groups:
        return 0.0
    group = optimizer.param_groups[0]
    scheduled_lr = group.get("scheduled_lr", None)
    if scheduled_lr is not None:
        return float(scheduled_lr)
    return float(group["lr"])


def set_optimizer_train_mode(optimizer: Optional[Optimizer]) -> None:
    """Switch optimizers with explicit train/eval buffers into train mode."""
    if optimizer is None or not hasattr(optimizer, "train"):
        return
    cast(Any, optimizer).train()


def set_optimizer_eval_mode(optimizer: Optional[Optimizer]) -> None:
    """Switch optimizers with explicit train/eval buffers into eval mode."""
    if optimizer is None or not hasattr(optimizer, "eval"):
        return
    cast(Any, optimizer).eval()


def uses_schedulefree_scheduler(scheduler_name: str) -> bool:
    """Return whether the config scheduler defers LR to ScheduleFree."""
    return str(scheduler_name).lower() in ("schedulefree", "none_schedulefree")
