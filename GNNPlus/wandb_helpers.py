"""W&B summary helpers for GNNPlus training."""

from __future__ import annotations

import os
from typing import Any

from torch_geometric.graphgym.config import cfg

from GNNPlus.preprocessing.graph_augmentations import parse_cfg_bool
from GNNPlus.utils import cfg_to_dict, make_wandb_name


def _parse_wandb_tags() -> list[str]:
    """Return W&B tags from ``cfg.wandb.tags`` (list or comma-separated string)."""
    raw = getattr(cfg.wandb, "tags", None)
    if raw is None or raw == "" or raw == []:
        tags: list[str] = []
    elif isinstance(raw, (list, tuple)):
        tags = [str(tag).strip() for tag in raw if str(tag).strip()]
    else:
        tags = [part.strip() for part in str(raw).split(",") if part.strip()]
    return _append_slurm_wandb_tags(tags)


def _append_slurm_wandb_tags(tags: list[str]) -> list[str]:
    """Add ``job_<SLURM_JOB_ID>`` (and array task) tags when running under SLURM."""
    extra: list[str] = []
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if job_id:
        extra.append(f"job_{job_id}")
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if array_job and array_task:
        extra.append(f"array_{array_job}_{array_task}")
    if not extra:
        return tags
    return list(dict.fromkeys([*tags, *extra]))


def init_wandb_run() -> Any | None:
    """Initialize W&B once per process if ``cfg.wandb.use`` (no-op if already active)."""
    if not cfg.wandb.use:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("WandB is not installed.") from exc
    if wandb.run is not None:
        return wandb.run

    wandb_name = make_wandb_name(cfg) if cfg.wandb.name == "" else cfg.wandb.name
    wandb_kwargs: dict[str, object] = {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
        "name": wandb_name,
    }
    wandb_group = str(getattr(cfg.wandb, "group", "") or "").strip()
    if wandb_group:
        wandb_kwargs["group"] = wandb_group
    wandb_tags = _parse_wandb_tags()
    if wandb_tags:
        wandb_kwargs["tags"] = wandb_tags
    run = wandb.init(**wandb_kwargs)
    run.config.update(cfg_to_dict(cfg))
    record_preprocessing_wandb_flags()
    return run


def _effective_readout_preset() -> str:
    """Return the active readout preset name (MOE ``hybrid_readout_mlp`` alias)."""
    preset = str(getattr(cfg.gnn, 'readout_mlp', '') or '').strip().lower()
    return preset if preset not in ('', 'mlp_graph', 'legacy') else 'mlp_graph'


def record_preprocessing_wandb_flags() -> None:
    """Log graph preprocessing and hybrid readout flags on the active W&B run.

    Writes both GNNPlus keys (``dataset/...``, ``preprocess/...``) and MOE-compatible
    keys (``add_virtual_nodes``, ``hybrid_readout_mlp``) for cross-project filters.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    add_vn = parse_cfg_bool(getattr(cfg.dataset, 'add_virtual_nodes', False))
    num_vn = int(getattr(cfg.dataset, 'num_virtual_nodes', 0) or 0)
    if not add_vn:
        num_vn = 0
    readout = _effective_readout_preset()

    summary: dict[str, Any] = {
        'preprocess/add_virtual_nodes': add_vn,
        'preprocess/num_virtual_nodes': num_vn,
        'preprocess/readout_mlp': readout,
        'add_virtual_nodes': add_vn,
        'num_virtual_nodes': num_vn,
        'hybrid_readout_mlp': readout,
    }
    if cfg.model.type == 'hybrid_gnn':
        summary['hybrid/readout_mlp'] = readout

    config_updates: dict[str, Any] = {
        'dataset/add_virtual_nodes': add_vn,
        'dataset/num_virtual_nodes': num_vn,
        'gnn/readout_mlp': readout,
        'add_virtual_nodes': add_vn,
        'num_virtual_nodes': num_vn,
        'hybrid_readout_mlp': readout,
    }

    wandb.run.summary.update(summary)
    wandb.config.update(config_updates, allow_val_change=True)

    extra_tags: list[str] = ['gnnplus_preprocess']
    if add_vn:
        extra_tags.extend(['virtual_nodes', f'vn{num_vn}'])
    if readout != 'mlp_graph':
        extra_tags.append(f'readout_{readout}')
    wandb.run.tags = list(dict.fromkeys([*list(wandb.run.tags), *extra_tags]))
