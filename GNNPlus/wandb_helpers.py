"""W&B summary helpers for GNNPlus training."""

from __future__ import annotations

from typing import Any

from torch_geometric.graphgym.config import cfg

from GNNPlus.preprocessing.graph_augmentations import parse_cfg_bool


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
