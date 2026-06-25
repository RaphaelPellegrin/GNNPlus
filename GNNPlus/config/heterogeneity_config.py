from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_heterogeneity")
def set_cfg_heterogeneity(cfg: CN) -> None:
    """Heterogeneity-profile experiment settings (per-graph test tracking)."""

    cfg.heterogeneity = CN()

    # Target test-set appearances per graph before stopping (default 10).
    cfg.heterogeneity.required_test_appearances = 10

    # Safety cap on number of training trials.
    cfg.heterogeneity.max_trials = 200

    # Base seed for trial t uses base_seed + (t - 1).
    cfg.heterogeneity.base_seed = -1

    # Use random train/val/test each trial (Heterogeneity_Profile-style).
    cfg.heterogeneity.use_random_split = True
    cfg.heterogeneity.split = [0.5, 0.25, 0.25]

    # Write pickle + plots under cfg.out_dir.
    cfg.heterogeneity.save_pickle = True
    cfg.heterogeneity.plot = True
