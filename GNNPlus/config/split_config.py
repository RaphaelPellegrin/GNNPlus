from torch_geometric.graphgym.register import register_config


@register_config('split')
def set_cfg_split(cfg):
    """Reconfigure the default config value for dataset split options.

    Returns:
        Reconfigured split configuration use by the experiment.
    """

    # Default to selecting the standard split that ships with the dataset
    cfg.dataset.split_mode = 'standard'

    # Choose a particular split to use if multiple splits are available
    cfg.dataset.split_index = 0

    # Dir to cache cross-validation splits
    cfg.dataset.split_dir = './splits'

    # Errica (ICLR 2020) protocol — vendored splits live under split_dir/errica/.
    cfg.dataset.errica_split_dir = ''
    # Override auto feature mode: chem | social_constant | social_degree
    cfg.dataset.errica_feature_mode = ''

    # Choose to run multiple splits in one program execution, if set,
    # takes the precedence over cfg.dataset.split_index for split selection
    cfg.run_multiple_splits = []
