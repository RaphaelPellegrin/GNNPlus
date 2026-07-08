from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = True
    cfg.gnn.ffn = True

    # Graph readout preset for ``mlp_graph`` head (MOE hybrid_readout_mlp).
    # Empty / mlp_graph: legacy ``layers_post_mp`` same-width hidden stack.
    # linear | narrow2 | pyramid | deep4
    cfg.gnn.readout_mlp = ''

    # Optional output gating on ``gcne`` / ``gcn`` layers (Level-1 fairness repro).
    # Values: '' (none) | headwise | elementwise
    cfg.gnn.gate = ''
    cfg.gnn.log_gate_stats = False
