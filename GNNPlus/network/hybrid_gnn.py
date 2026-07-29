"""Hybrid GNN: gated attention + message-passing heads (Heterogeneity_Profile)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import scatter

from GNNPlus.layer.gated_hybrid_layer import (
    AttnMaskType,
    AttnType,
    GateMode,
    GatedHybridGraphLayer,
    NormType,
    parse_hybrid_gnn_types,
)


class _PostHybridFFN(nn.Module):
    """Optional FFN after each hybrid block (matches GNNPlus conv layers)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.ff_linear1 = nn.Linear(dim, dim * 2)
        self.ff_linear2 = nn.Linear(dim * 2, dim)
        self.act = register.act_dict[cfg.gnn.act]()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout2(self.ff_linear2(self.dropout1(self.act(self.ff_linear1(self.norm(x))))))
        return x


@register_network('hybrid_gnn')
class HybridGNN(torch.nn.Module):
    """GNNPlus network with stacked :class:`GatedHybridGraphLayer` blocks.

    Set ``model.type: hybrid_gnn`` and tune ``gnn.hybrid.*`` in the yaml.
    Uses the same encoders, heads, and ``train.mode: custom`` pipeline as
    ``custom_gnn`` (MNIST, COCO, OGB, etc.).

    When ``posenc_RRWP.enable`` is True, absolute/relative RRWP encoders are
    applied after the GraphGym :class:`FeatureEncoder`. Relative RRWP may pad
    to a full graph for GRIT attention while storing sparse ``edge_index_mp`` /
    ``edge_attr_mp`` for message-passing heads.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, 'The inner and hidden dims must match.'

        hcfg = cfg.gnn.hybrid
        num_attn = int(hcfg.num_attn_heads)
        num_gnn = int(hcfg.num_gnn_heads)
        if num_attn < 0 or num_gnn < 0 or num_attn + num_gnn < 1:
            raise ValueError(
                'hybrid_gnn requires num_attn_heads + num_gnn_heads >= 1 '
                f'(got {num_attn} + {num_gnn})'
            )

        d_h = int(hcfg.d_h)
        attn_mask = cast(AttnMaskType, str(hcfg.attn_mask))
        from GNNPlus.layer.gated_hybrid_layer import _normalize_gate_mode

        gate_mode = cast(GateMode, _normalize_gate_mode(str(hcfg.gate)))
        mp_gate_raw = str(getattr(hcfg, "mp_gate", "") or "").strip()
        mp_gate_mode = (
            gate_mode
            if not mp_gate_raw
            else cast(GateMode, _normalize_gate_mode(mp_gate_raw))
        )
        norm_type = cast(NormType, str(hcfg.norm))
        mp_drop = float(hcfg.mp_dropout) if float(hcfg.mp_dropout) > 0 else float(cfg.gnn.dropout)
        gnn_types: List[str] = parse_hybrid_gnn_types(
            str(hcfg.gnn_types) if hcfg.gnn_types else None,
            num_gnn,
        )

        hybrid_residual = bool(getattr(hcfg, 'residual', True))
        identity_proj = bool(getattr(hcfg, 'identity_proj', False))
        attn_type = cast(
            AttnType,
            str(getattr(hcfg, 'attn_type', 'vanilla')).strip().lower(),
        )
        grit_cfg = getattr(hcfg, 'grit', None)
        grit_clamp = float(getattr(grit_cfg, 'clamp', 5.0)) if grit_cfg is not None else 5.0
        grit_edge_enhance = (
            bool(getattr(grit_cfg, 'edge_enhance', True)) if grit_cfg is not None else True
        )
        grit_act = str(getattr(grit_cfg, 'act', cfg.gnn.act)) if grit_cfg is not None else str(cfg.gnn.act)
        grit_use_bias = (
            bool(getattr(grit_cfg, 'use_bias', False)) if grit_cfg is not None else False
        )
        pad_to_full = (
            bool(getattr(grit_cfg, 'pad_to_full_graph', True)) if grit_cfg is not None else True
        )

        # Optional RRWP encoders (GRIT / grit attention). Applied after FeatureEncoder.
        self.rrwp_abs_encoder: Optional[nn.Module] = None
        self.rrwp_rel_encoder: Optional[nn.Module] = None
        rrwp_cfg = getattr(cfg, 'posenc_RRWP', None)
        if rrwp_cfg is not None and bool(getattr(rrwp_cfg, 'enable', False)):
            ksteps = int(rrwp_cfg.ksteps)
            dim_edge = int(getattr(cfg.gnn, 'dim_edge', 0)) or int(cfg.gnn.dim_inner)
            self.rrwp_abs_encoder = register.node_encoder_dict['rrwp_linear'](
                ksteps,
                cfg.gnn.dim_inner,
            )
            self.rrwp_rel_encoder = register.edge_encoder_dict['rrwp_linear'](
                ksteps,
                dim_edge,
                pad_to_full_graph=pad_to_full,
                add_node_attr_as_self_loop=False,
                fill_value=0.0,
            )

        self.layers = nn.ModuleList([
            GatedHybridGraphLayer(
                d_model=cfg.gnn.dim_inner,
                num_attn_heads=num_attn,
                num_gnn_heads=num_gnn,
                d_h=d_h,
                attn_mask_type=attn_mask,
                gate_mode=gate_mode,
                mp_gate_mode=mp_gate_mode,
                norm_type=norm_type,
                gnn_types=gnn_types,
                attn_dropout=float(hcfg.attn_dropout),
                mp_gnn_dropout=mp_drop,
                block_bn=bool(hcfg.block_bn),
                block_dropout=mp_drop,
                residual=hybrid_residual,
                identity_proj=identity_proj,
                attn_type=attn_type,
                edge_dim=cfg.gnn.dim_inner,
                grit_clamp=grit_clamp,
                grit_edge_enhance=grit_edge_enhance,
                grit_act=grit_act,
                grit_use_bias=grit_use_bias,
            )
            for _ in range(cfg.gnn.layers_mp)
        ])

        self.ffn_blocks: Optional[nn.ModuleList] = None
        if cfg.gnn.ffn:
            self.ffn_blocks = nn.ModuleList([
                _PostHybridFFN(cfg.gnn.dim_inner, cfg.gnn.dropout)
                for _ in range(cfg.gnn.layers_mp)
            ])

        gnn_head = register.head_dict[cfg.gnn.head]
        self.post_mp = gnn_head(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def _encode_batch(
        self, batch: Batch
    ) -> Tuple[
        torch.Tensor,
        Batch,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Run encoders and return dual edge sets for attn vs MP.

        Returns:
            ``(x, batch, edge_index_attn, edge_attr_attn,
            edge_index_mp, edge_attr_mp, edge_index, edge_attr)``
            where ``edge_index`` / ``edge_attr`` are the post-encoder defaults
            (full-graph when RRWP padded).
        """
        batch = self.encoder(batch)
        if self.rrwp_abs_encoder is not None:
            batch = self.rrwp_abs_encoder(batch)
        if self.rrwp_rel_encoder is not None:
            batch = self.rrwp_rel_encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        edge_attr = getattr(batch, 'edge_attr', None)
        edge_index = batch.edge_index
        edge_index_mp = getattr(batch, 'edge_index_mp', edge_index)
        edge_attr_mp = getattr(batch, 'edge_attr_mp', edge_attr)
        # After RRWP pad, batch.edge_* are the full-graph attention edges.
        edge_index_attn = edge_index
        edge_attr_attn = edge_attr
        return (
            batch.x,
            batch,
            edge_index_attn,
            edge_attr_attn,
            edge_index_mp,
            edge_attr_mp,
            edge_index,
            edge_attr,
        )

    def collect_gate_stats(self, batch: Batch) -> Dict[str, float]:
        """Per-layer headwise gate means from one forward (no prediction head).

        Keys match Heterogeneity_Profile: ``layer{i}/attn_{h}_gate_mean``,
        ``layer{i}/gnn_{h}_gate_mean``.
        """
        (
            x,
            batch,
            edge_index_attn,
            edge_attr_attn,
            edge_index_mp,
            edge_attr_mp,
            _edge_index,
            _edge_attr,
        ) = self._encode_batch(batch)
        all_stats: Dict[str, float] = {}
        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(
                x,
                edge_index_mp,
                batch.batch,
                edge_attr_mp,
                edge_index_attn=edge_index_attn,
                edge_attr_attn=edge_attr_attn,
                edge_index_mp=edge_index_mp,
                edge_attr_mp=edge_attr_mp,
                return_gate_stats=True,
            )
            assert isinstance(layer_out, tuple)
            x, layer_aux = layer_out
            for key, val in layer_aux.get('gate_stats', {}).items():
                all_stats[f'layer{layer_idx}/{key}'] = float(val)
            if self.ffn_blocks is not None:
                x = self.ffn_blocks[layer_idx](x)
        return all_stats

    def collect_per_graph_gates(self, batch: Batch) -> Dict[str, Any]:
        """Per-graph mean gate γ for each layer and head (no prediction head).

        For headwise gates, each node has a scalar γ; this averages over nodes
        within each graph. For elementwise gates, averages over nodes and the
        ``d_h`` feature dim.

        Returns:
            Dict with:
              ``attn``: ``FloatTensor [G, L, Na]``
              ``gnn``: ``FloatTensor [G, L, Ng]``
              ``y``: graph labels when present (``LongTensor [G]`` or None)
              ``num_graphs``: int
        """
        (
            x,
            batch,
            edge_index_attn,
            edge_attr_attn,
            edge_index_mp,
            edge_attr_mp,
            _edge_index,
            _edge_attr,
        ) = self._encode_batch(batch)
        graph_ids = batch.batch
        num_graphs = int(graph_ids.max().item()) + 1 if graph_ids.numel() else 0

        attn_layers: List[torch.Tensor] = []
        gnn_layers: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(
                x,
                edge_index_mp,
                batch.batch,
                edge_attr_mp,
                edge_index_attn=edge_index_attn,
                edge_attr_attn=edge_attr_attn,
                edge_index_mp=edge_index_mp,
                edge_attr_mp=edge_attr_mp,
                return_gate_stats=True,
            )
            assert isinstance(layer_out, tuple)
            x, layer_aux = layer_out
            gate_values = cast(
                Dict[str, List[torch.Tensor]],
                layer_aux.get('gate_values', {}),
            )

            def _heads_to_graph(heads: List[torch.Tensor]) -> torch.Tensor:
                """Stack per-head per-graph means into ``[G, H]``."""
                if not heads:
                    return torch.zeros(num_graphs, 0, device=x.device)
                cols: List[torch.Tensor] = []
                for gamma in heads:
                    # [N, 1] or [N, d_h] -> [N] mean over feature dims.
                    node_gate = gamma.detach().float().mean(dim=-1)
                    graph_gate = scatter(
                        node_gate,
                        graph_ids,
                        dim=0,
                        dim_size=num_graphs,
                        reduce='mean',
                    )
                    cols.append(graph_gate)
                return torch.stack(cols, dim=-1)

            attn_layers.append(_heads_to_graph(gate_values.get('attn', [])))
            gnn_layers.append(_heads_to_graph(gate_values.get('gnn', [])))
            if self.ffn_blocks is not None:
                x = self.ffn_blocks[layer_idx](x)

        attn = (
            torch.stack(attn_layers, dim=1)
            if attn_layers
            else torch.zeros(num_graphs, 0, 0)
        )
        gnn = (
            torch.stack(gnn_layers, dim=1)
            if gnn_layers
            else torch.zeros(num_graphs, 0, 0)
        )
        y: Optional[torch.Tensor] = None
        if hasattr(batch, 'y') and batch.y is not None:
            y_t = batch.y
            if y_t.dim() == 0:
                y = y_t.view(1)
            elif y_t.numel() == num_graphs:
                y = y_t.view(num_graphs)
            else:
                # Node-level labels: take first label per graph is undefined —
                # leave None for graph-level dump safety.
                y = None
        return {
            'attn': attn.cpu(),
            'gnn': gnn.cpu(),
            'y': None if y is None else y.detach().cpu(),
            'num_graphs': num_graphs,
        }

    def forward(self, batch: Batch) -> Batch:
        """Run encoder, hybrid blocks, optional FFN, and prediction head."""
        (
            x,
            batch,
            edge_index_attn,
            edge_attr_attn,
            edge_index_mp,
            edge_attr_mp,
            _edge_index,
            _edge_attr,
        ) = self._encode_batch(batch)
        for i, layer in enumerate(self.layers):
            x = cast(
                torch.Tensor,
                layer(
                    x,
                    edge_index_mp,
                    batch.batch,
                    edge_attr_mp,
                    edge_index_attn=edge_index_attn,
                    edge_attr_attn=edge_attr_attn,
                    edge_index_mp=edge_index_mp,
                    edge_attr_mp=edge_attr_mp,
                ),
            )
            if self.ffn_blocks is not None:
                x = self.ffn_blocks[i](x)
            batch.x = x

        return self.post_mp(batch)

    def forward_all_layer_features(
        self, batch: Batch
    ) -> Tuple[List[torch.Tensor], Batch]:
        """Return node features after every hybrid block (before the head).

        Each entry is ``x`` after layer ``i`` (and its FFN, if any). Updates
        ``batch.x`` to the final layer features.

        Returns:
            ``(layer_features, batch)`` where ``layer_features[i]`` has shape
            ``[N, F]`` for layer ``i``.
        """
        (
            x,
            batch,
            edge_index_attn,
            edge_attr_attn,
            edge_index_mp,
            edge_attr_mp,
            _edge_index,
            _edge_attr,
        ) = self._encode_batch(batch)
        layer_features: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x = cast(
                torch.Tensor,
                layer(
                    x,
                    edge_index_mp,
                    batch.batch,
                    edge_attr_mp,
                    edge_index_attn=edge_index_attn,
                    edge_attr_attn=edge_attr_attn,
                    edge_index_mp=edge_index_mp,
                    edge_attr_mp=edge_attr_mp,
                ),
            )
            if self.ffn_blocks is not None:
                x = self.ffn_blocks[i](x)
            batch.x = x
            layer_features.append(x)
        return layer_features, batch

    def forward_node_features(self, batch: Batch) -> Tuple[torch.Tensor, Batch]:
        """Return last-layer node features ``x`` (before the prediction head).

        Convenience wrapper around :meth:`forward_all_layer_features`.
        """
        layer_features, batch = self.forward_all_layer_features(batch)
        return layer_features[-1], batch
