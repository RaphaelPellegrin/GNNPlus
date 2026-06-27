# GRIT (Graph Inductive Bias Transformer)

Standalone GRIT configs matching [Ma et al., ICML 2023](https://arxiv.org/pdf/2305.17589) /
[official implementation](https://github.com/LiamMa/GRIT).

## Standalone model

```yaml
model:
  type: GritTransformer
posenc_RRWP:
  enable: True
  ksteps: 21
gt:
  layers: 10
  n_heads: 8
  dim_hidden: 64
gnn:
  dim_inner: 64   # must match gt.dim_hidden
```

Example: `configs/grit/pattern-grit-rrwp.yaml`

```bash
python main.py --cfg configs/grit/pattern-grit-rrwp.yaml seed 0 wandb.use True
```

## Hybrid MP head

Set `gnn.hybrid.gnn_types: "GRIT"` (requires `d_h` divisible by `gnn.grit.n_heads`, default 8).

Example: `configs/gated_hybrid/pattern-grit-repro-a1g1.yaml`

Hybrid GRIT uses **sparse graph edges** (same as GCN/GINE heads). Full paper GRIT uses RRWP + full-graph attention via the standalone `GritTransformer` network.

## Dependencies

`opt_einsum` (in `requirements-cluster.txt`). RRWP precompute uses `torch_sparse` (already in cluster env).
