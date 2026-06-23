# Gated hybrid GNN (`model.type: hybrid_gnn`)

Attention + message-passing heads with sigmoid gating (ported from Heterogeneity_Profile).

## Architecture (cluster default)

- **2 attention heads** + **2 MP heads** (`a2g2`), gate `headwise`, `d_h: 16`
- **MP types**: `GCN,GIN` (superpixels / TUDataset); `GCN,GINE` (OGB molecular)
- **Outer hyperparams**: copied from `configs/gcn/<dataset>.yaml` (GNN+ paper gcne)
- **W&B**: project `MOE_6`, tags `gnnplus`, `hybrid_gnn`, `cluster`

## Cluster submit (priority order)

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export ENV_NAME=gnnplus
conda deactivate 2>/dev/null || true
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Smoke (~5 epochs)
sbatch bash_interface/cluster/smoke_test_hybrid_mnist.sh

# By priority tier (parallel within each sbatch call)
bash bash_interface/cluster/submit_hybrid_suite.sh tier1    # MNIST, CIFAR10
bash bash_interface/cluster/submit_hybrid_suite.sh tier2    # COCO, Pascal VOC
bash bash_interface/cluster/submit_hybrid_suite.sh tier3    # peptides func + struct
bash bash_interface/cluster/submit_hybrid_suite.sh tier4    # ENZYMES
bash bash_interface/cluster/submit_hybrid_suite.sh tier5    # hiv, ppa, zinc, …

# Or everything at once
bash bash_interface/cluster/submit_hybrid_suite.sh all
```

| Tier | Datasets | Seeds | Notes |
|------|----------|-------|-------|
| 1 | mnist, cifar10 | 2 | GNNBenchmark superpixels |
| 2 | coco, voc | 2 | COCO needs **128GB** mem |
| 3 | peptides-func, peptides-struct | 4 | OGB; GCN+GINE heads |
| 4 | enzymes | 2 | TUDataset; baseline `configs/gcn/enzymes.yaml` |
| 5 | hiv, ppa, zinc, mutag, mal, pcba, code2, cluster, pattern | 1–5 | Remaining GNN+ paper suite |

Configs live in `configs/gated_hybrid/<stem>.yaml` where `<stem>` matches `configs/gcn/` (e.g. `hiv` not `molhiv`).

## MP head types (`gnn.hybrid.gnn_types`)

| String | Implementation | Edge features |
|--------|----------------|-----------------|
| `GCN`, `SAGE`, `GIN`, `GAT` | PyG conv at `d_h` | No |
| `GCNE` | `GCNConvWithEdges` (matches GNN+ `gcne` baseline) | Yes |
| `GATEDGCN` | `GatedGCNLayer` (matches GNN+ `gatedgcn` baseline) | Yes |
| `RESGATEDGCN` | PyG `ResGatedGraphConv` (legacy; see below) | No |
| `GINE`, `GGNN` / `GATEDGRAPH` | Custom hybrid heads | Partial / yes |

Also available in **GraphGym** fork as `gnn.stage_type: gated_hybrid`.

### `GATEDGCN` semantics (old vs new)

**Commit `2f8ad6b` (2026-06)** changed what the config string `GATEDGCN` means in hybrid MP heads.

| | **Before `2f8ad6b` (old)** | **From `2f8ad6b` (current)** |
|--|------------------------------|------------------------------|
| Config string | `GATEDGCN` | `GATEDGCN` |
| Actual layer | PyG `ResGatedGraphConv` | GNN+ `GatedGCNLayer` (GatedGCN+) |
| Edge features | Ignored (`edge_dim=None`) | Used (projected to `d_h`) |
| FFN + residual | No | Yes (inside the MP head) |
| Fair vs `layer_type: gatedgcn` baseline | **No** | **Yes** |

The old behavior is kept under **`RESGATEDGCN`** for ablations only. Do **not** compare W&B runs labeled `GATEDGCN` across this boundary without noting the code version.

**Fair repro sweeps (baseline vs baseline + 1 attn):**

| Dataset | Baseline | Hybrid MP type | Introduced |
|---------|----------|----------------|------------|
| CIFAR10 | `configs/gatedgcn/cifar10.yaml` | `GATEDGCN` (post-`2f8ad6b`) | replaces unfair sweep `5q8upl19` |
| MNIST | `configs/gatedgcn/mnist.yaml` | `GATEDGCN` (post-`2f8ad6b`) | seed **1**; sweep `mnist_repro_baseline_vs_attn_sweep.yaml` |
| peptides-func | `configs/gcn/peptides-func.yaml` | `GCNE` (commit `8a1a0cb`) | replaces unfair sweep `hymdtc7m` |

For CIFAR hybrid configs using `GATEDGCN`, set **`gnn.ffn: false`** so FFN lives only inside `GatedGCNLayer` (same as the custom_gnn baseline stack).
