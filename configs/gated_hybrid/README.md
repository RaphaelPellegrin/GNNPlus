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

MP head types in code: `GCN`, `GIN`, `GINE`, `GGNN`, `GATEDGCN`, `SAGE`, `GAT`.

Also available in **GraphGym** fork as `gnn.stage_type: gated_hybrid`.
