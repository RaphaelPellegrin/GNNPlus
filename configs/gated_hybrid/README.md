# Gated hybrid GNN (`model.type: hybrid_gnn`)

Attention + message-passing heads with sigmoid gating (ported from Heterogeneity_Profile).

## Architecture (cluster default)

- **2 attention heads** + **2 MP heads** (`num_attn_heads: 2`, `num_gnn_heads: 2`)
- **MP types** (GNN+ layer stack): `GCN,GIN` on superpixels; `GCN,GINE` on peptides
- **Outer hyperparams** (depth, width, optim, encoders): copied from `configs/gcn/<dataset>.yaml` (GNN+ paper gcne)
- **W&B**: project `MOE_6`, tags `gnnplus`, `hybrid_gnn`, `cluster`

## Local run

```bash
python main.py --cfg configs/gated_hybrid/mnist.yaml --repeat 2 seed 0 \
  wandb.use True wandb.project MOE_6
```

## Cluster submit

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export ENV_NAME=gnnplus
conda deactivate 2>/dev/null || true
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Smoke (~5 epochs)
sbatch bash_interface/cluster/smoke_test_hybrid_mnist.sh

# Full paper datasets (parallel arrays)
bash bash_interface/cluster/submit_hybrid_suite.sh
bash bash_interface/cluster/submit_hybrid_suite.sh mnist cifar10   # subset
```

| Dataset | Config | Seeds | MP heads | Source gcn yaml |
|---------|--------|-------|----------|-----------------|
| MNIST | `mnist.yaml` | 2 | GCN, GIN | `configs/gcn/mnist.yaml` |
| CIFAR10 | `cifar10.yaml` | 2 | GCN, GIN | `configs/gcn/cifar10.yaml` |
| peptides-func | `peptides-func.yaml` | 4 | GCN, GINE | `configs/gcn/peptides-func.yaml` |
| COCO-SP | `coco.yaml` | 2 | GCN, GIN | `configs/gcn/coco.yaml` (128GB) |
| Pascal VOC | `voc.yaml` | 2 | GCN, GIN | `configs/gcn/voc.yaml` |

MP head types in code: `GCN`, `GIN`, `GINE`, `GGNN`, `GATEDGCN`, `SAGE`, `GAT` (`GNNPlus/layer/gated_hybrid_layer.py`).

Also available in **GraphGym** fork as `gnn.stage_type: gated_hybrid`.
