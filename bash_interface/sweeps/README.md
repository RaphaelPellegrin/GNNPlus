# GNNPlus W&B sweeps — hybrid gated GNN (`hybrid_gnn`)

Bayesian sweeps for **GNNPlus** `model.type: hybrid_gnn`, mirroring
[Heterogeneity_Profile/bash_interface/sweeps](../../../Heterogeneity_Profile/bash_interface/sweeps/).

## Sweep name

Each dataset sweep is named **`GNNplus_hybriddgatedGNN-<dataset>`** in W&B project **`GNNPlus`**.

## Swept hyperparameters (fixed grid in Bayes search)

| Parameter | Values |
|-----------|--------|
| `hybrid_num_attn_heads` | 2, 4, 8 |
| `hybrid_num_gnn_heads` | 2, 4, 8 |
| `hybrid_d_h` | 8, 16, 32, 64 |
| `hybrid_attn_dropout` | 0.1, 0.2, 0.5 |
| `hybrid_attn_mask` | `full`, `graph_restricted` |
| `hybrid_gate` | `elementwise`, `headwise` |
| `hybrid_norm` | `layernorm`, `rmsnorm` |
| `hybrid_gnn_types` | dataset-specific MP presets (see below) |

`seed` is fixed at **0** in `sweep_wrapper_gnnplus.sh` (not swept).

**COCO / VOC (superpixel):** separate sweep template — `hybrid_num_attn_heads` **2, 4 only**; swept `batch_size` **8, 16**; yaml default `batch_size: 8`.

### `hybrid_gnn_types` pools

**General** (MNIST, CIFAR10, COCO, VOC, enzymes, mutag, mal, cluster, pattern):

- `GCN,GIN`, `GCN,GIN,SAGE,GAT`, `GINE,GIN`, **`GINE,SAGE`**, `GINE,GGNN`, `GIN,SAGE`, `SAGE,GAT`, `GCN,SAGE`, `GATEDGRAPH,GATEDGRAPH`, `GATEDGCN,GATEDGCN`

**Molecular** (peptides, hiv, zinc, ppa, pcba):

- `GCN,GINE`, `GINE,GINE`, `GINE,GGNN`, **`GINE,SAGE`**, `GGNN,GINE`, `GIN,GINE`, `GCN,GIN,SAGE,GAT`, `GCN,GIN`, `SAGE,GAT`, `GATEDGRAPH,GATEDGRAPH`, `GATEDGCN,GATEDGCN`

Python `parse_hybrid_gnn_types` pads or truncates the list to match `hybrid_num_gnn_heads`.

**Gate metrics** (`gates/layer0/attn_0_gate_mean`, …): logged every epoch for `hybrid_gnn` when `log_gate_stats: true` (elementwise and headwise). Sweeps force `gnn.hybrid.log_gate_stats True` in the wrapper. In W&B: **Charts → Add panel → search `gates/`**.

**More runs on an existing sweep** (same W&B sweep id, no new sweep):

```bash
# 24 agents × 4 runs each = up to 96 new trials; auto lookup id from sweeps.log
SWEEP_ARRAY_TASKS=24 RUNS_PER_AGENT=4 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh tier1

# Or explicit ids (from sweeps.log / .hybrid_sweep_ids):
bash bash_interface/sweeps/relaunch_sweep_agents.sh \
  mnist weber-geoml-harvard-university/GNNPlus/mhc71f9c
bash bash_interface/sweeps/relaunch_sweep_agents.sh \
  cifar10 weber-geoml-harvard-university/GNNPlus/0yksmizq
```

Pull latest code on cluster before relaunch so gate logging fixes are active.

Outer training hyperparams (depth, width, LR, encoders) stay fixed per dataset in `configs/gated_hybrid/<dataset>.yaml` (from GNN+ `configs/gcn/`).

MP head types are auto-filled from head count: **GCN,GIN** (superpixels) or **GCN,GINE** (molecular).

## Files

```
bash_interface/sweeps/
├── sweep_wrapper_gnnplus.sh       # W&B --key=val → GraphGym key val
├── _hybrid_sweep_template.yaml
├── generate_hybrid_sweep_yamls.sh
├── create_sweep.sh
├── run_wandb_sweep_agent.sh       # SLURM → wandb agent (gnnplus env)
├── launch_hybrid_sweeps.sh        # create + sbatch agents (tier1–4 default)
├── mnist_hybrid_gnnplus_sweep.yaml
├── cifar10_hybrid_gnnplus_sweep.yaml
└── ...
```

## Cluster workflow

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull origin harvard_cluster
source ~/.gnnplus_env
export WANDB_PROJECT=GNNPlus
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export ENV_NAME=gnnplus
conda deactivate 2>/dev/null || true

# 1) Generate YAMLs (if not in repo yet)
bash bash_interface/sweeps/generate_hybrid_sweep_yamls.sh

# 2a) One dataset
bash bash_interface/sweeps/create_sweep.sh \
  bash_interface/sweeps/mnist_hybrid_gnnplus_sweep.yaml
SWEEP_ID=$(cat bash_interface/sweeps/.last_sweep_id)
SWEEP_DATASET=mnist RUNS_PER_AGENT=2 \
  sbatch --array=1-16%8 --mem=64GB --time=96:00:00 \
  --export=ALL,SWEEP_ID="${SWEEP_ID}",SWEEP_DATASET=mnist,RUNS_PER_AGENT=2,ENV_NAME=gnnplus \
  bash_interface/sweeps/run_wandb_sweep_agent.sh

# 2b) Tier 1–4: create all sweeps + launch agents
bash bash_interface/sweeps/launch_hybrid_sweeps.sh tier1 tier2 tier3 tier4

# Create sweeps only (no sbatch)
bash bash_interface/sweeps/launch_hybrid_sweeps.sh --create-only all
```

**Memory:** use `--mem=128GB` for `coco`, `voc`, `cluster`, `pattern`, `pcba` (set automatically in `launch_hybrid_sweeps.sh`).

## Metrics per dataset

| Dataset | W&B metric | Goal |
|---------|------------|------|
| mnist, cifar10, enzymes, mutag, ppa | `test/accuracy` | maximize |
| coco, voc | `test/f1` | maximize |
| peptides-func, pcba | `test/ap` | maximize |
| peptides-struct, zinc | `test/mae` | minimize |
| hiv | `test/auc` | maximize |
| cluster, pattern | `test/accuracy-SBM` | maximize |

## Note on GatedGCN

This sweeps **`hybrid_gnn`** only — not the GNN+ paper baseline **`gatedgcn`** (`custom_gnn` + `layer_type: gatedgcn`).
