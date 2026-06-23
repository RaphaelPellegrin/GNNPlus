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

**Fair repro sweeps** (grid, baseline vs +1 attn; use `sweep_wrapper_gnnplus_repro.sh`):

| Dataset | YAML | W&B name | Notes |
|---------|------|----------|-------|
| CIFAR10 | `cifar10_repro_baseline_vs_attn_sweep.yaml` | `GNNplus_repro-cifar10-gatedgcn-baseline-vs-attn1` | e.g. `o7tsb3k1` |
| MNIST | `mnist_repro_baseline_vs_attn_sweep.yaml` | `GNNplus_repro-mnist-gatedgcn-baseline-vs-attn1` | seed 1 |

**CIFAR10 best-hybrid** (Bayes; `cifar10_best_hybrid_sweep.yaml` → `GNNplus_best_hybrid-cifar10`):

- Base `cifar10-gatedgcn-best-hybrid.yaml` (fair `GATEDGCN` MP, 400 ep)
- Sweeps `hybrid_num_attn_heads` / `hybrid_num_gnn_heads` in `{1,2,4}`, `hybrid_d_h`, `hybrid_layers_mp`, MP presets, mask/gate/norm, dropout, LR

**MNIST / CIFAR10 + GatedGCN MP experts** (pairs GatedGCN with GCN/GIN/SAGE/GAT):

| Dataset | YAML | W&B sweep name |
|---------|------|----------------|
| MNIST | `mnist_hybrid_gatedgcn_mp_sweep.yaml` | `GNNplus_hybriddgatedGNN-mnist-gatedgcn-mp` |
| MNIST LR follow-up | `mnist_hybrid_gatedgcn_mp_lr_sweep.yaml` | `GNNplus_hybriddgatedGNN-mnist-gatedgcn-mp-lr` (18-grid: attn 4/8, d_h, lr) |
| CIFAR10 | `cifar10_hybrid_gatedgcn_mp_sweep.yaml` | `GNNplus_hybriddgatedGNN-cifar10-gatedgcn-mp` |

- `hybrid_num_gnn_heads`: **2, 4 only** (presets match 2- or 4-head lists)
- Presets: `GATEDGCN,GCN`, `GATEDGCN,GIN`, `GATEDGCN,SAGE`, `GATEDGCN,GAT`, `GATEDGCN,GATEDGCN`, and 4-head alternates

```bash
bash bash_interface/sweeps/create_sweep.sh \
  bash_interface/sweeps/cifar10_hybrid_gatedgcn_mp_sweep.yaml

SWEEP_ARRAY_TASKS=8 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=4 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    cifar10 weber-geoml-harvard-university/GNNPlus/<SWEEP_ID>
```

**Peptides-func + GCN MP experts** (every preset includes GCN; molecular):

| YAML | W&B sweep name |
|------|----------------|
| `peptides_func_hybrid_gcn_mp_sweep.yaml` | `GNNplus_hybriddgatedGNN-peptides_func-gcn-mp` |

- `--molecular=true`, metric **`test/ap`** (maximize), `hybrid_num_gnn_heads`: **2, 4**
- Presets: `GCN,GCN`, `GCN,GINE`, `GCN,GIN`, `GCN,SAGE`, `GCN,GAT`, `GCN,GATEDGCN`, … and 4-head mixes

```bash
bash bash_interface/sweeps/create_sweep.sh \
  bash_interface/sweeps/peptides_func_hybrid_gcn_mp_sweep.yaml

SWEEP_ARRAY_TASKS=8 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=4 \
  SWEEP_SLURM_TIME=96:00:00 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    peptides_func weber-geoml-harvard-university/GNNPlus/<SWEEP_ID>
```

**Finding runs in W&B:** sweep trials often show as **stopped** or **crashed** when Hyperband prunes them (e.g. at epoch 15) — they still appear under the sweep. Filter **Runs** by created date, or open the sweep page directly. SLURM logs: `logs_gnnplus/sweep_agent_<JOBID>_<TASK>.log` (not `gnnplus_sweep_mnist_*`).

**Gate metrics** (`gates/layer0/attn_0_gate_mean`, …) — **confirmed working** on cluster (2026-06, `harvard_cluster` ≥ `644ff9b`):

- Every epoch for `hybrid_gnn` when `log_gate_stats: true` (headwise and elementwise)
- Sweeps force `gnn.hybrid.log_gate_stats True` in `sweep_wrapper_gnnplus.sh`
- `GraphGymModule` unwrapped via `.model` before `collect_gate_stats`
- Dedicated `run.log(gates/...)` + `run.summary.update` each epoch
- W&B: Charts → search `gates/`; Summary → `gates/_num_metrics` > 0
- Log sanity check: `grep 'Hybrid gate stats: logging' logs_gnnplus/sweep_agent_<JOBID>_1.log`

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

**Specialized sweeps (recreate + launch):** `create_sweep.sh` on the `*_gatedgcn_mp_*` / `*_gcn_mp_*` yaml, then copy the printed `sbatch` block (not the `wandb:` log lines). Example ids (replace after recreate):

| Dataset | W&B name | Example id |
|---------|----------|------------|
| MNIST GatedGCN-mp | `GNNplus_hybriddgatedGNN-mnist-gatedgcn-mp` | `nkwgduxb` |
| CIFAR GatedGCN-mp | `GNNplus_hybriddgatedGNN-cifar10-gatedgcn-mp` | `yt923k6q` |
| Peptides-func GCN-mp | `GNNplus_hybriddgatedGNN-peptides_func-gcn-mp` | `hrfmtir9` |

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
├── mnist_hybrid_gatedgcn_mp_sweep.yaml
├── cifar10_hybrid_gatedgcn_mp_sweep.yaml
├── peptides_func_hybrid_gcn_mp_sweep.yaml
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

## Note on `GATEDGCN` (hybrid MP head semantics)

This repo sweeps **`hybrid_gnn`**, not the GNN+ paper baseline **`gatedgcn`** (`custom_gnn` + `layer_type: gatedgcn`).

**Important:** the string `GATEDGCN` in `hybrid_gnn_types` / `gnn.hybrid.gnn_types` **changed meaning** at commit **`2f8ad6b`**:

| | Pre-`2f8ad6b` | Post-`2f8ad6b` |
|--|---------------|----------------|
| `GATEDGCN` | `ResGatedGraphConv`, no edges | **`GatedGCNLayer`** (edge-aware GatedGCN+) |
| Old behavior today | — | use **`RESGATEDGCN`** explicitly |

When comparing W&B sweeps, check git SHA or sweep creation date. Pre-fix CIFAR repro sweep **`5q8upl19`** used the old (unfair) mapping. Recreate from `cifar10_repro_baseline_vs_attn_sweep.yaml` after pulling `2f8ad6b`.

Full table: [`configs/gated_hybrid/README.md`](../../configs/gated_hybrid/README.md#gatedgcn-semantics-old-vs-new).
