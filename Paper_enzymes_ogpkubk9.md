# ENZYMES — port of Heterogeneity_Profile best HybridGated (`ogpkubk9`)

Source run: [`ogpkubk9`](https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/ogpkubk9)  
Project: `MOE_6` · best/test_acc ≈ **0.607** · a4g4 `GCN,GIN,SAGE,GAT` · plateau · L12 / H64 / dh16

---

## Frozen architecture

| Knob | Value |
|------|-------|
| Model | `hybrid_gnn` (SiGMA) |
| Heads | a4g4 — `GCN,GIN,SAGE,GAT` |
| `d_h` | 16 |
| Gate / norm / mask | headwise / layernorm / full |
| Depth / width | L=12, H=64 |
| FFN + residual | True |
| LR | 0.001 |
| WD | 0 |
| Batch | 64 |
| Split | random 50/25/25 (HP-style, not 10-fold) |

Configs:

- Plateau (source scheduler): `configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml`
- Cosine (non-plateau): `configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-cosine-anchor.yaml`

---

## 1. Seed grids (10 jobs)

5 seeds × {plateau, cosine}:

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_enzymes_ogpkubk9_seed_grids.sh
```

| Field | Value |
|-------|-------|
| **SLURM array** | *(paste JOBID)* |
| **Tasks** | `1-10%5` |
| **W&B** | `enzymes_ogpkubk9_a4g4_plateau_seeds` / `enzymes_ogpkubk9_a4g4_cosine_seeds` |

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group enzymes_ogpkubk9_a4g4_plateau_seeds --metric best_test_perf --state finished
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group enzymes_ogpkubk9_a4g4_cosine_seeds --metric best_test_perf --state finished
```

---

## 2. Centered sweep (lr × #gates × d_h)

Varies around ogpkubk9:

- `optim.base_lr` ∈ log-uniform [2e-4, 5e-3]
- `num_attn_heads` ∈ {2,4,6,8} (gated attn heads)
- `num_gnn_heads` ∈ {2,4,6,8} (gated MP heads)
- `d_h` ∈ {8,16,32,64}

YAML: `bash_interface/sweeps/enzymes_ogpkubk9_centered_sweep.yaml`

```bash
source ~/.gnnplus_env
export WANDB_PROJECT=GNNPlus
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

bash bash_interface/sweeps/create_sweep.sh \
  bash_interface/sweeps/enzymes_ogpkubk9_centered_sweep.yaml
# then run the printed sbatch agent line (or agents from create_sweep output)
```

| Field | Value |
|-------|-------|
| **Sweep ID** | *(paste)* |
| **Agent job** | *(paste)* |
