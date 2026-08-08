# SiGMA parameter-budget campaign (baby / tiny)

Fill Table `tab:sigma_budget` for cells where **main Table III/IV SiGMA exceeds** the budget.
Shrink recipe: keep ≤1 attn head; keep ≤1–2 MP heads (prefer GATEDGCN / GIN / GCNE); shrink `H` / `d_h` / `L` if still over.

Entity/project: `weber-geoml-harvard-university/GNNPlus`

## Skip (do not re-run)

| Dataset | Why |
|---------|-----|
| ZINC | Main `fotdo14c` = **450,281** ≤500k |
| MNIST @1M/2M | Main = **965,002** |
| PATTERN @2M | Main = **1,988,179** |
| CLUSTER @2M | Main = **1,025,902** |
| Pep-func all | Main 1.18M fits 2M; ≤500k/1M reuse `zc371e1n` a0g1 (**450,915**, n=5) |
| Pep-struct @1M/2M | Reuse `rholn782` (**736,407**, n=5, MAE 0.2491±0.0012) |
| COCO @1M/2M | Main = **897,589** |
| MalNet @1M/2M | Main = **548,843** |

## New launches (this campaign)

| Family | Budget | Arch shrink | Config | W&B group |
|--------|--------|-------------|--------|-----------|
| mnist_b500k | ≤500k | a2g2→a1g1, H48 dh32 | `budget/mnist-b500k-a1g1.yaml` | `paper_budget_mnist_b500k` |
| cifar10_b500k | ≤500k | a1g1 **H66 dh52** (was H35/dh64→541k) | `budget/cifar10-b500k-a1g1.yaml` | `paper_budget_cifar10_b500k_fit` |
| cifar10_b1m | ≤1M | a1g1 **H86 dh76** (was H48/dh96→1.18M) | `budget/cifar10-b1m-a1g1.yaml` | `paper_budget_cifar10_b1m_fit` |
| cifar10_b2m | ≤2M | a1g2 **H82 dh84** (was H56/dh96→2.24M) | `budget/cifar10-b2m-a1g2.yaml` | `paper_budget_cifar10_b2m_fit` |
| pattern_b500k | ≤500k | a2g2→a1g1 GRIT | `budget/pattern-b500k-a1g1-grit.yaml` | `paper_budget_pattern_b500k` |
| pattern_b1m | ≤1M | a1g1 GRIT H64 | `budget/pattern-b1m-a1g1-grit.yaml` | `paper_budget_pattern_b1m` |
| cluster_b500k | ≤500k | a1g1 H40 dh32 | `budget/cluster-b500k-a1g1.yaml` | `paper_budget_cluster_b500k` |
| cluster_b1m | ≤1M | a1g1 H48 dh48 | `budget/cluster-b1m-a1g1.yaml` | `paper_budget_cluster_b1m` |
| peptides_struct_b500k | ≤500k | a1g1 H64 dh64 | `budget/peptides-struct-b500k-a1g1.yaml` | `paper_budget_peptides_struct_b500k` |
| voc_b500k | ≤500k | a2g2→a1g1 | `budget/voc-b500k-a1g1.yaml` | `paper_budget_voc_b500k` |
| voc_b1m | ≤1M | a1g1 | `budget/voc-b1m-a1g1.yaml` | `paper_budget_voc_b1m` |
| voc_b2m | ≤2M | a1g1 | `budget/voc-b2m-a1g1.yaml` | `paper_budget_voc_b2m` |
| coco_b500k | ≤500k | a1g1 H36 dh36 | `budget/coco-b500k-a1g1.yaml` | `paper_budget_coco_b500k` |
| malnet_b500k | ≤500k | a1g1 H96 dh56 | `budget/malnet-b500k-a1g1.yaml` | `paper_budget_malnet_b500k` |

**14 × 5 seeds = 70 jobs**, parallel **20**.

### Cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# if configs missing:
# python scripts/generate_sigma_budget_configs.py

bash bash_interface/cluster/submit_sigma_budget.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37600400`** |
| **Tasks** | `1-70%20` |
| **Scripts** | `submit_sigma_budget.sh` / `run_sigma_budget.sh` |
| **Generator** | `scripts/generate_sigma_budget_configs.py` |
| **Logs** | `logs_gnnplus/sigma_budget_37600400_<TASK>.log` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_budget/<fam>_seed<s>/` |

### After runs

1. Confirm W&B `params` ≤ budget for each group (abort/shrink if over).
2. Aggregate mean±std `best_test_perf` (n=5) into `tab:sigma_budget`.
3. ZINC / MalNet params for the params table: **450,281** (`fotdo14c`) and **548,843** (`figmqani`).

### Notes

- First epoch logs `params` to W&B — if a family lands over budget, shrink `dim_inner`/`d_h` and re-array that block only.
- CIFAR baby still uses `max_epoch=400` (slow); VOC/COCO use 128GB mem.

### CIFAR overshoot fix (params fit)

First CIFAR budget launch (`37600400` / groups `paper_budget_cifar10_b*`) finished but **overshot**:
541k / 1.18M / 2.24M. Recounted L=10 GATEDGCN a1g1/a1g2 widths and relaunch with
new W&B groups `*_fit`.

| Budget | Arch | H / d_h | Local param count |
|--------|------|---------|-------------------|
| ≤500k | a1g1 | 66 / 52 | ~498 774 |
| ≤1M | a1g1 | 86 / 76 | ~998 854 |
| ≤2M | a1g2 | 82 / 84 | ~1 997 748 |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_cifar_budget_fit.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37727415`** (`1-15%15`) |
| **Tasks** | `1-15%15` · 3 × 5 seeds |
| **Scripts** | `submit_cifar_budget_fit.sh` / `run_cifar_budget_fit.sh` |
| **W&B** | `paper_budget_cifar10_b{500k,1m,2m}_fit` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_budget/cifar10_b*_fit_seed<s>/` |
| **Logs** | `logs_gnnplus/cifar_budget_fit_37727415_<TASK>.log` |

---

## ∼100k budget row (colleague table)

New table budgets: **∼100k / ∼500k / ∼1M**. Existing babies mostly cover ∼500k / ∼1M;
the **∼100k** row needs dedicated shrinks (all recounted ≤100k, a1g1).

| Family | Arch | H / d_h / L | Local params | W&B group |
|--------|------|-------------|--------------|-----------|
| ZINC | a1g1 GINE | 38 / 12 / 10 | 99 915 | `paper_budget_zinc_b100k` |
| MNIST | a1g1 GATEDGCN | 28 / 26 / 6 | 99 966 | `paper_budget_mnist_b100k` |
| PATTERN | a1g1 GCNE+GRIT | 48 / 20 / 4 | 99 969 | `paper_budget_pattern_b100k` |
| CLUSTER | a1g1 GATEDGCN | 56 / 16 / 10 | 99 994 | `paper_budget_cluster_b100k` |
| Pep-func | a1g1 GINE | 36 / 36 / 6 | 99 998 | `paper_budget_peptides_func_b100k` |
| Pep-struct | a1g1 GINE | 44 / 44 / 4 | 99 887 | `paper_budget_peptides_struct_b100k` |
| VOC | a1g1 GATEDGCN | 40 / 8 / 10 | 99 929 | `paper_budget_voc_b100k` |

**7 × 5 seeds = 35 jobs.**

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_sigma_budget_100k.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *paste JOBID after submit* |
| **Tasks** | `1-35%20` |
| **Scripts** | `submit_sigma_budget_100k.sh` / `run_sigma_budget_100k.sh` |
| **Configs** | `configs/gated_hybrid/budget/*-b100k-*.yaml` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_budget/<fam>_b100k_seed<s>/` |

Note: the MNIST `98.54±0.15` currently pasted into the ∼100k LaTeX row is from the **∼201k** baby (`paper_budget_mnist_b500k`) — replace after this campaign finishes.
