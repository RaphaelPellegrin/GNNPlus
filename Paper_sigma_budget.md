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
| cifar10_b500k | ≤500k | a8g4→a1g1, H35 dh64 | `budget/cifar10-b500k-a1g1.yaml` | `paper_budget_cifar10_b500k` |
| cifar10_b1m | ≤1M | a1g1 H48 dh96 | `budget/cifar10-b1m-a1g1.yaml` | `paper_budget_cifar10_b1m` |
| cifar10_b2m | ≤2M | a1g2 H56 dh96 | `budget/cifar10-b2m-a1g2.yaml` | `paper_budget_cifar10_b2m` |
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
| **SLURM** | 🛑 *paste JOBID* |
| **Tasks** | `1-70%20` |
| **Scripts** | `submit_sigma_budget.sh` / `run_sigma_budget.sh` |
| **Generator** | `scripts/generate_sigma_budget_configs.py` |
| **Logs** | `logs_gnnplus/sigma_budget_<JOBID>_<TASK>.log` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_budget/<fam>_seed<s>/` |

### After runs

1. Confirm W&B `params` ≤ budget for each group (abort/shrink if over).
2. Aggregate mean±std `best_test_perf` (n=5) into `tab:sigma_budget`.
3. ZINC / MalNet params for the params table: **450,281** (`fotdo14c`) and **548,843** (`figmqani`).

### Notes

- First epoch logs `params` to W&B — if a family lands over budget, shrink `dim_inner`/`d_h` and re-array that block only.
- CIFAR baby still uses `max_epoch=400` (slow); VOC/COCO use 128GB mem.
