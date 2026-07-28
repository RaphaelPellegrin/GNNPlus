# SiGMA paper Table 7 — MNIST + CIFAR10 + PATTERN (homog / hetero MP)

Multi-MP baselines (like VOC): keep attention/MP head counts from the paper-best
SiGMA anchors; ablate **homogeneous vs heterogeneous MP types ± gating**.

Code W&B prefix: `paper_T6_*` (same as LRGB/VOC Table 7).  
Architectural Table 6 for these datasets: [`Paper_ablations_mnist_cifar.md`](Paper_ablations_mnist_cifar.md).

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Anchors: [`Paper_final_runs.md`](Paper_final_runs.md) · [`Paper_ablations_mnist_cifar.md`](Paper_ablations_mnist_cifar.md)

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑🛑🛑  TO RUN  ·  not submitted yet  🛑🛑🛑                          ║
║  🧪  3 ds × 5 variants × 5 seeds = 75 jobs                               ║
║  🚀  bash bash_interface/cluster/submit_paper_table6_mnist_cifar_pattern.sh ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🛑 **TO RUN** |
| **SLURM array** | 🛑 *not submitted yet* |
| **Job name** | `sigma_T6_mc` |
| **Tasks** | `1-75%10` |
| **W&B** | `paper_T6_{mnist,cifar10,pattern}_{SiGMA,Homog_MP,Hetero_MP,Homog_MP_ungated,Hetero_MP_ungated}` |

---

## 1. Frozen best baselines

| Dataset | Paper Acc (%) | Anchor | Exemplar | Homog MP | Hetero MP |
|---------|---------------|--------|----------|----------|-----------|
| MNIST | 98.628 ± 0.105 | `mnist-hybrid-lcvbyyss-a2g2-anchor.yaml` | [`uh7nxm4e`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/uh7nxm4e) | `GATEDGCN,GATEDGCN` | `GATEDGCN,GCN` |
| CIFAR10 | 79.528 ± 0.180 | `cifar10-hybrid-ulij45a2-anchor.yaml` | [`3tx560wq`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3tx560wq) | `GATEDGCN`×4 | `GATEDGCN,GCN,GATEDGCN,GCN` |
| PATTERN | 86.991 ± 0.039 | `pattern-gcne-best-hybrid.yaml` | [`ta9qtxb9`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) | `GCNE,GCNE` | `GCNE,GINE` |

Attention head counts stay as in the anchor. Only `gnn_types` / `gate` change (and explicit `num_gnn_heads` for clarity).

---

## 2. Variants

| Variant (W&B) | Meaning |
|---------------|---------|
| **`SiGMA`** | Best gated hybrid as-is |
| **`Homog_MP`** | Homogeneous MP types, gated (= SiGMA arch) |
| **`Hetero_MP`** | Heterogeneous MP types, gated |
| **`Homog_MP_ungated`** | Homogeneous MP, `gate=none` |
| **`Hetero_MP_ungated`** | Heterogeneous MP, `gate=none` |

This follows the **VOC Table 7** recipe (already multi-MP), not the LRGB **+1 MP head** recipe used for peptides/COCO a1g1.

---

## 3. Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# once if needed:
# bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10

bash bash_interface/cluster/submit_paper_table6_mnist_cifar_pattern.sh
```

Optional: `PAPER_T6_MC_PARALLEL=N` (default **10**).

| Field | Value |
|-------|-------|
| **Scripts** | `submit_paper_table6_mnist_cifar_pattern.sh` → `run_paper_table6_mnist_cifar_pattern.sh` |
| **Logs** | `logs_gnnplus/sigma_T6_mc_<JOBID>_<TASK>.log` |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## 4. Aggregate

```bash
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6mc
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6mc --detail
```

### Fill-in

| Variant | MNIST ↑ | CIFAR10 ↑ | PATTERN ↑ | n |
|---------|---------|-----------|-----------|---|
| SiGMA | | | | 5 |
| Homog_MP | | | | 5 |
| Hetero_MP | | | | 5 |
| Homog_MP_ungated | | | | 5 |
| Hetero_MP_ungated | | | | 5 |
