# SiGMA Table 6 ablations — MNIST + CIFAR10 + PATTERN

Same five architectural variants as LRGB Table 6 (`paper_T5_*` groups), on Dwivedi benchmarks.

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Paper baselines: [`Paper_final_runs.md`](Paper_final_runs.md)  
Master tracker: [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑🛑🛑  TO RUN  ·  not submitted yet  🛑🛑🛑                          ║
║  🧪  3 ds × 5 variants × 5 seeds = 75 jobs                               ║
║  🚀  bash bash_interface/cluster/submit_paper_table5_mnist_cifar_ablations.sh ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🛑 **TO RUN** |
| **SLURM array** | 🛑 *not submitted yet* — paste JOBID here after launch |
| **Job name** | `sigma_T5_mc` |
| **Tasks** | `1-75%10` |
| **W&B** | `paper_T5_{mnist,cifar10,pattern}_{SiGMA,SiGMA_ungated,SiGMA_attn_gate,Attn_only,MP_only}` |

---

## 1. Frozen best baselines

| Dataset | Paper Acc (%) | Anchor config | Exemplar run | Arch |
|---------|---------------|---------------|--------------|------|
| MNIST | 98.628 ± 0.105 (n=5) | `configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml` | [`uh7nxm4e`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/uh7nxm4e) seed0 | a2g2 GATEDGCN×2, elementwise, L6/H60/d_h64, lr=5e-4, ep=200 |
| CIFAR10 | 79.528 ± 0.180 (n=5) | `configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml` | [`3tx560wq`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3tx560wq) seed1 | a8g4 GATEDGCN×4, headwise, L10/H35/d_h256, lr≈4.66e-4, ep=400 |
| PATTERN | 86.991 ± 0.039 (n=5) | `configs/gated_hybrid/pattern-gcne-best-hybrid.yaml` | [`ta9qtxb9`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) seed0 | a2g2 GCNE×2, elementwise, L12/H90/d_h90, RWSE, ep=200 |

---

## 2. Variants (same names as LRGB Table 6 / `paper_T5_*`)

| Variant | Meaning | Override |
|---------|---------|----------|
| **`SiGMA`** | Best gated hybrid | none |
| **`SiGMA_ungated`** | Same heads, no gating | `gnn.hybrid.gate none` |
| **`SiGMA_attn_gate`** | Attention gated; MP ungated | `gnn.hybrid.mp_gate none` |
| **`Attn_only`** | Drop MP → attention | `num_attn=Na+Ng`, `num_gnn=0` |
| **`MP_only`** | Drop attn → same MP type | `num_attn=0`, `num_gnn=Na+Ng` |

W&B groups: `paper_T5_mnist_*` / `paper_T5_cifar10_*` / `paper_T5_pattern_*`  
Tags: `paper_table5`, `paper_table6`, `<Variant>`, `mnist|cifar10|pattern`, `seed<k>`

---

## 3. Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# once if datasets not prepped:
# bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10
# PATTERN downloads via PyG into GNNPLUS_DATASET_DIR on first run

bash bash_interface/cluster/submit_paper_table5_mnist_cifar_ablations.sh
```

Optional: `PAPER_T5_MC_PARALLEL=N` (default **10**).

| Field | Value |
|-------|-------|
| **Status** | 🛑 **TO RUN** |
| **SLURM array** | 🛑 *not submitted yet* |
| **Job name** | `sigma_T5_mc` |
| **Tasks** | `1-75%10` = 3×5×5 |
| **Scripts** | `submit_paper_table5_mnist_cifar_ablations.sh` → `run_paper_table5_mnist_cifar_ablations.sh` |
| **Logs** | `logs_gnnplus/sigma_T5_mc_<JOBID>_<TASK>.log` |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## 4. Aggregate

```bash
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc --detail
```

### Fill-in results

| Model | MNIST ↑ | CIFAR10 ↑ | PATTERN ↑ | n |
|-------|---------|-----------|-----------|---|
| MP_only | | | | 5 |
| Attn_only | | | | 5 |
| SiGMA_ungated | | | | 5 |
| SiGMA_attn_gate | | | | 5 |
| **SiGMA** | | | | 5 |

Paper Acc (%) = fraction × 100 (PATTERN uses `accuracy-SBM` / `best_test_perf`).
