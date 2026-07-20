# 🚦 Cluster launches tracker (SiGMA paper week)

> **Status legend**  
> ✅ **SUBMITTED / RUNNING** — job already on FASRC  
> 🛑 **TO RUN** — cluster full / wait — do **not** forget these  
> 📊 Aggregate locally with W&B when finished  

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)

---

## ✅ SUBMITTED (do not re-submit)

### 🧪 Table 5 — LRGB ablations (80 jobs)

| | |
|--|--|
| **SLURM** | **`32232124`** |
| **When** | 2026-07-17 |
| **Tasks** | `1-80%18` |
| **Docs** | [`Paper_ablations.md`](Paper_ablations.md) |
| **W&B** | `paper_T5_<ds>_{SiGMA,SiGMA_ungated,Attn_only,MP_only}` |
| **Logs** | `logs_gnnplus/sigma_T5_abl_32232124_<TASK>.log` |

```bash
# 📊 aggregate
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5
```

---

### 🧪 Table 6 — PascalVOC hetero MP (15 jobs)

| | |
|--|--|
| **SLURM** | **`32717593`** |
| **When** | 2026-07-18 |
| **Tasks** | `1-15%3` (parallel 3) |
| **Docs** | [`Paper_table6_voc.md`](Paper_table6_voc.md) |
| **W&B** | `paper_T6_voc_{SiGMA,Hetero_MP,Hetero_MP_ungated}` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_32717593_<TASK>.log` |

---

### 🧪 Table 6 — 1-MP LRGB (75 jobs)

| | |
|--|--|
| **SLURM** | **`32717625`** |
| **When** | 2026-07-18 |
| **Tasks** | `1-75%7` (parallel 7) |
| **Docs** | [`Paper_table6_lrgb_1mp.md`](Paper_table6_lrgb_1mp.md) |
| **W&B** | `paper_T6_{peptides_func,peptides_struct,coco}_*` |
| **Logs** | `logs_gnnplus/sigma_T6_1mp_32717625_<TASK>.log` |

```bash
# 📊 aggregate Table 6 (VOC + 1-MP)
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6
```

---

### 🧪 SiGMA + GRIT attention — PATTERN + CLUSTER (10 jobs)

| | |
|--|--|
| **SLURM** | **`33458567`** |
| **When** | 2026-07-20 |
| **Tasks** | `1-10%5` |
| **Docs** | [`Paper_sigma_grit_attn.md`](Paper_sigma_grit_attn.md) |
| **W&B** | `paper_sigma_grit_attn_pattern` / `paper_sigma_grit_attn_cluster` |
| **Tags** | `sigma_grit_attn`, `attn_type_grit`, `grit_attn` |
| **Logs** | `logs_gnnplus/sigma_grit_attn_33458567_<TASK>.log` |

```bash
# 📊 aggregate
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_pattern --metric best_test_perf --state finished
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_cluster --metric best_test_perf --state finished
```

---

## 🛑🛑🛑 TO RUN — cluster was full — COPY/PASTE WHEN SLOTS FREE 🛑🛑🛑

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  Peptides UniGCN a0g2 MP mixes (20 jobs)                 ║
║  🧪  func+struct × {UNIGCN+GINE, UNIGCN+GATEDGCN} × 5 seeds · no attn   ║
║  📄  Paper_peptides_unigcn_a0g2_mp_mixes.md                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🚀 20 jobs, ≤8 GPUs
bash bash_interface/cluster/submit_peptides_unigcn_a0g2_mp_mixes.sh
# 👉 paste JOBID into Paper_peptides_unigcn_a0g2_mp_mixes.md + here
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **W&B** | `paper_peptides_{peptides_func,peptides_struct}_a0g2_{UNIGCN_GINE,UNIGCN_GATEDGCN}` |
| **HPs** | func=Homog a1g2 / o5cdk766 · struct=g3bsaq32 |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  Peptides-func Homog_MP → MP_only (5 jobs)               ║
║  🧪  NEW best a1g2 GCN×2 → a0g3 GCN×3 gated (drop attn → GCN)           ║
║  📄  Paper_peptides_func_homog_a1g2_mp_only.md                           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🚀 5 jobs, ≤5 GPUs
bash bash_interface/cluster/submit_peptides_func_homog_a1g2_mp_only.sh
# 👉 paste JOBID into Paper_peptides_func_homog_a1g2_mp_only.md + here
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **W&B** | `paper_T5_peptides_func_HomogMP_MPonly` |
| **Aggregate** | `python scripts/api_wanndb_query/aggregate_paper_repro.py --group paper_T5_peptides_func_HomogMP_MPonly --metric best_test_perf --state finished` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  Table 5 MNIST + CIFAR10 ablations (40 jobs)             ║
║  🧪  SiGMA / ungated / Attn_only / MP_only × 5 seeds                     ║
║  📄  Paper_ablations_mnist_cifar.md                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# once if needed:
# bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10

# 🚀 40 jobs, ≤10 GPUs
bash bash_interface/cluster/submit_paper_table5_mnist_cifar_ablations.sh
# 👉 paste JOBID into Paper_ablations_mnist_cifar.md + here
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **W&B** | `paper_T5_{mnist,cifar10}_{SiGMA,SiGMA_ungated,Attn_only,MP_only}` |
| **Aggregate** | `python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  ENZYMES ogpkubk9 seed grids (10 jobs)                   ║
║  🧬  plateau×5 + cosine×5  ·  source MOE_6/ogpkubk9                     ║
║  📄  Paper_enzymes_ogpkubk9.md                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🚀 10 jobs, ≤5 GPUs
bash bash_interface/cluster/submit_enzymes_ogpkubk9_seed_grids.sh
# 👉 paste JOBID into Paper_enzymes_ogpkubk9.md + here
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **W&B** | `enzymes_ogpkubk9_a4g4_plateau_seeds` / `enzymes_ogpkubk9_a4g4_cosine_seeds` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  ENZYMES ogpkubk9 centered W&B sweep                     ║
║  🎯  vary lr · #attn/MP heads (gates) · hybrid d_h                      ║
║  📄  Paper_enzymes_ogpkubk9.md                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export WANDB_PROJECT=GNNPlus
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/sweeps/create_sweep.sh \
  bash_interface/sweeps/enzymes_ogpkubk9_centered_sweep.yaml
# 👉 then paste the printed sbatch agent block
# 👉 paste SWEEP_ID + agent JOBID into Paper_enzymes_ogpkubk9.md + here
```

| Field | Value |
|-------|-------|
| **Sweep ID** | 🛑 *not created yet* |
| **Agent job** | 🛑 *not submitted yet* |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  Heterogeneity profiles (TU)                             ║
║  📈  MUTAG / ENZYMES / PROTEINS × {GCN, GIN, SiGMA} = 9 jobs            ║
║  🔁  50/25/25 · 300 ep · ≥100 test appearances per graph                ║
║  📄  Paper_heterogeneity.md                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🚀 paper protocol (long!)
bash bash_interface/cluster/submit_heterogeneity_tu.sh

# 🧪 smoke first (recommended):
# HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=20 \
#   bash bash_interface/cluster/submit_heterogeneity_tu.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **Outputs** | `results/heterogeneity/<dataset>_<MODEL>/` |

---

## Quick checklist

| Campaign | Status | JOBID |
|----------|--------|-------|
| Table 5 LRGB | ✅ | `32232124` |
| Table 5 MNIST+CIFAR | 🛑 TO RUN | — |
| Peptides UniGCN a0g2 mixes | 🛑 TO RUN | — |
| Peptides-func Homog→MP_only a0g3 | 🛑 TO RUN | — |
| SiGMA + GRIT attn (PATTERN/CLUSTER) | ✅ | `33458567` |
| Table 6 VOC | ✅ | `32717593` |
| Table 6 1-MP | ✅ | `32717625` |
| ENZYMES ogpkubk9 seeds | 🛑 TO RUN | — |
| ENZYMES ogpkubk9 sweep | 🛑 TO RUN | — |
| Heterogeneity TU profiles | 🛑 TO RUN | — |
