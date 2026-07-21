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

### 🧪 Table 6 — PascalVOC Homog_MP ± ungated (10 jobs)

| | |
|--|--|
| **SLURM** | **`33810534`** |
| **When** | 2026-07-21 |
| **Tasks** | `1-10%5` |
| **Docs** | [`Paper_table6_voc.md`](Paper_table6_voc.md) |
| **W&B** | `paper_T6_voc_{Homog_MP,Homog_MP_ungated}` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_homog_33810534_<TASK>.log` |

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

### 🧪 Peptides UniGCN a0g2 MP mixes (20 jobs)

| | |
|--|--|
| **SLURM** | **`33651463`** |
| **When** | 2026-07-20 |
| **Tasks** | `1-20%4` |
| **Docs** | [`Paper_peptides_unigcn_a0g2_mp_mixes.md`](Paper_peptides_unigcn_a0g2_mp_mixes.md) |
| **W&B** | `paper_peptides_{peptides_func,peptides_struct}_a0g2_{UNIGCN_GINE,UNIGCN_GATEDGCN}` |
| **Logs** | `logs_gnnplus/pep_unigcn_a0g2_33651463_<TASK>.log` |

---

### 🧪 Peptides-func Homog → MP_only a0g3 (5 jobs)

| | |
|--|--|
| **SLURM** | **`33651464`** |
| **When** | 2026-07-20 |
| **Tasks** | `1-5%2` |
| **Docs** | [`Paper_peptides_func_homog_a1g2_mp_only.md`](Paper_peptides_func_homog_a1g2_mp_only.md) |
| **W&B** | `paper_T5_peptides_func_HomogMP_MPonly` |
| **Logs** | `logs_gnnplus/sigma_func_a0g3_33651464_<TASK>.log` |

---

### 🧪 ENZYMES ogpkubk9 seed grids (10 jobs)

| | |
|--|--|
| **SLURM** | **`33651466`** |
| **When** | 2026-07-20 |
| **Tasks** | `1-10%2` (plateau×5 + cosine×5) |
| **Docs** | [`Paper_enzymes_ogpkubk9.md`](Paper_enzymes_ogpkubk9.md) |
| **W&B** | `enzymes_ogpkubk9_a4g4_plateau_seeds` / `enzymes_ogpkubk9_a4g4_cosine_seeds` |
| **Logs** | `logs_gnnplus/enz_ogpkubk9_33651466_<TASK>.log` |

---

## 🛑🛑🛑 TO RUN — cluster was full — COPY/PASTE WHEN SLOTS FREE 🛑🛑🛑

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
║  🛑  TO RUN  ·  Heterogeneity profiles (TU) · ≤5 GPUs                    ║
║  📈  MUTAG / ENZYMES / PROTEINS × {GCN, GIN, SiGMA} = 9 jobs            ║
║  🔁  50/25/25 · 300 ep · ≥100 test appearances per graph                ║
║  ☁️  W&B groups: building_hetero_profile_<dataset>                      ║
║  📄  Paper_heterogeneity.md                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🚀 paper protocol (≤5 GPUs on H200; long!)
HETERO_PARALLEL=5 HETERO_PARTITION=gpu_h200 \
  bash bash_interface/cluster/submit_heterogeneity_tu.sh
# if TimeLimit rejected: also set HETERO_TIME=72:00:00

# 🧪 smoke first (recommended):
# HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=20 \
#   HETERO_PARTITION=gpu_h200 bash bash_interface/cluster/submit_heterogeneity_tu.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **Partition** | `gpu_h200` (override via `HETERO_PARTITION`) |
| **Parallel** | ≤5 GPUs |
| **Local outs** | `results/heterogeneity/<dataset>_<MODEL>/` |
| **W&B** | groups `building_hetero_profile_{mutag,enzymes,proteins}`; artifact = pickle + appearances CSV + profile PNGs |

---

## Quick checklist

| Campaign | Status | JOBID |
|----------|--------|-------|
| Table 5 LRGB | ✅ | `32232124` |
| Table 5 MNIST+CIFAR | 🛑 TO RUN | — |
| Peptides UniGCN a0g2 mixes | ✅ | `33651463` |
| Peptides-func Homog→MP_only a0g3 | ✅ | `33651464` |
| SiGMA + GRIT attn (PATTERN/CLUSTER) | ✅ | `33458567` |
| Table 6 VOC (SiGMA / Hetero ± ungated) | ✅ | `32717593` |
| Table 6 VOC Homog_MP ± ungated | ✅ | `33810534` |
| Table 6 1-MP | ✅ | `32717625` |
| ENZYMES ogpkubk9 seeds | ✅ | `33651466` |
| ENZYMES ogpkubk9 sweep | 🛑 TO RUN | — |
| Heterogeneity TU profiles | 🛑 TO RUN | — |
