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

**COCO gap relaunch** (inode-quota recovery, 2026-07-21, `mweber_gpu` 192h):

| JOBID | Tasks | What |
|-------|-------|------|
| **`34070241`** | `71-75%3` | COCO Attn_only × 5 |
| **`34070242`** | `78-80%3` | COCO MP_only seeds 2–4 — ❌ all FAILED epoch0 `OSError 122` holylabs quota |
| **`34070243`** | `67%1` | COCO SiGMA_ungated seed 1 — ❌ same `OSError 122` |
| **Relaunch** | ✅ **`34081524`** · `67,78-80%3` · 192h (2026-07-22; `GNNPLUS_OUT_DIR` if exported) |

(Prior H200 Attn attempt `33813232` failed; superseded by `34070241`.)

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
| **SLURM** | **`33810534`** (Homog_MP ✅; ungated failed) |
| **Relaunch ungated** | ✅ **`34070244`** · `6-10%3` · 192h — task **7** ❌ Errno 122; resubmit `ARRAY=7` + `GNNPLUS_OUT_DIR` |
| **When** | 2026-07-21 |
| **Docs** | [`Paper_table6_voc.md`](Paper_table6_voc.md) |
| **W&B** | `paper_T6_voc_{Homog_MP,Homog_MP_ungated}` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_homog_34070244_<TASK>.log` |

---

### 🧪 Table 6 — 1-MP LRGB (75 jobs)

| | |
|--|--|
| **SLURM** | **`32717625`** (peptides ✅; COCO failed/empty) |
| **COCO relaunch** | ✅ **`34070245`** · `51-75%3` · 192h (2026-07-21) |
| **When** | 2026-07-18 / relaunch 2026-07-21 |
| **Docs** | [`Paper_table6_lrgb_1mp.md`](Paper_table6_lrgb_1mp.md) |
| **W&B** | `paper_T6_{peptides_func,peptides_struct,coco}_*` |
| **Logs** | `logs_gnnplus/sigma_T6_1mp_34070245_<TASK>.log` |

```bash
# 📊 aggregate Table 6 (VOC + 1-MP)
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6
```

---

### 🧪 SiGMA + GRIT attention — PATTERN + CLUSTER

| | seeds 0–4 (no VN) | seeds 5–9 ± VN=4 |
|--|--|--|
| **SLURM** | ✅ **`33458567`** | 🛑 **TO RUN** |
| **When** | 2026-07-20 | — |
| **Tasks** | `1-10%5` | `1-20%5` |
| **Docs** | [`Paper_sigma_grit_attn.md`](Paper_sigma_grit_attn.md) | same |
| **W&B** | `paper_sigma_grit_attn_{pattern,cluster}` | + `_vn4` groups |
| **Logs** | `logs_gnnplus/sigma_grit_attn_33458567_<TASK>.log` | `logs_gnnplus/sigma_grit_attn_<JOBID>_<TASK>.log` |

```bash
# launch reseed + VN (after git pull — needs VN logger/loss fix)
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
SIGMA_GRIT_ATTN_SEED_OFFSET=5 SIGMA_GRIT_ATTN_NUM_VARIANTS=2 \
  SIGMA_GRIT_ATTN_NUM_VN=4 \
  bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh

# 📊 aggregate
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_pattern --metric best_test_perf --state finished
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_cluster --metric best_test_perf --state finished
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_pattern_vn4 --metric best_test_perf --state finished
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_cluster_vn4 --metric best_test_perf --state finished
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
| **SLURM** | **`33651466`** (0 W&B — died under inode quota) |
| **Failed** | ❌ **`34070247`** — all 10 tasks `FAILED` (~40s): `LinearEdge` + empty `times_func` on ENZYMES (0 edge feats) |
| **Fix** | `edge_encoder: False` in ogpkubk9 configs (+ hetero sigma); clear error in `linear_edge_encoder.py` |
| **Failed** | ❌ **`34076119`** plateau: `ReduceLROnPlateau` `_last_lr` ([ck2dwdc7](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ck2dwdc7)) |
| **Relaunch** | ✅ **`34081517`** · `1-10%5` · 96h (2026-07-22; scheduler + prefer `GNNPLUS_OUT_DIR`) |
| **Docs** | [`Paper_enzymes_ogpkubk9.md`](Paper_enzymes_ogpkubk9.md) |
| **W&B** | `enzymes_ogpkubk9_a4g4_plateau_seeds` / `enzymes_ogpkubk9_a4g4_cosine_seeds` |
| **Logs** | `logs_gnnplus/enz_ogpkubk9_34081517_<TASK>.log` |

---

### 🧪 Heterogeneity TU profiles (9 jobs)

| | |
|--|--|
| **SLURM** | **`33811552`** (`gpu_h200` — fake-finish under quota) |
| **Failed** | ❌ **`34073629`** — GCN/GIN `IndexError` after trial 1; SiGMA `LinearEdge` |
| **Relaunch** | 🛑 **TO RUN** after `Dataset.get` fix + `GNNPLUS_OUT_DIR` |
| **Docs** | [`Paper_heterogeneity.md`](Paper_heterogeneity.md) |
| **W&B** | `building_hetero_profile_{mutag,enzymes,proteins}` |
| **Logs** | `logs_gnnplus/hetero_tu_34073629_<TASK>.log` |

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

## Quick checklist

| Campaign | Status | JOBID |
|----------|--------|-------|
| Table 5 LRGB | ✅ | `32232124` |
| Table 5 COCO gaps relaunch | ✅ Attn `34070241`; MP/ungated ✅ `34081524` (prior ❌ `34070242`/`43`) | `34081524` |
| ENZYMES ogpkubk9 seeds | ✅ | `34081517` (priors ❌ `34076119` / `34070247`) |
| Table 5 MNIST+CIFAR | 🛑 TO RUN | — |
| Peptides UniGCN a0g2 mixes | ✅ | `33651463` |
| Peptides-func Homog→MP_only a0g3 | ✅ | `33651464` |
| SiGMA + GRIT attn (PATTERN/CLUSTER) | ✅ seeds0–4 `33458567`; 🛑 reseed+VN | — |
| Table 6 VOC (SiGMA / Hetero ± ungated) | ✅ | `32717593` |
| Table 6 VOC Homog_MP | ✅ | `33810534` |
| Table 6 VOC Homog_MP_ungated relaunch | ✅ | `34070244` |
| Table 6 1-MP peptides | ✅ | `32717625` |
| Table 6 COCO relaunch | ✅ | `34070245` |
| ENZYMES ogpkubk9 sweep | 🛑 TO RUN | — |
| Heterogeneity TU relaunch | ✅ | `34073629` |
