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

**COCO full H200 twin** (keep mweber jobs; `gpu_h200`, ≤25 GPUs total, **72h** MaxTime, `_h200` W&B names):

| JOBID | Tasks | What |
|-------|-------|------|
| ✅ **`34098505`** | T5 `61-80%12` | all COCO Table 5 variants × seeds |
| ✅ **`34098527`** | T6 `51-75%13` | all COCO Table 6 variants × seeds |

Submitted 2026-07-22. Logs: `logs_gnnplus/sigma_T5_abl_34098505_<TASK>.log`, `logs_gnnplus/sigma_T6_1mp_34098527_<TASK>.log`.

**COCO ep=150 insurance twin** (keep 300-ep jobs; same T5/T6 recipes; distinct W&B groups):

| JOBID | Tasks | What |
|-------|-------|------|
| ✅ **`34682558`** | T5 `61-80%5` | COCO Table 5 × seeds · `mweber_gpu` · ep=150 · `_ep150` |
| ✅ **`34682560`** | T6 `51-75%5` | COCO Table 6 × seeds · `mweber_gpu` · ep=150 · `_ep150` |

Submitted 2026-07-23. W&B: `paper_T5_ep150_coco_*` / `paper_T6_ep150_coco_*`. Logs: `logs_gnnplus/sigma_T5_abl_34682558_<TASK>.log`, `logs_gnnplus/sigma_T6_1mp_34682560_<TASK>.log`.

```bash
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
bash bash_interface/cluster/submit_coco_ep150_relaunch.sh
```

**VOC Table 5 SiGMA + SiGMA_ungated H200 twin** (tasks `41-50`, keep `32232124`):

| JOBID | Tasks | What |
|-------|-------|------|
| ✅ **`34099247`** | T5 `41-50%5` | VOC SiGMA + ungated × 5 seeds · `gpu_h200` · 72h · `_h200` |

Submitted 2026-07-22. Logs: `logs_gnnplus/sigma_T5_abl_34099247_<TASK>.log`.

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
| **Relaunch ungated** | ✅ **`34070244`** · `6-10%3` · 192h — task **7** ❌ Errno 122 |
| **Seed1 retry** | ✅ **`34409933`** · `ARRAY=7` (2026-07-22) — **must** use `GNNPLUS_OUT_DIR` or Errno 122 again |
| **When** | 2026-07-21 |
| **Docs** | [`Paper_table6_voc.md`](Paper_table6_voc.md) |
| **W&B** | `paper_T6_voc_{Homog_MP,Homog_MP_ungated}` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_homog_34070244_<TASK>.log` |

---

### 🧪 Table 6 — 1-MP LRGB (75 jobs)

| | |
|--|--|
| **SLURM** | **`32717625`** (peptides ✅; COCO failed/empty) |
| **COCO relaunch** | ✅ **`34070245`** · `51-75%3` · 192h (mweber) · H200 twin ✅ **`34098527`** |
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
| **SLURM** | ✅ **`33458567`** | ✅ **`34096507`** |
| **When** | 2026-07-20 | 2026-07-22 |
| **Tasks** | `1-10%5` | `1-20%5` |
| **Docs** | [`Paper_sigma_grit_attn.md`](Paper_sigma_grit_attn.md) | same |
| **W&B** | `paper_sigma_grit_attn_{pattern,cluster}` | + `_vn4` groups |
| **Logs** | `logs_gnnplus/sigma_grit_attn_33458567_<TASK>.log` | `logs_gnnplus/sigma_grit_attn_34096507_<TASK>.log` |

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

**CLUSTER VN×LR grid** (10 configs × 5 seeds = 50 · `mweber_gpu` ≤5 GPUs):

| | |
|--|--|
| **SLURM** | ✅ **`34416930`** (submitted 2026-07-22) |
| **Tasks** | `1-50%5` |
| **Docs** | [`Paper_sigma_grit_attn.md`](Paper_sigma_grit_attn.md) |
| **W&B** | `paper_sigma_grit_cluster_<novn\|vnK>_lr<tag>` |
| **Logs** | `logs_gnnplus/sigma_grit_vn_lr_34416930_<TASK>.log` |

---

### 🧪 Peptides-func SiGMA (o5cdk766) VN×LR grid

| | |
|--|--|
| **SLURM** | ✅ **`34427481`** (submitted 2026-07-23) |
| **Tasks** | `1-50%5` · `mweber_gpu` · 192h |
| **Docs** | [`Paper_peptides_func_vn.md`](Paper_peptides_func_vn.md) |
| **W&B** | `paper_sigma_peptides_func_<novn\|vnK>_lr<tag>_<pyr\|nopyr>` |
| **Anchor** | `peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml` (a1g1 GCN, lr≈2.083e-4, ep=900) |
| **Logs** | `logs_gnnplus/pep_func_vn_lr_34427481_<TASK>.log` |

```bash
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
bash bash_interface/cluster/submit_peptides_func_o5cdk766_vn_lr_grid.sh
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
| **Failed** | ❌ **`34073629`** — GCN/GIN `IndexError` after trial 1 (`dataset[idx]` on split-local view) |
| **Relaunch** | ✅ **`34410913`** · `gpu_h200` · `%3` · 72h (2026-07-22) — ❌ **binary acc bug** (MUTAG/PROTEINS junk; ENZYMES OK). Fix in `run_heterogeneity_profiles.py` — **re-run MUTAG+PROTEINS** after pull |
| **proteins_sigma** | ✅ **`34869869`** task `9` · `mweber_gpu` · 192h (2026-07-24) — prior `fls85zer` crashed @ ~85/100 apps |
| **Docs** | [`Paper_heterogeneity.md`](Paper_heterogeneity.md) |
| **W&B** | `building_hetero_profile_{mutag,enzymes,proteins}` |
| **Logs** | `logs_gnnplus/hetero_tu_34410913_<TASK>.log` · `hetero_tu_34869869_9.log` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/<ds>_<model>/` |
| **Priors** | ❌ `34409940` / `34073629` / `34070246` (94 graphs → IndexError) |

---

### 🧪 COCO Table 6 Attn a3 + MP a0g3 @ ep150 (10 jobs)

| | |
|--|--|
| **SLURM** | ✅ **`34869787`** (2026-07-24) |
| **Tasks** | `1-10%5` (1–5 Attn a3g0; 6–10 MP a0g3 GATEDGCN×3) |
| **Epochs** | **150** (`_ep150`) |
| **Docs** | [`Paper_ablations.md`](Paper_ablations.md) |
| **W&B** | `paper_T5_coco_Attn_only_a3` · `paper_T5_coco_MP_only_a0g3` |
| **Logs** | `logs_gnnplus/sigma_T5_coco_a3_34869787_<TASK>.log` |

---

### 🧪 TU all-layer activations (3 jobs)

| | |
|--|--|
| **SLURM** | ✅ **`34869795`** (2026-07-24) |
| **Tasks** | `1-3%3` (MUTAG / ENZYMES ogpkubk9 / PROTEINS · seed 0) |
| **Docs** | [`Paper_last_layer_activations.md`](Paper_last_layer_activations.md) |
| **W&B** | `layer_act_{mutag,enzymes,proteins}` |
| **Logs** | `logs_gnnplus/tu_last_act_34869795_<TASK>.log` |
| **Outs** | `$GNNPLUS_OUT_DIR/activations/<ds>_<tag>_seed0/{mid,last,best}/` |

---

## 🛑🛑🛑 TO RUN — cluster was full — COPY/PASTE WHEN SLOTS FREE 🛑🛑🛑

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  CLUSTER push-to-80% SiGMA+GRIT Bayes sweep               ║
║  🎯  lr · VN · d_h · mp_dropout · weight_decay (+ prior_runs)            ║
║  📄  Paper_cluster_80_sweep.md                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export WANDB_PROJECT=GNNPlus
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

PRIOR_RUNS="tu8cr0fp 63inaz23 hx3v1ybs er5vpx7j 80ngb67n 3j7zj86o hmy6di2u q6bi1pqc q6b3ofpj opqkgsxi nhuyof1w f6k8rjip" \
  bash bash_interface/sweeps/create_sweep.sh \
    bash_interface/sweeps/cluster_sigma_grit_vn_lr_dh_sweep.yaml
# ✅ Sweep h02m95qg created 2026-07-23 — agents still need sbatch (below)
```

| Field | Value |
|-------|-------|
| **Sweep** | ✅ [`h02m95qg`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/h02m95qg) |
| **Agent job** | 🛑 *paste after sbatch* |

```bash
# launch agents (≤4 GPUs)
SWEEP_ID=weber-geoml-harvard-university/GNNPlus/h02m95qg SWEEP_DATASET=cluster RUNS_PER_AGENT=3 \
sbatch --job-name=cluster_push80_cluster --array=1-16%4 --mem=128GB --time=120:00:00 \
  --export=ALL,SWEEP_ID=weber-geoml-harvard-university/GNNPlus/h02m95qg,SWEEP_DATASET=cluster,RUNS_PER_AGENT=3,WANDB_PROJECT=GNNPlus,ENV_NAME=gnnplus,GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-},GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR:-} \
  bash_interface/sweeps/run_wandb_sweep_agent.sh
```

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
| Table 5+6 COCO H200 twin | ✅ T5 `34098505` · T6 `34098527` | `34098505` / `34098527` |
| Table 5+6 COCO ep150 twin | ✅ T5 `34682558` · T6 `34682560` | `34682558` / `34682560` |
| COCO Table6 Attn a3 + MP a0g3 | ✅ | `34869787` |
| CLUSTER push-to-80% sweep | 🛑 TO RUN | — |
| Table 5 VOC SiGMA+ungated H200 | ✅ `34099247` (`41-50%5`) | `34099247` |
| ENZYMES ogpkubk9 seeds | ✅ | `34081517` (priors ❌ `34076119` / `34070247`) |
| Table 5 MNIST+CIFAR | 🛑 TO RUN | — |
| Peptides UniGCN a0g2 mixes | ✅ | `33651463` |
| Peptides-func Homog→MP_only a0g3 | ✅ | `33651464` |
| SiGMA + GRIT attn (PATTERN/CLUSTER) | ✅ seeds0–4 `33458567`; ✅ reseed+VN `34096507` | `34096507` |
| SiGMA+GRIT CLUSTER VN×LR grid | ✅ `34416930` (50 jobs) | `34416930` |
| Peptides-func o5cdk766 VN×LR | ✅ `34427481` (50 jobs) | `34427481` |
| Table 6 VOC (SiGMA / Hetero ± ungated) | ✅ | `32717593` |
| Table 6 VOC Homog_MP | ✅ | `33810534` |
| Table 6 VOC Homog_MP_ungated relaunch | ✅ `34070244` (4/5); seed1 retry ✅ `34409933` | `34409933` |
| Heterogeneity TU relaunch | ✅ `34410913`; proteins_sigma ✅ `34869869` | `34869869` |
| ENZYMES SiGMA a8g8 hetero | 🛑 TO RUN | — |
| TU all-layer activations | ✅ | `34869795` |
| Table 6 1-MP peptides | ✅ | `32717625` |
| Table 6 COCO relaunch | ✅ mweber `34070245`; H200 twin `34098527` | `34070245` / `34098527` |
| ENZYMES ogpkubk9 sweep | 🛑 TO RUN | — |
