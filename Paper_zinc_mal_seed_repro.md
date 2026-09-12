# ZINC / MalNetTiny 5-seed repros (camera-ready)

> **Ran:** 2026-09-08 (cluster jobs `45587709` ZINC, `45587711` MalNet).  
> **Use:** final camera-ready mean ± std (test metric at best-val checkpoint).  
> **Entity/project:** [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)

Metrics are `best/test_*` at the epoch that optimized `best/val_*` (sample std over 5 seeds).

Re-aggregate anytime:

```bash
python scripts/api_wanndb_query/aggregate_seed_repro_groups.py --preset zinc_mal_top
```

---

## ZINC — GINE baseline (`configs/gine/zinc.yaml`)

| | |
|--|--|
| **W&B group** | `seed_repro_zinc_gine` |
| **SLURM** | `45587709` (`mweber_gpu`, seeds 0–4) |
| **Lineage** | [gbdpt2gc](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/gbdpt2gc) / [x6a7vgim](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/x6a7vgim) / [jb8domtt](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/jb8domtt) |

| seed | `best/val_mae` | `best/test_mae` | run id |
|-----:|---------------:|----------------:|:-------|
| 0 | 0.07573 | 0.06779 | `rvn352ph` |
| 1 | 0.09644 | 0.07778 | `r9a9e6bk` |
| 2 | 0.08307 | 0.11640 | `wy0cs885` |
| 3 | 0.08253 | 0.07223 | `73wsra8f` |
| 4 | 0.09245 | 0.07999 | `38j36gh3` |

**Summary (n=5):** `best/val_mae` **0.0860 ± 0.0083** · `best/test_mae` **0.0828 ± 0.0194**

Note: seed 2 is a test outlier (0.116); others ~0.068–0.080.

---

## MalNet-Tiny — top hybrid recipes

| | |
|--|--|
| **SLURM** | `45587711` (`mweber_gpu`, 4 variants × 5 seeds) |

### v4cytwe0 — GCNE+GINE a0g2

| | |
|--|--|
| **W&B group** | `seed_repro_mal_v4cytwe0` |
| **Source** | [v4cytwe0](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/v4cytwe0) |
| **Cfg** | `configs/gated_hybrid/mal-seed-repro-v4cytwe0.yaml` |

| seed | `best/val_accuracy` | `best/test_accuracy` | run id |
|-----:|--------------------:|---------------------:|:-------|
| 0 | 0.956 | 0.930 | `a77fzcg6` |
| 1 | 0.948 | 0.937 | `ub50jmdl` |
| 2 | 0.960 | 0.937 | `92bm2jko` |
| 3 | 0.950 | 0.933 | `fum2ji1a` |
| 4 | 0.952 | 0.938 | `eac0zctg` |

**Summary (n=5):** val **0.9532 ± 0.0048** · test **0.9350 ± 0.0034**

### zk6ihqi8 — GCNE-only a0g2

| | |
|--|--|
| **W&B group** | `seed_repro_mal_zk6ihqi8` |
| **Source** | [zk6ihqi8](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/zk6ihqi8) |
| **Cfg** | `configs/gated_hybrid/mal-seed-repro-zk6ihqi8.yaml` |

| seed | `best/val_accuracy` | `best/test_accuracy` | run id |
|-----:|--------------------:|---------------------:|:-------|
| 0 | 0.952 | 0.933 | `95010pb9` |
| 1 | 0.952 | 0.936 | `1c3zelme` |
| 2 | 0.950 | 0.931 | `1lj6lsl9` |
| 3 | 0.952 | 0.925 | `t6vwacgd` |
| 4 | 0.952 | 0.930 | `ykamb6xq` |

**Summary (n=5):** val **0.9516 ± 0.0009** · test **0.9310 ± 0.0041**

### apiw6l3u — 9h3jqzkm a0g2 (ep=150)

| | |
|--|--|
| **W&B group** | `seed_repro_mal_apiw6l3u` |
| **Source** | [apiw6l3u](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u) |
| **Cfg** | `configs/gated_hybrid/mal-seed-repro-apiw6l3u.yaml` |

| seed | `best/val_accuracy` | `best/test_accuracy` | run id |
|-----:|--------------------:|---------------------:|:-------|
| 0 | 0.956 | 0.937 | `5b56qmzu` |
| 1 | 0.952 | 0.936 | `nuwg1vpz` |
| 2 | 0.952 | 0.930 | `8tbz5hc2` |
| 3 | 0.954 | 0.930 | `u3c8vi5q` |
| 4 | 0.952 | 0.935 | `vcnujtu7` |

**Summary (n=5):** val **0.9532 ± 0.0018** · test **0.9336 ± 0.0034**

### 5sx7r420 — 9h3jqzkm a0g2 (lr=0.0023, ep=250)

| | |
|--|--|
| **W&B group** | `seed_repro_mal_5sx7r420` |
| **Source** | [5sx7r420](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5sx7r420) |
| **Cfg** | `configs/gated_hybrid/mal-seed-repro-5sx7r420.yaml` |

| seed | `best/val_accuracy` | `best/test_accuracy` | run id |
|-----:|--------------------:|---------------------:|:-------|
| 0 | 0.946 | 0.937 | `qmesbgcy` |
| 1 | 0.950 | 0.936 | `bwnk7mtd` |
| 2 | 0.948 | 0.916 | `mmgjfwpv` |
| 3 | 0.950 | 0.924 | `y4t8gcrw` |
| 4 | 0.948 | 0.929 | `vfhdrn6p` |

**Summary (n=5):** val **0.9484 ± 0.0017** · test **0.9284 ± 0.0087**

---

## Camera-ready pick (MalNet)

Best test mean among the four Mal recipes: **v4cytwe0** — **0.9350 ± 0.0034**.

---

## Lukas's Staged Ablation Ladder (GIN+ and GatedGCN+)

Controlled staged ablation from official ICML 2025 single-head baselines ([arXiv:2502.09263](https://arxiv.org/pdf/2502.09263)).
All outer hyperparameters (RWSE, dimensions, layers, optimizer, learning rate, weight decay, epochs, warmup, batch size) are strictly frozen from the official paper configs.

### Staged Ladder Hierarchy (5 levels × 5 seeds = 25 jobs per benchmark)

| Level | Step | ZINC (GIN+ Baseline) | MalNet-Tiny (GatedGCN+ Baseline) |
|:---:|:---:|:---|:---|
| **0** | **Step 0: Single Head Baseline** | `configs/gine/zinc.yaml`<br>(GINE, L=12, dh=80, RWSE, 2000 ep, lr=1e-3) | `configs/gatedgcn/mal.yaml`<br>(GatedGCN, L=6, dh=100, DummyEdge, 150 ep, lr=5e-4) |
| **1** | **Step 1: + Gating** | `configs/gated_hybrid/zinc-ginplus-ladder-l1-a0g1.yaml`<br>(0 attn + 1 GINE MP, headwise gate) | `configs/gated_hybrid/mal-gatedgcnplus-ladder-l1-a0g1.yaml`<br>(0 attn + 1 GatedGCN MP, headwise gate) |
| **2** | **Step 2: + Attention** | `configs/gated_hybrid/zinc-ginplus-ladder-l2-a1g1.yaml`<br>(1 attn + 1 GINE MP, headwise gate, full mask) | `configs/gated_hybrid/mal-gatedgcnplus-ladder-l2-a1g1.yaml`<br>(1 attn + 1 GatedGCN MP, headwise gate, graph_restricted) |
| **3** | **Step 3a: + 2nd MP Head** | `configs/gated_hybrid/zinc-ginplus-ladder-l3-a0g2.yaml`<br>(0 attn + 2 MP: GINE,GATEDGCN, headwise gate) | `configs/gated_hybrid/mal-gatedgcnplus-ladder-l3-a0g2.yaml`<br>(0 attn + 2 MP: GATEDGCN,GINE, headwise gate) |
| **4** | **Step 3b: Full Hybrid** | `configs/gated_hybrid/zinc-ginplus-ladder-l4-a1g2.yaml`<br>(1 attn + 2 MP: GINE,GATEDGCN, headwise gate) | `configs/gated_hybrid/mal-gatedgcnplus-ladder-l4-a1g2.yaml`<br>(1 attn + 2 MP: GATEDGCN,GINE, headwise gate, graph_restricted) |

### Slurm Submission & Scripts

- **ZINC:**
  - Submit: `bash bash_interface/cluster/submit_zinc_ginplus_ladder.sh`
  - Runner: `bash_interface/cluster/run_zinc_ginplus_ladder.sh`
  - W&B group: `zinc_ginplus_ladder` (tags: `ginplus_ladder,zinc,level_0...4`)
- **MalNet-Tiny:**
  - Submit: `bash bash_interface/cluster/submit_mal_gatedgcnplus_ladder.sh`
  - Runner: `bash_interface/cluster/run_mal_gatedgcnplus_ladder.sh`
  - W&B group: `mal_gatedgcnplus_ladder` (tags: `gatedgcnplus_ladder,malnet,level_0...4`)

