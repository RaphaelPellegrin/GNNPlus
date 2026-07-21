# SiGMA paper Table 6 — PascalVOC-SP heterogeneous MP heads

Track VOC-only Table 6 ablations (homogeneous vs heterogeneous MPGNN heads ± gates).

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Best VOC baseline: [`Paper_final_runs.md`](Paper_final_runs.md) / Table 5 [`Paper_ablations.md`](Paper_ablations.md)

---

## 1. Frozen best VOC model

| Field | Value |
|-------|-------|
| Dataset | PascalVOC-SP |
| Anchor | `configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml` |
| Exemplar | [`vyt7hjj5`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vyt7hjj5) |
| Paper group | `paper_bestmodel_v1_voc_j7ukyzdm` |
| Arch | a2g2 — 2×attn + 2×MP, `d_h=64`, headwise gate, RMSNorm, L16/H95, ep=200 |
| Best MP types | **`GATEDGCN,GATEDGCN`** (homogeneous) |

---

## 2. Variants

### Already done (hetero campaign) — 3 × 5 = 15 jobs · SLURM `32717593`

| Variant (W&B) | Meaning | Override vs anchor |
|---------------|---------|--------------------|
| **`SiGMA`** | Best gated hybrid (homogeneous MP) | none |
| **`Hetero_MP`** | Heterogeneous MP, still gated | `gnn.hybrid.gnn_types GATEDGCN,GCN` |
| **`Hetero_MP_ungated`** | Heterogeneous MP, no gates | `gnn.hybrid.gnn_types GATEDGCN,GCN` + `gnn.hybrid.gate none` |

### 🛑 TO RUN — Homog_MP fresh seeds — 2 × 5 = 10 jobs

Same architecture as SiGMA, but logged under `Homog_MP` / `Homog_MP_ungated` so the VOC column matches the 1-MP row names.

| Variant (W&B) | Meaning | Override vs anchor |
|---------------|---------|--------------------|
| **`Homog_MP`** | Homogeneous MP, gated (= SiGMA arch) | none |
| **`Homog_MP_ungated`** | Homogeneous MP, no gates | `gnn.hybrid.gate none` |

Maps to paper Table 6 rows (VOC column):

| Paper row | W&B group |
|-----------|-----------|
| Homogeneous MP heads (gated) | **`Homog_MP`** (also had `SiGMA`) |
| Homogeneous MP heads, no gates | **`Homog_MP_ungated`** |
| Heterogeneous MP heads (gated) | **`Hetero_MP`** |
| Heterogeneous MP heads, no gates | **`Hetero_MP_ungated`** |
| SiGMA | **`SiGMA`** |

### W&B

- **Group:** `paper_T6_voc_<Variant>`
  - `paper_T6_voc_SiGMA` ✅
  - `paper_T6_voc_Homog_MP` 🛑
  - `paper_T6_voc_Homog_MP_ungated` 🛑
  - `paper_T6_voc_Hetero_MP` ✅
  - `paper_T6_voc_Hetero_MP_ungated` ✅
- **Tags:** `paper_table6`, `<Variant>`, `voc`, `seed<k>`, `source_vyt7hjj5`

---

## 3. Launch on cluster

### Hetero (already submitted)

```bash
bash bash_interface/cluster/submit_paper_table6_voc_hetero.sh
```

| Field | Value |
|-------|-------|
| **SLURM array** | ✅ **`32717593`** (submitted 2026-07-18, `%3`) |
| **Job name** | `sigma_T6_voc` |
| **Tasks** | `1-15%3` = 3×5 |
| **Scripts** | `submit_paper_table6_voc_hetero.sh` → `run_paper_table6_voc_hetero.sh` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_32717593_<TASK>.log` |

### Homog_MP ± ungated (🛑 TO RUN)

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table6_voc_homog.sh
# optional: PAPER_T6_VOC_HOMOG_PARALLEL=5 (default)
```

| Field | Value |
|-------|-------|
| **SLURM array** | 🛑 *TO RUN — not submitted* |
| **Job name** | `sigma_T6_voc_homog` |
| **Tasks** | `1-10%5` = 2×5 |
| **Scripts** | `submit_paper_table6_voc_homog.sh` → `run_paper_table6_voc_homog.sh` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_homog_<JOBID>_<TASK>.log` |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## 4. Aggregate

```bash
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6 --detail
```

### Fill-in (PascalVOC-SP F1 ↑)

| Model | VOC-SP F1 ↑ | n | W&B group |
|-------|-------------|---|-----------|
| SiGMA | 0.4669±0.0083 | 5 | `paper_T6_voc_SiGMA` |
| Homog_MP | 🛑 | 5 | `paper_T6_voc_Homog_MP` |
| Homog_MP_ungated | 🛑 | 5 | `paper_T6_voc_Homog_MP_ungated` |
| Hetero_MP | 0.4484±0.0064 | 5 | `paper_T6_voc_Hetero_MP` |
| Hetero_MP_ungated | 0.4518±0.0055 | 5 | `paper_T6_voc_Hetero_MP_ungated` |

---

## 5. Files

| Path | Role |
|------|------|
| `bash_interface/cluster/submit_paper_table6_voc_hetero.sh` | hetero campaign (done) |
| `bash_interface/cluster/run_paper_table6_voc_hetero.sh` | hetero worker |
| `bash_interface/cluster/submit_paper_table6_voc_homog.sh` | **launch Homog_MP ± ungated** |
| `bash_interface/cluster/run_paper_table6_voc_homog.sh` | Homog worker |
| `configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml` | VOC SiGMA / Homog anchor |
| `GNNPlus/layer/gated_hybrid_layer.py` | `gate=none` for ungated variant |
