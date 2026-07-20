# SiGMA paper Table 6 — LRGB best models with 1 MPGNN head

For datasets whose **paper-best SiGMA** uses `num_gnn_heads=1`, we re-run the best model and ablate **adding one MP head** (same type vs different type), with and without gating.

VOC already has 2 MP heads → see [`Paper_table6_voc.md`](Paper_table6_voc.md).

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)

---

## 1. Frozen 1-MP best models

| Dataset | Paper SiGMA | Anchor | Exemplar | Base MP |
|---------|-------------|--------|----------|---------|
| Peptides-func | AP 0.7052±0.0056 | `peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml` | [`l31u4b3k`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/l31u4b3k) | `GCN` |
| Peptides-struct | MAE 0.2441±0.0017 | `peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml` | [`bqkect9l`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/bqkect9l) | `GINE` |
| COCO-SP | F1 0.4155±0.0076 | `coco-hybrid-5b4z9l3u-a1g1-anchor.yaml` | [`xgjakrz0`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xgjakrz0) | `GATEDGCN` |

Attention head counts stay as in the anchor (`na=1`). Only `num_gnn_heads` / `gnn_types` / `gate` change in ablations.

---

## 2. Five variants × 5 seeds × 3 datasets = 75 jobs

| Variant (W&B) | Meaning | Override |
|---------------|---------|----------|
| **`SiGMA`** | Best model as-is | none |
| **`Homog_MP`** | +1 MP head, **same** type, gated | `num_gnn_heads=2`, types `T,T` |
| **`Hetero_MP`** | +1 MP head, **different** type, gated | `num_gnn_heads=2`, types `T,U` |
| **`Homog_MP_ungated`** | +1 same-type MP, no gates | above + `gate=none` |
| **`Hetero_MP_ungated`** | +1 different-type MP, no gates | above + `gate=none` |

### Second-head type choices

| Dataset | Homog (`T,T`) | Hetero (`T,U`) | Rationale |
|---------|---------------|----------------|-----------|
| peptides_func | `GCN,GCN` | `GCN,GINE` | Classic local mix used in hybrid sweeps |
| peptides_struct | `GINE,GINE` | `GINE,GGNN` | Best multi-MP struct configs used `GINE,GGNN` |
| coco | `GATEDGCN,GATEDGCN` | `GATEDGCN,GCN` | Matches VOC Table 6 hetero mix |

### W&B

- **Group:** `paper_T6_<dataset>_<Variant>`
  - e.g. `paper_T6_coco_Hetero_MP_ungated`
- **Tags:** `paper_table6`, `<Variant>`, `<dataset>`, `seed<k>`, `source_<runid>`

---

## 3. Launch on cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table6_lrgb_1mp_hetero.sh
```

Optional: `PAPER_T6_1MP_PARALLEL=N` (default **10**).

| Field | Value |
|-------|-------|
| **SLURM array** | ✅ **`32717625`** (submitted 2026-07-18, `%7`) |
| **Job name** | `sigma_T6_1mp` |
| **Tasks** | `1-75%7` = 3×5×5 |
| **Scripts** | `submit_paper_table6_lrgb_1mp_hetero.sh` → `run_paper_table6_lrgb_1mp_hetero.sh` |
| **Logs** | `logs_gnnplus/sigma_T6_1mp_32717625_<TASK>.log` |
| **Needs** | `gate=none` support (same branch as Table 5) |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## 4. Aggregate

```bash
# All Table 6 (VOC + 1-MP), or 1-MP pivot only via --table 6
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6 --detail
```

### Fill-in

| Variant | Peptides-func ↑ | Peptides-struct ↓ | COCO-SP ↑ | n |
|---------|-----------------|-------------------|-----------|---|
| SiGMA | | | | 5 |
| Homog_MP | | | | 5 |
| Hetero_MP | | | | 5 |
| Homog_MP_ungated | | | | 5 |
| Hetero_MP_ungated | | | | 5 |

---

## 5. Files

| Path | Role |
|------|------|
| `bash_interface/cluster/submit_paper_table6_lrgb_1mp_hetero.sh` | **launch this** |
| `bash_interface/cluster/run_paper_table6_lrgb_1mp_hetero.sh` | SLURM array worker |
| `configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml` | func SiGMA |
| `configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml` | struct SiGMA |
| `configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml` | COCO SiGMA |

Related: VOC Table 6 → [`Paper_table6_voc.md`](Paper_table6_voc.md).

**Follow-up (🛑 TO RUN):** Homog_MP beat paper SiGMA on peptides-func (0.7080). MP-only control a0g3 GCN×3 → [`Paper_peptides_func_homog_a1g2_mp_only.md`](Paper_peptides_func_homog_a1g2_mp_only.md).
