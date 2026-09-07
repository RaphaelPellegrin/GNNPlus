# 🚦 Cluster launches tracker (SiGMA paper week)

> **Status legend**  
> ✅ **SUBMITTED / RUNNING** — job already on FASRC  
> 🛑 **TO RUN** — cluster full / wait — do **not** forget these  
> 📊 Aggregate locally with W&B when finished  

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)

---

## ✅ SUBMITTED (do not re-submit)

### 🧪 TU preferred-head MASK eval (GCN+SAGE gated)

| | |
|--|--|
| **Status** | ✅ **COMPLETED** `44924809` (exit 0, ~2.5 min; gcs a0g2+a1g2 gated) |
| **When** | 2026-09-06 · prior `44903886` FAILED (MUTAG yaml / ENZYMES ckpt) |
| **Submit** | `bash bash_interface/cluster/submit_eval_tu_preferred_head_masks.sh` |
| **Worker** | `bash_interface/cluster/run_eval_tu_preferred_head_masks.sh` |
| **Script** | `scripts/heterogeneity/eval_tu_preferred_head_masks.py` |
| **Family** | `TU_PREF_MASK_FAMILY=gcs` · ops `GCN,SAGE` · seeds 0–4 · mutag+enzymes |
| **Partition** | `gpu_h200` · nice=0 |
| **Also later** | `TU_PREF_MASK_FAMILY=a2g4` or `both` |
| **Docs** | [`Paper_tu_gate_hetero_bridge.md`](Paper_tu_gate_hetero_bridge.md) |
| **Log** | `logs_gnnplus/tu_pref_mask_44924809.log` |
| **Outs** | `results/heterogeneity/tu_pref_mask_gcs_{a0g2_gated,a1g2_gated}/` |

```bash
squeue -u $USER -j 44924809
tail -f logs_gnnplus/tu_pref_mask_44924809.log
# after DONE, rsync results/heterogeneity/tu_pref_mask_gcs_* back to Mac
```

### 🧪 Xu SiGMA GCN+SAGE only (a0g2/a1g2 × gated/ungated × 5 seeds)

| | |
|--|--|
| **Status** | ✅ **DONE** 40/40 gate dumps (2026-09-06) |
| **SLURM** | **`44886381`** `1-40%10` all COMPLETED |
| **When** | 2026-09-06 |
| **Tasks** | 40 = 4 variants × 2 ds × 5 seeds |
| **Variants** | `a0g2_gated` · `a0g2_ungated` · `a1g2_gated` · `a1g2_ungated` (`gnn_types=GCN,SAGE`) |
| **Docs** | [`Paper_tu_gate_hetero_bridge.md`](Paper_tu_gate_hetero_bridge.md) |
| **Logs** | `logs_gnnplus/xu_sigma_gcs_44886381_<TASK>.log` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage/<ds>_SiGMA_hetero_<variant>_seed<s>/` |
| **Configs** | `configs/heterogeneity/powerful_gnns/sigma-gcn-sage-{a0g2,a1g2}-{gated,ungated}-ckpt.yaml` |

```bash
ls $GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage/*/gate_values_per_graph.pt | wc -l
# = 40
```

### 🧪 GIN depth-routing synthetic (2-layer SiGMA, 20 jobs)

| | |
|--|--|
| **Status** | ✅ **COMPLETED** 2026-09-04 (`20/20`) |
| **SLURM** | **`44510747`** `1-20%8` |
| **When** | 2026-09-04 |
| **Tasks** | gated + ungated × lr001/lr01 × seeds 0–4 |
| **Dataset** | `$GNNPLUS_DATASET_DIR/GinDepthRouting` (10k/2k/2k) |
| **Docs** | [`results/gin_routing_depth/README.md`](results/gin_routing_depth/README.md) |
| **W&B** | `paper_gin_depth_routing_l2_a0g1_{gated,ungated}_{lr001,lr01}` |
| **Gates** | `gates_by_tau_depth/{val,test}/layer{k}/tau{0,1}/mean_gamma` |
| **Logs** | `logs_gnnplus/gin_depth_rt_44510747_<TASK>.log` |
| **Out** | `$GNNPLUS_OUT_DIR/gin_routing_depth/toy/<model>_<lr>_seed<s>/` |

### 🧪 GIN depth-routing full analysis (gcn_gin-style)

| | |
|--|--|
| **Status** | 🔄 **RESUBMITTED** 2026-09-04 after NaN-summary fix |
| **SLURM** | **`44514923`** (prior fail **`44514347`**) |
| **When** | 2026-09-04 |
| **What** | per-τ acc + layer×τ gates + opposite-sign + layer-mask + gate dump + ranked plots |
| **Root** | `$GNNPLUS_OUT_DIR/gin_routing_depth` |
| **Out** | `results/gin_routing_depth/analysis/` |
| **Logs** | `logs_gnnplus/gin_depth_full_an_44514347.log` |

```bash
squeue -u $USER -j 44514923
sacct -j 44514923 -X --format=State,Elapsed,ExitCode -n
tail -f logs_gnnplus/gin_depth_full_an_44514923.log
# after COMPLETED, on Mac:
# bash bash_interface/local/pull_gin_depth_routing_results.sh --bundle
```

### 🧪 GAT specialist hetero profiles (MUTAG / ENZYMES, ≥100 apps)

| | |
|--|--|
| **Status** | ✅ **DONE** enzymes_gat (`44507938_2`) · mutag_gat (`44264271_1`) |
| **SLURM** | **`44507938`** ✅ · prior **`44264271`** (1 ✅ / 2 CUDA → retry) |
| **When** | 2026-09-05 |
| **Tasks** | mutag_gat + enzymes_gat pickles on disk |
| **Docs** | [`Paper_tu_gate_hetero_bridge.md`](Paper_tu_gate_hetero_bridge.md) |
| **Logs** | `logs_gnnplus/hetero_gate_bridge_44507938_2.log` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/{mutag,enzymes}_gat/` |
| **Configs** | `configs/heterogeneity/powerful_gnns/{mutag,enzymes}-gat.yaml` |

```bash
ls $GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/{mutag,enzymes}_gat/*graph_dict.pickle
```

Re-join with `--operators GCN,GIN,SAGE,GAT` after both dirs have `*_graph_dict.pickle`.

### 🧪 Xu SiGMA a2g4 ckpt + gates (MUTAG / ENZYMES × 5 seeds)

| | |
|--|--|
| **Status** | ✅ **DONE** 10/10 gate dumps (2026-09-05) |
| **SLURM** | retry **`44507936`** ✅ + prior **`44264269`** tasks 2/5/8 |
| **When** | 2026-09-05 |
| **Tasks** | mutag/enzymes seeds 0–4 all have `gate_values_per_graph.pt` |
| **Docs** | [`Paper_tu_gate_hetero_bridge.md`](Paper_tu_gate_hetero_bridge.md) |
| **W&B** | `xu_sigma_a2g4_{mutag,enzymes}` |
| **Logs** | `logs_gnnplus/xu_sigma_a2g4_44507936_<TASK>.log` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/<ds>_SiGMA_hetero_xu_seed<s>/` |

```bash
ls $GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/*/gate_values_per_graph.pt | wc -l
# = 10
```

### 🧪 Table 5 — PATTERN + CLUSTER seed-20 fill (165 jobs)

| | |
|--|--|
| **Status** | 🔄 **IN PROGRESS** (2026-09-03 ~08:50) — CLUSTER ✅ **60/60** · PATTERN **20/90** |
| **SLURM** | CLUSTER **`43796006`** (SiGMA 1–15 ✅) + **`43916889`** (ablations 16–75 ✅) · PATTERN **`43796007`** 🔄 20/90 |
| **When** | 2026-09-01 original · 2026-09-02 CLUSTER ablation resubmit |
| **Tasks** | CLUSTER `1-75%5` · PATTERN `1-90%5` |
| **Docs** | [`Paper_table5_seed20_fill.md`](Paper_table5_seed20_fill.md) |
| **W&B** | `paper_T5_cluster_*` (seeds 5–19) · `paper_T5_pattern_gritvn4_*` (seeds 10–24) |
| **Logs** | `sigma_T5_cl_s20_43796006_<TASK>.log` · `sigma_T5_cl_s20_43916889_<TASK>.log` · `sigma_T5_pat_s20_43796007_<TASK>.log` |
| **Note** | 43796006 tasks 16–75 failed (`sbatch --export` comma split); resubmitted as 43916889 ✅ |

```bash
# 📊 re-export paired t-test CSVs when done
python scripts/api_wanndb_query/export_table5_paired_ttest_data.py
```

### 🧪 Table 5 — LRGB ablations (80 jobs)

| | |
|--|--|
| **SLURM** | **`32232124`** |
| **When** | 2026-07-17 |
| **Tasks** | `1-80%18` |
| **Docs** | [`Paper_ablations.md`](Paper_ablations.md) |
| **W&B** | `paper_T5_<ds>_{SiGMA,SiGMA_ungated,SiGMA_attn_gate,Attn_only,MP_only}` |
| **Logs** | `logs_gnnplus/sigma_T5_abl_32232124_<TASK>.log` |

```bash
# 📊 aggregate
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5
```

### 🧪 Table 6 — SiGMA_attn_gate (20 jobs)

| | |
|--|--|
| **SLURM** | **`35354579`** |
| **When** | 2026-07-26 |
| **Tasks** | `1-20%10` |
| **Partition** | `mweber_gpu` · 120h · out_dir netscratch |
| **Docs** | [`Paper_ablations.md`](Paper_ablations.md) |
| **W&B** | `paper_T5_<ds>_SiGMA_attn_gate` |
| **Override** | `gnn.hybrid.mp_gate none` (yaml `gate` kept) |
| **Logs** | `logs_gnnplus/sigma_T5_attn_gate_35354579_<TASK>.log` |

```bash
# 📊 aggregate (includes SiGMA_attn_gate row)
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5
```

### 🧪 Table 6 — Hybrid ungated Att (`SiGMA_ungated_attn`, 35 jobs)

| | |
|--|--|
| **SLURM** | ✅ **`36605829`** |
| **When** | 2026-07-31 |
| **Tasks** | `1-35%10` |
| **Partition** | `mweber_gpu` · 120h · out_dir netscratch |
| **Docs** | [`Paper_ablations.md`](Paper_ablations.md) |
| **W&B** | `paper_T5_<ds>_SiGMA_ungated_attn` |
| **Override** | `gate none` (attn) + `mp_gate` = yaml style (MP gated) |
| **Logs** | `logs_gnnplus/sigma_T5_ungated_attn_36605829_<TASK>.log` |

### 🧪 Heterogeneity full TU (Xu HPs, 24 jobs)

| | |
|--|--|
| **SLURM** | ✅ **`36604947`** · `1-24%8` |
| **Gate-viz** | ✅ **`36604951`** · `1-6` |
| **Docs** | [`Paper_heterogeneity.md`](Paper_heterogeneity.md) |

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

**COCO dead-seed retry** (H200 ≤10 GPUs; only seeds with no finished twin):

| JOBID | Tasks | What |
|-------|-------|------|
| ✅ **`3524720?`** | T6 `57,60,61,63,65,67` | Homog_MP 1/4 · Hetero_MP 0/2/4 · Homog_ungated 1 · full ep · `_h200_retry` — *paste first JOBID from submit if not 35247207* |
| ✅ **`35247208`** | T5 `62,63` | ep150 SiGMA seeds 1,2 · `_ep150_h200_retry` |
| ✅ **`35247209`** | T6 `55` | ep150 SiGMA seed 4 · `_ep150_h200_retry` |

Submitted 2026-07-26. W&B: `paper_T6_coco_*` / `paper_T5_ep150_coco_SiGMA` / `paper_T6_ep150_coco_SiGMA`.  
Logs: `logs_gnnplus/sigma_T6_1mp_<JOBID>_*.log`, `sigma_T5_abl_35247208_*.log`.

```bash
COCO_DEAD_PARTITION=gpu_h200 COCO_DEAD_PARALLEL=10 \
  COCO_DEAD_INCLUDE_EP150=1 \
  bash bash_interface/cluster/submit_coco_dead_seeds_relaunch.sh
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

### 🧪 COCO Table 6 a1g2 twins — ungated + attn_gate (10 jobs, 300 ep)

| | |
|--|--|
| **SLURM** | ✅ **`36161505`** (2026-07-30) |
| **Submit** | `bash_interface/cluster/submit_paper_table5_coco_ungated_a1g2.sh` |
| **Tasks** | `1-10%5` · 300 ep · a1g2 GATEDGCN×2 |
| **W&B** | `paper_T5_coco_SiGMA_ungated_a1g2` · `paper_T5_coco_SiGMA_attn_gate_a1g2` |
| **Logs** | `logs_gnnplus/sigma_T5_coco_a1g2_36161505_<TASK>.log` |

---

### 🧪 ENZYMES ogpkubk9 gate-viz (1 job)

| | |
|--|--|
| **SLURM** | ✅ **`36148089`** (2026-07-29) |
| **Docs** | [`Paper_enzymes_ogpkubk9.md`](Paper_enzymes_ogpkubk9.md) §3 |
| **Submit** | `bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh` |
| **Default** | plateau · seed 2 · `enable_ckpt` · period 50 |
| **out_dir** | `$GNNPLUS_OUT_DIR/gate_viz_enzymes_ogpkubk9_plateau_seed2` |
| **W&B** | `enzymes_ogpkubk9_gate_viz` / `enzymes_gate_viz_plateau_seed2` |
| **Logs** | `logs_gnnplus/enz_gate_viz_36148089.log` |
| **Dump** | `submit_dump_enzymes_ogpkubk9_gates.sh` → `gate_values_per_graph.pt` |

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
| **SLURM** | ✅ **`34869787`** (2026-07-24) · seed4 retry ✅ **`35773784`** |
| **Tasks** | `1-10%5` (1–5 Attn a3g0; 6–10 MP a0g3 GATEDGCN×3) |
| **Epochs** | **150** (`_ep150`) |
| **Docs** | [`Paper_ablations.md`](Paper_ablations.md) |
| **W&B** | `paper_T5_coco_Attn_only_a3` · `paper_T5_coco_MP_only_a0g3` |
| **Logs** | `logs_gnnplus/sigma_T5_coco_a3_34869787_<TASK>.log` · retry `…_35773784_10.log` |

---

### 🧪 COCO Table 6 a1g1 @ ep150 — fill crashed / never launched

| | |
|--|--|
| **Status** | 🔄 **RUNNING** |
| **Baseline** | a1g1 SiGMA → Attn a2 / MP a0g2 |
| **T5 SLURM** | ✅ **`35720666`** · `62,75,76-80%10` · `mweber_gpu` · 96h |
| **attn_gate SLURM** | ✅ **`35720667`** · `16-20%10` · `mweber_gpu` · 96h |
| **Skip** | ungated done; Attn seeds 0–3 still running (prior array) |
| **W&B** | `paper_T5_ep150_coco_{SiGMA,Attn_only,MP_only,SiGMA_attn_gate}` |
| **Logs** | `logs_gnnplus/sigma_T5_abl_35720666_<TASK>.log` · `sigma_T5_attn_gate_35720667_<TASK>.log` |

Submitted 2026-07-28. Tasks: 62=SiGMA s1 · 75=Attn s4 · 76–80=MP a0g2 · 16–20=attn_gate.
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
║  ✅  DONE  ·  GCN/GIN routing synthetic · toy + sigma training         ║
║  🛑  TO RUN  ·  d_h fill toy_dh2/toy_dh4/sigma_dh2  (%3)                ║
║  📄  Paper_gcn_gin_routing_synthetic.md                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
# after git pull — Appendix H width fill (same 4 models × 2 LR × 5 seeds)
GCN_GIN_ROUTING_PARALLEL=3 \
  bash bash_interface/cluster/submit_gcn_gin_routing.sh dh_fill
```

| Field | Value |
|-------|-------|
| **Main train toy** | ✅ **42432154** · Track A · **d_h=1** |
| **Main train sigma** | ✅ **42432155** · Track B · **d_h=4** |
| **dh_fill** | ✅ **DONE** Track A `d_h∈{2,3,4}` **45066651–53** · Track B `d_h∈{1,2,3}` **45066654–56** · 240/240 |
| **routing_2 analysis** | `submit_gcn_gin_routing_2_pipeline.sh` → `results/gcn_gin_routing_2/analysis/` |
| **Models** | SiGMA gated · SiGMA ungated · GCN-only · GIN-only |
| **Dataset** | existing `$GNNPLUS_DATASET_DIR/GcnGinRouting` |
| **Out** | `$GNNPLUS_OUT_DIR/gcn_gin_routing/{toy_dh2,toy_dh3,toy_dh4,sigma_dh1,sigma_dh2,sigma_dh3}/` |

| Field | Value |
|-------|-------|
| **Forward gated** | ✅ **42759900** — 3/4 PNGs in `forward_traces/` |
| **Forward GCN-only** | 🔄 **42816151** → `forward_traces/gcn_only/` |
| **Forward GIN-only** | 🔄 **42816156** → `forward_traces/gin_only/` |
| **Noxenc toy train** | 🔄 **42816595** (`1-4%4`) — gated/ungated/gcn/gin |
| **Next** | Pull figures · noxenc forward traces after **42816595** · τ=1 incorrect panel |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Monitor
squeue -u $USER | grep -E 'gcn_gin|fwd_trace'
tail -f logs_gnnplus/gcn_gin_fwd_trace_42816151.log
tail -f logs_gnnplus/gcn_gin_noxenc_42816595_*.log

# After noxenc train
GCN_GIN_FORWARD_RUN_DIR=$GNNPLUS_OUT_DIR/gcn_gin_routing/toy/a0g2_gated_noxenc_lr001_seed0 \
GCN_GIN_FORWARD_OUT_DIR=$PWD/results/gcn_gin_routing/analysis/forward_traces/noxenc_gated \
  bash bash_interface/cluster/submit_plot_gcn_gin_routing_forward_trace.sh
```

| Field | Value |
|-------|-------|
| **Tracks** | toy (Track A routing convs) + sigma (PyG GIN/GCN) |
| **Partition** | `mweber_gpu` |
| **W&B** | tag `gcn_gin_routing_synthetic` · `paper_gcn_gin_routing_{toy,sigma}_*` |
| **Out** | `$GNNPLUS_OUT_DIR/gcn_gin_routing/<track>/` |
| **toy JOBID** | ✅ **42432154** |
| **sigma JOBID** | ✅ **42432155** |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  GatedGCN retry  ·  TU gate–operator bridge · 44218244 (4,8)        ║
║  📄  Paper_tu_gate_hetero_bridge.md                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **GCN/GIN/SAGE** | ✅ mutag + enzymes (`43789365_1` · `44164801`) |
| **GatedGCN retry** | 🔄 **`44218244`** · `4,8%4` · ≥100 appearances |
| **When** | 2026-09-03 · holylogin08 |
| **Out** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/` |
| **Logs** | `logs_gnnplus/hetero_gate_bridge_44218244_{4,8}.log` |

```bash
squeue -u $USER -j 44218244
head -60 logs_gnnplus/hetero_gate_bridge_44218244_4.log   # mutag_gatedgcn
head -60 logs_gnnplus/hetero_gate_bridge_44218244_8.log   # enzymes_gatedgcn
```

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑  TO RUN  ·  Hetero MUTAG/ENZYMES · Xu et al. ICLR 2019 HPs (6 jobs) ║
║  🧪  GCN / GIN / SiGMA · arXiv:1810.00826 / weihua916/powerful-gnns      ║
║  📄  Paper_heterogeneity.md                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_heterogeneity_powerful_gnns.sh
# 👉 paste JOBID into Paper_heterogeneity.md + here
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **W&B** | `building_hetero_profile_{mutag,enzymes}_powerful_gnns` |
| **Configs** | `configs/heterogeneity/powerful_gnns/` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  RUNNING  ·  Table 6 MNIST + CIFAR10 + PATTERN · 35720034           ║
║  🧪  SiGMA / ungated / attn_gate / Attn_only / MP_only × 5 seeds         ║
║  📄  Paper_ablations_mnist_cifar.md                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **SLURM** | `35720034` (`1-75%10`, 96GB / 96h) |
| **Logs** | `logs_gnnplus/sigma_T5_mc_35720034_<TASK>.log` |
| **W&B** | `paper_T5_{mnist,cifar10,pattern}_{SiGMA,SiGMA_ungated,SiGMA_attn_gate,Attn_only,MP_only}` |
| **Aggregate** | `python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc` |

**Retries (2026-07-28 / 07-29):**
| SLURM | What |
|-------|------|
| ✅ **`35773781`** | MNIST `SiGMA_ungated` seed4 (task 10) |
| ✅ **`35773784`** | COCO `MP_only_a0g3` seed4 (task 10, `_ep150_retry`) |
| ✅ **`36000721`** | MNIST `SiGMA_attn_gate` redo seeds 0–4 (tasks `11-15`) |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  SUBMITTED 2026-08-03 · remaining Table 5/6 gaps                     ║
║  🧪  gap-fill + PATTERN gritvn4 T5/T6 + CLUSTER ht9bntg2 T5/T6             ║
║  📄  submit_paper_table56_remaining_gaps.sh                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Array | JOBID | What |
|-------|-------|------|
| T5 gap-fill | **`36912369`** | CIFAR `MP_only` (1–5) + COCO `ungated_attn` seeds 1–4 (6–9) |
| T5 PATTERN gritvn4 | **`36912370`** | 25 jobs · `paper_T5_pattern_gritvn4_*` · seeds 5–9 |
| T6 PATTERN gritvn4 | **`36912372`** | 15 jobs · `paper_T6_pattern_gritvn4_*` |
| T5 CLUSTER | **`36912373`** | 25 jobs · `paper_T5_cluster_*` · ht9bntg2 |
| T6 CLUSTER | **`36912374`** | 20 jobs · `paper_T6_cluster_*` · +1 MP |

| Field | Value |
|-------|-------|
| **Anchor PATTERN SiGMA** | `paper_sigma_grit_attn_pattern_vn4` · **87.395±0.194%** |
| **Anchor CLUSTER SiGMA** | `paper_bestmodel_v1_cluster_ht9bntg2` · **78.956±0.112%** |
| **Already done** | CIFAR `ungated_attn` 79.754±0.339% · CIFAR `Hetero_MP` 79.262±0.405% |
| **Watch** | `squeue -u $USER` · logs `logs_gnnplus/sigma_T5_gap_36912369_*.log` etc. |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  RUNNING  ·  Table 7 MNIST/CIFAR/PATTERN · 35721068 (45 jobs)       ║
║  🧪  Homog_ungated / Hetero / Hetero_ungated (1-head swap)               ║
║  📄  Paper_table6_mnist_cifar_pattern.md                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`35721068`** (`1-45%10`, 96GB / 96h) · cancelled prior `35720920` |
| **Logs** | `logs_gnnplus/sigma_T6_mc_35721068_<TASK>.log` |
| **W&B** | `paper_T6_{mnist,cifar10,pattern}_{Homog_MP_ungated,Hetero_MP,Hetero_MP_ungated}` |
| **Reuse** | SiGMA/Homog gated ← `paper_bestmodel_v1_*` |
| **CIFAR hetero** | `GATEDGCN×3,GCN` (one head swap) |
| **Aggregate** | `python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6mc` |


---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37434534  ·  2026-08-05  ·  1-150%8            ║
║  🎯  6 TU × {GCN, homo a2g4×2LR, hetero a2g4×2LR} × 5 seeds            ║
║  📄  Paper_tu_sigma_homo_hetero.md                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_sigma_homo_hetero.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37434534`** |
| **Tasks** | `1-150%8` |
| **Docs** | [`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) |
| **W&B** | `tu_hh_<ds>_{GCN,SiGMA_homo,SiGMA_hetero}_{lr001,lr01}` |
| **Ckpt + gates** | best-val `ckpt/` · SiGMA auto `gate_values_per_graph.pt` |
| **Logs** | `logs_gnnplus/tu_sigma_hh_37434534_<TASK>.log` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  DONE  ·  SLURM 37574967  ·  REDDIT SiGMA n=5 both LRs               ║
║  🎯  TU social COLLAB / IMDB-BINARY / REDDIT-BINARY                      ║
║  📄  Paper_tu_sigma_homo_hetero.md                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ finished — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_sigma_social.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37574967`** (done) |
| **Tasks** | `1-75%20` · mem 128GB |
| **Docs** | [`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) |
| **Batches** | COLLAB 32 · IMDB 64 · REDDIT 16 |
| **W&B** | `tu_hh_{collab,imdb_binary,reddit_binary}_*` |
| **Logs** | `logs_gnnplus/tu_sigma_soc_37574967_<TASK>.log` |
| **REDDIT** | GCN 92.60±1.62 · homo 87.92±7.51 · **hetero 92.72±1.01** |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37649411  ·  1-90%20                            ║
║  🎯  TU standalone GIN / SAGE / GAT (paper table set)                    ║
║  📄  Paper_tu_sigma_homo_hetero.md                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_mpgnn_baselines.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37649411`** |
| **Submit** | `bash_interface/cluster/submit_tu_mpgnn_baselines.sh` |
| **Tasks** | `1-90%20` · 6 ds × {GIN,SAGE,GAT} × 5 seeds |
| **W&B** | `tu_hh_<ds>_{GIN,SAGE,GAT}_lr001` |
| **Logs** | `logs_gnnplus/tu_mpgnn_37649411_<TASK>.log` |
| **Recipe** | same as GCN (L12/H64/lr=1e-3) |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37724579  ·  1-180%20                           ║
║  🎯  TU SiGMA ~1× GCN + GPS-style a1g1                                   ║
║  📄  Paper_tu_sigma_homo_hetero.md                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_sigma_1x_gcn.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37724579`** |
| **Submit** | `bash_interface/cluster/submit_tu_sigma_1x_gcn.sh` |
| **Tasks** | `1-180%20` · mem 128GB |
| **SiGMA** | a2g4 · L12 · H64 · **`d_h=4`** (~1.02× GCN; was ~1.65× at `d_h=16`) |
| **GPS** | a1g1 · GATEDGCN+Transformer · `d_h=8` (~1.01× GCN) |
| **W&B** | `tu_1x_<ds>_{SiGMA_homo,SiGMA_hetero,GPS}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_1x_gcn/` |
| **Docs** | [`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) |
| **Logs** | `logs_gnnplus/tu_1x_gcn_37724579_<TASK>.log` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 44713424  ·  2026-09-05  ·  1-120%20             ║
║  🎯  TU SiGMA ungated (Tab.17 dh16 + Tab.18 dh4) · gate=none · 120 jobs  ║
║  📄  Paper_tu_sigma_homo_hetero.md                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_sigma_ungated.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`44713424`** |
| **Submit** | `bash_interface/cluster/submit_tu_sigma_ungated.sh` |
| **Tasks** | `1-120%20` · mem 128GB |
| **Variant** | SiGMA hetero ungated (`gnn.hybrid.gate none`) |
| **W&B** | `tu_hh_<ds>_SiGMA_ungated_{lr001,lr01}` · `tu_1x_<ds>_SiGMA_ungated_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_{homo_hetero,1x_gcn}/` |
| **Docs** | [`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) |
| **Logs** | `logs_gnnplus/tu_sigma_ung_44713424_<TASK>.log` |
| **Verify gated** | `python scripts/api_wanndb_query/verify_tu_sigma_tables.py` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 44748984 (smoke) + 44748985 (full 1-60%20)       ║
║  🎯  TU gate-clamp eval · Tab.17+18 reported LRs · 2026-09-06            ║
║  📄  Paper_tu_sigma_homo_hetero.md                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_gate_clamp.sh
```

| Field | Value |
|-------|-------|
| **SLURM smoke** | ✅ **`44748984`** (task 1) |
| **SLURM full** | ✅ **`44748985`** (`1-60%20`) |
| **Submit** | `bash_interface/cluster/submit_tu_gate_clamp.sh` |
| **Tasks** | `1-60%20` · mem 64GB · 4h |
| **Ckpts** | `$GNNPLUS_OUT_DIR/tu_sigma_{homo_hetero,1x_gcn}/` (gated hetero) |
| **Out** | `results/gate_clamp/t{17,18}_*.csv` |
| **Logs** | `logs_gnnplus/tu_gate_clamp_44748985_<TASK>.log` |
| **Docs** | [`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 44750758 (smoke) + 44750759 (full 1-240%20)      ║
║  🎯  TU extreme d_h ∈ {1,2} gated vs ungated · 2026-09-06                ║
║  📄  Paper_tu_sigma_dh_extreme.md                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_sigma_dh_extreme.sh
```

| Field | Value |
|-------|-------|
| **SLURM smoke** | ✅ **`44750758`** (tasks `1,11`) |
| **SLURM full** | ✅ **`44750759`** (`1-240%20`) |
| **Submit** | `bash_interface/cluster/submit_tu_sigma_dh_extreme.sh` |
| **Tasks** | `1-240%20` · mem 128GB · 96h |
| **Configs** | `configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-dh{1,2}-anchor.yaml` |
| **W&B** | `tu_dh{1,2}_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_dh_extreme/` |
| **Logs** | `logs_gnnplus/tu_dh_ext_44750759_<TASK>.log` |
| **Docs** | [`Paper_tu_sigma_dh_extreme.md`](Paper_tu_sigma_dh_extreme.md) |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  ENZYMES 44897907 (801-1600) · ✅ MUTAG 44876760 done                ║
║  🎯  TU Tab.17/18 L×d_h×H gated vs ungated                               ║
║  📄  Paper_tu_sigma_depth_dh_h.md                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
# MUTAG done: 44876760 · 800/800
# ENZYMES:    44897907 · 801-1600
TU_LDHH_ARRAY=1601-2400 bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh  # PROTEINS
# TU_LDHH_ARRAY=2401-3200 bash ...  # COLLAB
# TU_LDHH_ARRAY=3201-4000 bash ...  # IMDB
# TU_LDHH_ARRAY=4001-4800 bash ...  # REDDIT
```

| Field | Value |
|-------|-------|
| **Smoke** | ✅ **`44876758`** · `1,11` · `%2` |
| **MUTAG** | ✅ **`44876760`** · `1-800` · **800/800 COMPLETED** |
| **ENZYMES** | ✅ **`44897907`** · `801-1600` · running |
| **Rest** | 🛑 PROTEINS…REDDIT not yet |
| **Submit** | `bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh` |
| **Tasks** | **4800** · dataset blocks of 800 · mem 128GB · 96h |
| **Grid** | L∈{1,2,4,8,16} · d_h∈{1,2,4,16} · H∈{64,8} · gated/ungated · 2 LR · 5 seeds |
| **Note** | Paper Tab.17/18 used **L=12** (not in this grid) |
| **W&B** | `tu_L<k>_dh<m>_H<h>_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/` |
| **Docs** | [`Paper_tu_sigma_depth_dh_h.md`](Paper_tu_sigma_depth_dh_h.md) |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  H∈{4,2}  ·  smoke 44899287 · MUTAG 44899291 (1-800)                 ║
║  🎯  narrow-trunk fill of L×d_h×H gated vs ungated                       ║
║  📄  Paper_tu_sigma_depth_dh_h.md                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
# already submitted:
# smoke  44899287  ·  1,11%2   (H=4)
# MUTAG  44899291  ·  1-800%20 (H∈{4,2})
# next after MUTAG:
TU_LDHH_HS="4 2" TU_LDHH_ARRAY=801-1600 \
  bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh  # ENZYMES
```

| Field | Value |
|-------|-------|
| **Smoke** | ✅ **`44899287`** · `1,11` · `%2` |
| **MUTAG** | ✅ **`44899291`** · `1-800` · `%20` |
| **H** | `{4, 2}` via `TU_LDHH_HS="4 2"` |
| **Tasks** | +2400 · same 800/ds blocks |
| **W&B** | `tu_L*_dh*_H{4,2}_*` |
| **Plots** | `fig_mutag_delta_heatmap_H{4,2}.png` after MUTAG |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  H∈{8,4,2} gpu_h200 · ENZYMES→REDDIT + PROTEINS                      ║
║  🎯  delta heatmaps · MUTAG skipped (existing)                           ║
║  📄  Paper_tu_sigma_depth_dh_h.md                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
squeue -u $USER -j 44930340,44930486,44930487,44930489,44930491 \
  -o "%.18i %.9P %.2t %.10M %.20R"
```

| Field | Value |
|-------|-------|
| **ENZYMES** | ✅ **`44930486`** · `1201-2400` · `gpu_h200` |
| **PROTEINS** | ✅ **`44930340`** · `2401-3600` · `gpu_h200` |
| **COLLAB** | ✅ **`44930491`** · `3601-4800` · `gpu_h200` |
| **IMDB** | ✅ **`44930487`** · `4801-6000` · `gpu_h200` |
| **REDDIT** | ✅ **`44930489`** · `6001-7200` · `gpu_h200` |
| **H** | `{8, 4, 2}` · 72h · Nice=0 · `%20` |
| **W&B** | `tu_L*_dh*_H{8,4,2}_*` |
| **Plots** | `fig_<ds>_delta_heatmap_H{8,4,2}.png` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  gate dump · smoke 44904350 · full 44904351 (1-20)                   ║
║  🎯  MUTAG L=1 H=8 best-LR gated → gate_values_per_graph.pt              ║
║  📄  Paper_tu_sigma_depth_dh_h.md                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
# already submitted (full before smoke OK — task 1 may duplicate):
# smoke  44904350  ·  1%1
# full   44904351  ·  1-20%10
squeue -u $USER -j 44904350,44904351
# after DONE:
# ls $GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/mutag_L1_dh*_H8_gated_*/gate_values_per_graph.pt | wc -l
```

| Field | Value |
|-------|-------|
| **Smoke** | ✅ **`44904350`** · task `1` |
| **Full** | ✅ **`44904351`** · `1-20` · `%10` |
| **Submit** | `bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/mutag_L1_*_H8_gated_*/gate_values_per_graph.pt` |
| **Logs** | `logs_gnnplus/tu_LdhH_gdmp_<JOBID>_<TASK>.log` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  RUNNING  ·  smoke 44752943 · PATTERN 44752944                       ║
║  🎯  Tab.3/4 progressive d_h × gated vs ungated (PATTERN first)          ║
║  📄  Paper_sigma_dh_matched_ungated.md                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
# already submitted:
# smoke  44752943  ·  81,91%2
# PATTERN 44752944 ·  1-100%20
# next:
SIGMA_DH_PROG_ARRAY=101-200 bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh  # CLUSTER
SIGMA_DH_PROG_ARRAY=201-300 bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh  # MNIST
SIGMA_DH_PROG_ARRAY=301-400 bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh  # Pep-func
```

| Field | Value |
|-------|-------|
| **Smoke** | ✅ **`44752943`** · `81,91` · `%2` |
| **PATTERN** | ✅ **`44752944`** · `1-100` · `%20` |
| **Rest** | 🛑 CLUSTER / MNIST / Pep-func not yet |
| **Submit** | `bash_interface/cluster/submit_sigma_dh_prog_ungated.sh` |
| **Ladders** | PATTERN 16→1 · CLUSTER 24→1 · MNIST 37→1 · Pep-func 23→1 |
| **W&B** | `paper_sigma_dh_prog_<fam>_dh<k>_{gated,ungated}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_dh_prog/` |
| **Docs** | [`Paper_sigma_dh_matched_ungated.md`](Paper_sigma_dh_matched_ungated.md) |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37600400  ·  1-70%20                            ║
║  🎯  SiGMA baby/tiny budget fills (≤500k / 1M / 2M)                      ║
║  📄  Paper_sigma_budget.md                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_sigma_budget.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37600400`** |
| **Tasks** | `1-70%20` · mem 128GB |
| **Docs** | [`Paper_sigma_budget.md`](Paper_sigma_budget.md) |
| **Logs** | `logs_gnnplus/sigma_budget_37600400_<TASK>.log` |
| **W&B** | `paper_budget_<ds>_<b500k\|b1m\|b2m>` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37727415  ·  1-15%15                            ║
║  🎯  CIFAR10 budget re-fit (params actually under cap)                   ║
║  📄  Paper_sigma_budget.md                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_cifar_budget_fit.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37727415`** |
| **Submit** | `bash_interface/cluster/submit_cifar_budget_fit.sh` |
| **Tasks** | `1-15%15` |
| **Fits** | ≤500k ~498.8k · ≤1M ~998.9k · ≤2M ~1.998M |
| **W&B** | `paper_budget_cifar10_b{500k,1m,2m}_fit` |
| **Docs** | [`Paper_sigma_budget.md`](Paper_sigma_budget.md) |
| **Logs** | `logs_gnnplus/cifar_budget_fit_37727415_<TASK>.log` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37732478  ·  1-35%20                            ║
║  🎯  SiGMA ∼100k budget row (7 ds × 5 seeds)                             ║
║  📄  Paper_sigma_budget.md                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_sigma_budget_100k.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37732478`** |
| **Submit** | `bash_interface/cluster/submit_sigma_budget_100k.sh` |
| **Tasks** | `1-35%20` · mem 128GB |
| **W&B** | `paper_budget_<ds>_b100k` |
| **Docs** | [`Paper_sigma_budget.md`](Paper_sigma_budget.md) |
| **Logs** | `logs_gnnplus/sigma_b100k_37732478_<TASK>.log` |

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SiGMA d_h-matched Tab. 3/4 — 3 tiers (2026-08-24)    ║
║  🎯  fast / slow (CIFAR+VOC) / coco · LR ∈ {1e-3,1e-2} · 5 seeds        ║
║  📄  Paper_sigma_dh_matched.md                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

```bash
# ✅ already submitted — do not re-run unless re-launching
# SIGMA_DH_MATCHED_PARALLEL=10 SIGMA_DH_MATCHED_PARTITION=mweber_gpu \
#   bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh
# SIGMA_DH_MATCHED_PARALLEL=5 SIGMA_DH_MATCHED_PARTITION=mweber_gpu \
#   bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh
# SIGMA_DH_MATCHED_PARALLEL=2 SIGMA_DH_MATCHED_PARTITION=mweber_gpu \
#   SIGMA_DH_MATCHED_TIME=14-00:00:00 \
#   bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh
```

| Tier | SLURM | Array | Parallel | Time | Contents |
|------|------:|-------|---------:|------|----------|
| **fast** | ✅ **`41709078`** (1–50) · rerun **`42412053`** (**46/50**; 4 MalNet fail) | `1-100%10` / `51-100%20` | 10 / 20 | 48h | PATTERN–MNIST ✅; Pep-* ✅; MalNet 4 seeds left |
| **slow** | ✅ **`41709082`** (**5/40**) | `1-40%5` | 5 | 120h | CIFAR `dh20` lr001 done; lr01 running |
| **coco** | ✅ **`41709085`** (2 run · 2 TIMEOUT) | `1-10%2` | 2 | **2d** (tasks 3–4) | COCO; resubmit tasks 1–2 |
| **Partition** | `mweber_gpu` (skipped `gpu_h200` Priority backlog) | | | | |
| **LRs** | `{0.001, 0.01}` × 5 seeds — pick better per family | | | | |
| **Worker** | `run_sigma_dh_matched.sh` (`SIGMA_DH_MATCHED_TIER=…`) | | | | |
| **Docs** | [`Paper_sigma_dh_matched.md`](Paper_sigma_dh_matched.md) · [`rebuttal.md`](rebuttal.md) | | | | |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_dh_matched/<fam>_<lr>_seed<s>/` | | | | |
| **Logs** | `logs_gnnplus/sigma_dh_{fast,slow,coco}_<JOB>_<TASK>.log` | | | | |
| **Fairshare** | Aug 23 EOD `mweber_lab: 0.749064` | | | | |
| **Skip** | ZINC (main 450k already ≤500k) | | | | |

**2026-08-28:** Fast tier **96/100** Slurm-complete. Pep-func `dh23` AP **0.7002±0.0084** vs paper 0.7080.
Resubmit: `SIGMA_DH_MATCHED_ARRAY=91,94,96,99` (MalNet); `SIGMA_DH_MATCHED_ARRAY=1,2` (COCO timeout).

Monitor: `squeue -u $USER -n sigma_dh_fast,sigma_dh_slow,sigma_dh_coco`

```bash
sacct -j 42412053,41709082,41709085 -X --format=JobID,State,ExitCode -n
```

---

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  RUNNING  ·  TU Errica SiGMA (fixed8 fill + full64 FAILED rerun)     ║
║  🎯  fixed8 fill 44507757 · full64 rerun 44509970 (2479, H200)          ║
║  📄  Paper_tu_errica_fair_comparison.md                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Phase | JOBID | Tasks | Status (2026-09-04) |
|-------|-------|-------|------------------------|
| `grid_select` GIN/SAGE/GCN/GAT | 42750648 / 43116245 / 43434937 / 43434950+44099901 | — | ✅ done |
| `grid_eval` GIN/SAGE/GCN/GAT | 44100531 / 66 / 96 / **44165919** | 210×4 | ✅ done |
| `sigma_grid_select` budget_bio | **43741550** | 400 | obsolete |
| `sigma_grid_eval` budget_bio | **44165958** | 210 | ignore / cancel |
| **`sigma_grid_select_fixed8`** | **44217420** + **44266489** | **560** | ✅ 559/560 · fill → **44507757** |
| **`sigma_grid_select_full64`** | **44262912** | **4,480** | ⚠️ half FAILED (quota/`0:53`) |
| **fixed8 REDDIT jump** | **44266489** | 80 (401–480) | ✅ done |
| **full64 REDDIT jump** | **44266493** | 640 (3201–3840) | ⚠️ mostly FAILED |
| **full64 REDDIT only (no cancel)** | **44750843** | 640 (3201–3840) | ❌ scancel · REDDIT via **44509970** |
| **full64 FAILED rerun** | **44509970** | 2479 | 🔄 ~464 COMP · 20 run · 12 FAIL |
| **fixed8 COLLAB fill** | **44507757** | 1 (task 560) | ✅ COMPLETED |
| **`sigma_grid_eval_fixed8`** | **44621846** | 210 | ✅ 209 + fill **44748166** |
| **eval COLLAB f0 s0 fill** | **44748166** | 1 (task 181) | ✅ COMPLETED |
| **`anchor_boost` select** | **44840486** | 480 | 🔄 REDDIT left · PROTEINS ✅ |
| **`anchor_boost` eval PROTEINS** | **44938699** | 30 (1–30) | ✅ **73.68±3.08** |
| **`a1g2_micro` select** | **45054117** | 80 | 🔄 REDDIT 2 R left (`_42`,`_50`) · PROTEINS ✅ |
| **`a1g2_micro` eval PROTEINS** | **45067214** | 30 (1–30) | ✅ **72.2±2.9** |
| **`a1g2_nci1_micro` select** | **45054174** | 40 | ✅ · agg → `sigma_a1g2_nci1_micro_per_fold.json` (10 folds) |
| **`a1g2_nci1_micro` eval** | **45074636** | 30 (1–30) | ✅ **80.4±2.0** |
| **`fixed8_ungated` select** | **44869251** | 560 | ⏸️ **HELD** · `scontrol release 44869251` when ready |
| `aggregate_sigma` → `sigma_grid_eval_fixed8` | — | 210 | ✅ selection done · eval **44621846** |

```bash
# After 44507757 COMPLETED:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma
#   # commit+push sigma_fixed8_per_fold.json, then:
#   bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_fixed8
```

| Field | Value |
|-------|-------|
| **Parallel** | 12 default · SiGMA select 128GB / 96h · `mweber_gpu` |
| **Submit** | `bash_interface/cluster/submit_tu_errica_fair.sh` |
| **full64 FAILED rerun** | `bash_interface/cluster/submit_tu_errica_full64_rerun_failed.sh` |
| **full64 REDDIT only** | `bash_interface/cluster/submit_tu_errica_full64_reddit_only.sh` |
| **Orchestrator** | `bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh` |
| **Worker** | `bash_interface/cluster/run_tu_errica_fair.sh` |
| **Logs** | prefer netscratch (`$GNNPLUS_OUT_DIR/logs_tu_errica_*`); holylabs `logs_gnnplus` hit quota |
| **W&B groups** | `tu_errica_<ds>_SiGMA_hetero_sigma_grid_select_fixed8_f<fold>_hp<id>` |
| **Next** | after **44507757**: aggregate fixed8 → `sigma_grid_eval_fixed8`; full64 via **44509970** |

Monitor:

```bash
for j in 44099901 44100531 44100566 44100596 43741550; do
  echo "=== $j ==="
  sacct -j $j -X --format=State,ExitCode -n | awk '{print $1}' | sort | uniq -c
done
squeue -u $USER | grep tu_errica
```

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  DONE (exploratory)  ·  TU Errica canonical (fixed GIN_CANONICAL)  ║
║  📄  Paper_tu_errica_fair_comparison.md — do not cite as final table     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`42673425`** (630) · SiGMA OOM rerun **`42746310`** (60) |
| **Progress** | 570/630 done · 60 OOM (DD/REDDIT-B bs128) |
| **Out** | `$GNNPLUS_OUT_DIR/tu_errica/canonical/` |

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
| Table 6 SiGMA_attn_gate (LRGB) | ✅ | `35354579` |
| Table 6 SiGMA_ungated_attn (7 ds) | ✅ | `36605829` |
| Table 5 COCO gaps relaunch | ✅ Attn `34070241`; MP/ungated ✅ `34081524` (prior ❌ `34070242`/`43`) | `34081524` |
| Table 5+6 COCO H200 twin | ✅ T5 `34098505` · T6 `34098527` | `34098505` / `34098527` |
| Table 5+6 COCO ep150 twin | ✅ T5 `34682558` · T6 `34682560` | `34682558` / `34682560` |
| COCO Table6 Attn a3 + MP a0g3 | ✅ | `34869787` |
| COCO dead-seed retry (Homog/Hetero + ep150 SiGMA) | ✅ `35247208`/`35247209` (+ T6 main JOBID) | `35247208` / `35247209` |
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
| Hetero MUTAG/ENZYMES Xu et al. HPs | 🛑 TO RUN (6-job) | — |
| Hetero full TU Xu HPs (6×4 + SAGE) | ✅ | `36604947` |
| TU SiGMA gate-viz (Xu a2g2) | ✅ | `36604951` |
| ENZYMES SiGMA a8g8 hetero | ✅ | `34875028` |
| TU all-layer activations | ✅ | `34869795` |
| Table 6 1-MP peptides | ✅ | `32717625` |
| Table 6 COCO relaunch | ✅ mweber `34070245`; H200 twin `34098527` | `34070245` / `34098527` |
| ENZYMES ogpkubk9 sweep | 🛑 TO RUN | — |
| TU GCN vs SiGMA homo vs hetero | ✅ | `37434534` |
| SiGMA d_h-matched Tab. 3/4 (3 tiers, 2 LRs) | 🔄 fast **96/100** · slow **5/40** · coco 2 run | `41709078` / `42412053` / `41709082` / `41709085` |
| TU Errica-fair canonical (630, exploratory) | ✅ 570/630 · 60 OOM | `42673425` · SiGMA rerun `42746310` |
| TU Errica hybrid | 🔄 fixed8 eval **44621846** · full64 rerun **44509970** | see `Paper_tu_errica_fair_comparison.md` |
| TU gate–operator bridge (MUTAG+ENZYMES hetero) | 🟢 6/8 ✅ · GatedGCN retry **`44218244`** | `44218244` |
| GCN/GIN routing synthetic (toy + sigma) | 🛑 TO RUN | — |
