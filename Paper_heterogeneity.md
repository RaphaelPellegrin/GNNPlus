# Heterogeneity profiles in GNNPlus (ported from Heterogeneity_Profile)

Graph-level heterogeneity analysis for **MUTAG**, **ENZYMES**, **PROTEINS** with **GCN**, **GIN**, and **SiGMA** (`hybrid_gnn`).

---

## Dataset support (native?)

| Dataset | Loader | Configs before this port | Notes |
|---------|--------|--------------------------|-------|
| **MUTAG** | `PyG-TUDataset` in `master_loader.py` | `configs/gcn/mutag.yaml`, `configs/gated_hybrid/mutag.yaml`, … | Native |
| **ENZYMES** | same | `configs/gcn/enzymes.yaml`, `configs/gated_hybrid/enzymes.yaml` | Native |
| **PROTEINS** | same (`PROTEINS` in allow-list) | **no yaml before** → added under `configs/heterogeneity/` | Native loader; new hetero configs only |

No loader changes needed. Added: plain **`gin`** `custom_gnn` layer (`GNNPlus/layer/gin_conv_layer.py`) for GIN baselines (previously only `gine`).

Existing TU configs use `split_mode: cv-stratifiedkfold-10`. Heterogeneity configs use **`split_mode: random`** with **`split: [0.5, 0.25, 0.25]`**.

---

## Protocol

Matches paper §3 / Heterogeneity_Profile:

1. Random 50/25/25 train/val/test each trial  
2. Train **300** epochs; keep **validation-best** weights  
3. Record per-graph test correctness (0/1)  
4. Repeat until every graph has been in the test set **≥ 100** times  
5. Plot average accuracy per graph → heterogeneity profile  

**Bug fix (2026-07-23):** Binary datasets (MUTAG, PROTEINS) use GraphGym
`dim_out=1` + BCE. The runner was scoring accuracy on **sigmoid probabilities**
with threshold `> 0` (always class 1), so val/test metrics collapsed to the
class prior and GCN/GIN/SiGMA looked identical. **ENZYMES (6-way) was fine.**
Fixed: score **raw logits**. Re-run MUTAG + PROTEINS after pulling the fix;
cancel leftover `34410913` task 9 if still running.

Implementation:

| Piece | Path |
|-------|------|
| Runner | `scripts/heterogeneity/run_heterogeneity_profiles.py` |
| Plot / pickle utils | `GNNPlus/experiments/track_avg_accuracy.py` |
| Configs | `configs/heterogeneity/{mutag,enzymes,proteins}-{gcn,gin,sigma}.yaml` |
| Cluster | `bash_interface/cluster/submit_heterogeneity_tu.sh` |

---

## Relaunch — Xu et al. ICLR 2019 hyperparams (arXiv:1810.00826)

Official code: [`weihua916/powerful-gnns`](https://github.com/weihua916/powerful-gnns).

**Recipe we freeze** (paper §7 + repo defaults; fixed train setting for fair model compare):

| Knob | Value |
|------|--------|
| Depth | 5 GNN layers incl. input → `layers_mp: 4` |
| Hidden | 64 |
| Batch | 128 |
| Dropout (final/head) | 0.5 |
| Optimizer | Adam `lr=0.01`, `weight_decay=0` |
| LR schedule | ×0.5 every 50 epochs (`scheduler: step`, milestones 50…300) |
| Epochs | 350 |
| Graph pooling | **sum** (`add`) on bio |
| GIN | sum neigh. agg, no residual/FFN (GIN-0 style) |
| GCN | mean neigh. agg |
| SiGMA | a2g2 hybrid under the **same** optim/train recipe |

**Note:** ENZYMES is **not** in Xu et al. (they use MUTAG/PTC/NCI1/PROTEINS); we apply the same bioinformatics recipe. Paper JK-sums layer-wise scores; our stack uses final-layer sum pooling.

Configs: `configs/heterogeneity/powerful_gnns/{mutag,enzymes}-{gcn,gin,sigma}.yaml`  
W&B: `building_hetero_profile_<ds>_powerful_gnns` / `<ds>_<model>_powerful_gnns`

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_heterogeneity_powerful_gnns.sh
# 👉 paste JOBID
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **Tasks** | `1-6` (mutag×3 then enzymes×3) |
| **Scripts** | `submit_heterogeneity_powerful_gnns.sh` → `run_heterogeneity_powerful_gnns.sh` |

---

## Full TU grid (Xu HPs) — GCN / GIN / SAGE / SiGMA · ≥100 test apps

Loader-supported [TUDataset](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.TUDataset.html)
names only (not every PyG TU set): **MUTAG, ENZYMES, PROTEINS, DD, NCI1, TRIANGLES**.

| Model | Architecture | Hyperparams |
|-------|--------------|-------------|
| **GCN** | `custom_gnn` · `layer_type: gcn` | Xu et al. (L4 / H64 / batch 128 / Adam 0.01 / step / 350 ep / sum pool) |
| **GIN** | `custom_gnn` · `layer_type: gin` · sum agg | same |
| **SAGE** | `custom_gnn` · `layer_type: sage` · mean agg | **same Xu schedule** (no dedicated TU graph-class paper; fair width/depth/optim match) |
| **SiGMA** | `hybrid_gnn` **a2g2** · `gnn_types: GIN,GIN` + 2 attn | same optim; GIN = Xu-best bio MP + attention |

Configs: `configs/heterogeneity/powerful_gnns/{mutag,enzymes,proteins,dd,nci1,triangles}-{gcn,gin,sage,sigma}.yaml`  
(24 YAMLs; SiGMA overwrites prior `GCN,GIN` sigma to **`GIN,GIN`**.)

**Also:** dedicated SiGMA **gate-viz** (ckpt every 50 ep) + dump → per-graph γ for plots.

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 1) Hetero profiles (24 jobs, ≥100 test appearances each)
bash bash_interface/cluster/submit_heterogeneity_tu_powerful_full.sh

# 2) SiGMA ckpt train for gate dumps (6 jobs)
bash bash_interface/cluster/submit_tu_sigma_gate_viz.sh

# 3) after (2) finishes — dump gate_values_per_graph.pt
bash bash_interface/cluster/submit_dump_tu_sigma_gates.sh
```

| Field | Value |
|-------|-------|
| **SLURM hetero** | ✅ **`36604947`** · `1-24%8` · mweber_gpu |
| **SLURM gate-viz** | ✅ **`36604951`** · `1-6%6` · seed 2 · ckpt period 50 |
| **Tasks hetero** | `1-24%8` |
| **Tasks gate** | `1-6%6` |
| **Scripts** | `submit_heterogeneity_tu_powerful_full.sh`, `submit_tu_sigma_gate_viz.sh`, `submit_dump_tu_sigma_gates.sh` |
| **Logs** | `logs_gnnplus/hetero_tu_full_36604947_<TASK>.log` · `tu_sigma_gate_36604951_<TASK>.log` |

---

## Launch (prior TU grid)

```text
╔══════════════════════════════════════════════════════════════════╗
║  ✅  RUNNING  ·  SLURM 34869869  ·  mweber_gpu  ·  2026-07-24    ║
║  📈 proteins_sigma only (task 9) · ≥100 appearances               ║
║  Prior full 9-job: 34410913 (MUTAG/PROTEINS fixed; ENZYMES OK)   ║
╚══════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🧪 smoke first (recommended!)
HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=20 \
  bash bash_interface/cluster/submit_heterogeneity_tu.sh

# 🚀 full paper protocol (submitted as 33811552)
HETERO_TIME=72:00:00 HETERO_PARALLEL=5 HETERO_PARTITION=gpu_h200 \
  bash bash_interface/cluster/submit_heterogeneity_tu.sh
```

Local smoke:

```bash
python scripts/heterogeneity/run_heterogeneity_profiles.py \
  --cfg configs/heterogeneity/mutag-gcn.yaml \
  --required_test_appearances 2 --max_trials 20 \
  --no-wandb \
  optim.max_epoch 5
```

| Field | Value |
|-------|-------|
| **SLURM array** | ✅ **`34869869`** task `9` (`proteins_sigma` relaunch, 2026-07-24). Prior full grid ✅ **`34410913`**. Priors ❌ `34409940` / `34073629`. |
| **Partition** | `mweber_gpu` · time `192:00:00` |
| **Tasks** | `9` = proteins × SiGMA |
| **Parallel** | ≤**1** GPU |
| **Logs** | `logs_gnnplus/hetero_tu_34869869_<TASK>.log` |
| **Local outs** | `results/heterogeneity/<dataset>_<MODEL>/` (pickle + appearances CSV + PNGs) |
| **W&B** | see below |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## W&B logging

Each (dataset, model) job is one W&B run.

| | |
|--|--|
| **Entity / project** | `weber-geoml-harvard-university` / `GNNPlus` |
| **Group** | `building_hetero_profile_<dataset>` (e.g. `building_hetero_profile_mutag`) |
| **Run name** | `<dataset>_<model>` (e.g. `mutag_gcn`) |
| **During run** | `hetero/min_appearances`, mean/max, trial val/test |
| **At job end** | profile PNGs + artifact |

**When is the hetero profile built?** At the **end of each job** (once every graph has ≥N test appearances). You do **not** need all 9 jobs finished to get a profile — each of the 9 produces its own pickle/CSV/PNGs and uploads them.

**Artifacts** (`type=heterogeneity_profile`, name `hetero_profile_<ds>_<model>`):

1. `*_graph_dict.pickle` — full per-graph 0/1 history + `test_appearances`
2. `*_test_appearances.csv` — `graph_idx, n_test_appearances, n_correct, avg_accuracy`
3. Profile PNGs (`*_by_index.png`, `*_by_accuracy.png`)

### Local figures + interactive HTML (by average accuracy)

Pulled W&B artifacts live under `results/heterogeneity/` (8/9 cells as of 2026-07-24;
`proteins_sigma` crashed at ~85/100 appearances — relaunch needed).

```bash
cd /Users/pellegrinraphael/Desktop/Academic_Research/Repos_GNN/GNNPlus

# regenerate by-accuracy PNGs + interactive HTML (toggle datasets & models)
python scripts/heterogeneity/build_heterogeneity_html.py

# open
open results/heterogeneity/heterogeneity_profiles.html

# MUTAG only, points colored by graph class (graph_idx ↔ TUDataset / TMD CSV)
python scripts/heterogeneity/build_heterogeneity_html.py \
  --datasets mutag --color_by_class \
  --output results/heterogeneity/heterogeneity_profiles_mutag_by_class.html
open results/heterogeneity/heterogeneity_profiles_mutag_by_class.html
```

- PNGs (x = rank hard→easy): `results/heterogeneity/paper_figures_by_accuracy/`
- HTML: `results/heterogeneity/heterogeneity_profiles.html`
- MUTAG by class: `results/heterogeneity/heterogeneity_profiles_mutag_by_class.html`  
  (class 0 is harder than class 1 across GCN / GIN / SiGMA; hover shows `graph_idx`)
- Paper PNGs (class-colored hetero profiles + model mean±std):
  ```bash
  python scripts/heterogeneity/plot_mutag_class_profiles.py --dataset mutag
  # → .../mutag_GCN_GIN_SiGMA_by_class.png
  python scripts/heterogeneity/plot_mutag_class_profiles.py --dataset enzymes
  # → .../enzymes_GCN_GIN_SiGMA_by_class.png  (3 panels)
  # → .../enzymes_GCN_GIN_SiGMA_by_class_overlay.png  (18-color overlay)
  python scripts/heterogeneity/build_heterogeneity_html.py \
    --datasets enzymes --color_by_class --skip_png \
    --output results/heterogeneity/heterogeneity_profiles_enzymes_by_class.html
  ```

When `proteins_sigma` finishes, pull its artifact then re-run the builder.

### ENZYMES SiGMA large model (a8g8 L12 — match MOE_6/`7dsqq7z2`)

Default grid task 6 uses small `enzymes-sigma.yaml` (a2g2 L4). For the
Heterogeneity_Profile–scale SiGMA (~0.58 test Acc), launch separately:

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_heterogeneity_enzymes_sigma_a8g8.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`34875028`** (2026-07-24, `mweber_gpu`) |
| **Config** | `configs/heterogeneity/enzymes-sigma-a8g8.yaml` |
| **W&B** | `building_hetero_profile_enzymes` / `enzymes_sigma_a8g8` |
| **Source** | [MOE_6/7dsqq7z2](https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/7dsqq7z2) |

Disable W&B: `HETERO_WANDB=0 bash …/submit_heterogeneity_tu.sh`

---

## Task map

| Task | Dataset | Model |
|------|---------|-------|
| 1 | mutag | GCN |
| 2 | mutag | GIN |
| 3 | mutag | SiGMA |
| 4 | enzymes | GCN |
| 5 | enzymes | GIN |
| 6 | enzymes | SiGMA |
| 7 | proteins | GCN |
| 8 | proteins | GIN |
| 9 | proteins | SiGMA |
