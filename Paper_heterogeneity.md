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

Implementation:

| Piece | Path |
|-------|------|
| Runner | `scripts/heterogeneity/run_heterogeneity_profiles.py` |
| Plot / pickle utils | `GNNPlus/experiments/track_avg_accuracy.py` |
| Configs | `configs/heterogeneity/{mutag,enzymes,proteins}-{gcn,gin,sigma}.yaml` |
| Cluster | `bash_interface/cluster/submit_heterogeneity_tu.sh` |

---

## Launch

```text
╔══════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 33811552  ·  gpu_h200  ·  2026-07-21    ║
║  📈 Heterogeneity profiles · MUTAG/ENZYMES/PROTEINS × GCN/GIN/SiGMA ║
║  🔁 ≥100 test appearances · 9 jobs · ≤5 GPUs · 72h               ║
║  📒 also listed in CLUSTER_LAUNCHES.md                           ║
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
| **SLURM array** | ✅ **`34073629`** (2026-07-21; `indices()` fix). Priors: `34070246` crash, `33811552` quota fake-finish. **Note:** SiGMA configs also needed `edge_encoder: False` (same `LinearEdge`/`times_func` bug as ENZYMES ogpkubk9) — `git pull` before sigma array tasks start, or relaunch failed sigma tasks. |
| **Partition** | `mweber_gpu` · time `192:00:00` |
| **Tasks** | `1-9%3` = 3 datasets × 3 models |
| **Parallel** | ≤**3** GPUs |
| **Logs** | `logs_gnnplus/hetero_tu_34073629_<TASK>.log` |
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
