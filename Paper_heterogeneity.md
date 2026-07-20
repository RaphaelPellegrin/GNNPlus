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
║  🛑🛑🛑  TO RUN  ·  cluster was FULL — submit when free  🛑🛑🛑  ║
║  📈 Heterogeneity profiles · MUTAG/ENZYMES/PROTEINS × GCN/GIN/SiGMA ║
║  🔁 ≥100 test appearances · 9 long jobs                          ║
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

# 🚀🚀🚀 full paper protocol (9 jobs, ≤9 GPUs) — after smoke OK
bash bash_interface/cluster/submit_heterogeneity_tu.sh
# 👉 paste JOBID below + into CLUSTER_LAUNCHES.md
```

Local smoke:

```bash
python scripts/heterogeneity/run_heterogeneity_profiles.py \
  --cfg configs/heterogeneity/mutag-gcn.yaml \
  --required_test_appearances 2 --max_trials 20 \
  optim.max_epoch 5
```

| Field | Value |
|-------|-------|
| **SLURM array** | 🛑 *TO RUN — not submitted* |
| **Tasks** | `1-9` = 3 datasets × 3 models |
| **Outputs** | `results/heterogeneity/<dataset>_<MODEL>/` (pickle + PNGs) |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

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
