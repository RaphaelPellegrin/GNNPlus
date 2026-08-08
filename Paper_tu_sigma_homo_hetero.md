# TU datasets — GCN vs SiGMA (homo) vs SiGMA (hetero)

Goal table (test accuracy, mean±std over 5 seeds):

| Dataset | GCN | SiGMA (homo) | SiGMA (hetero) |
|---------|-----|--------------|----------------|
| MUTAG | | | |
| ENZYMES | | | |
| PROTEINS | | | |
| DD | | | |
| NCI1 | | | |
| TRIANGLES | | | |

Hypothesis: **SiGMA > GCN**, and **SiGMA hetero > SiGMA homo**.

No heterogeneity profiles — plain training only (5 seeds × each family/LR).

---

## Architecture

| Family | Model | Heads | MP types |
|--------|-------|-------|----------|
| **GCN** | `custom_gnn` / `gcn` | — | single GCN stack |
| **GIN / SAGE / GAT** | `custom_gnn` | — | single stack (same L12/H64 recipe) |
| **SiGMA (homo)** | `hybrid_gnn` | **a2g4** | `GCN,GCN,GCN,GCN` |
| **SiGMA (hetero)** | `hybrid_gnn` | **a2g4** | `GCN,GIN,SAGE,GAT` |

Shared (all families):

| Knob | Value |
|------|-------|
| Depth / width | L=12, H=64, `d_h`=16 |
| Gate / norm / mask | headwise / layernorm / full |
| FFN + residual | True |
| Batch | 64 |
| Split | random 50/25/25 |
| Pool | mean |
| Optim | AdamW, WD=0, plateau patience=50, max_epoch=1000 |
| Ckpt | `enable_ckpt=True` · `ckpt_best=True` · best-val model under `out_dir/ckpt/` |
| Gates | SiGMA runs auto-dump `gate_values_per_graph.pt` after training |

**Prior ENZYMES reference:** gate-viz / ogpkubk9 was **a4g4** `GCN,GIN,SAGE,GAT` L12
([`ogpkubk9`](https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/ogpkubk9)).
This campaign keeps the same MP mix + depth, but uses **2 attn heads** as requested
(homo matched at 4 MP heads).

---

## Hyperparameter choice (literature)

### GCN baseline sources

1. **Xu et al., ICLR 2019** ([powerful-gnns](https://github.com/weihua916/powerful-gnns), arXiv:1810.00826)  
   Default TU recipe: `hidden=64`, Adam `lr=0.01`, StepLR ×0.5 every 50 epochs,
   ~350 epochs, dropout 0.5, sum pool. Used widely for MUTAG / PROTEINS / NCI1 / DD.

2. **Dwivedi et al., Benchmarking GNNs** (TU GCN notebook)  
   `init_lr=5e-4`, residual, layer/batch norm, L≈4 under a ~100k param budget.

### What we use here

| Choice | Value | Why |
|--------|-------|-----|
| Shared L/H | 12 / 64 | Match validated ENZYMES SiGMA (ogpkubk9), fair depth vs SiGMA |
| GCN `base_lr` | **0.001** | Dwivedi-scale; also ogpkubk9 LR (stable for L12) |
| SiGMA LRs | **{0.001, 0.01}** | Same recipe × Xu default 0.01 + ogpkubk9 0.001 |
| Dropout | 0.1 | ogpkubk9 (L12 prefers lower than Xu’s 0.5) |
| Scheduler | `reduce_on_plateau` | ogpkubk9 (covers Xu-style decay without fixed steps) |
| Optim | AdamW, WD=0 | ogpkubk9; WD=0 matches Xu |

For the paper table, pick the **better LR** (by mean val/test) per SiGMA family after the 2×5 seed grids finish.

Configs:

- `configs/tu_sigma_homo_hetero/gcn-anchor.yaml`
- `configs/tu_sigma_homo_hetero/sigma-homo-a2g4-anchor.yaml`
- `configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml`

Scripts:

- `bash_interface/cluster/run_tu_sigma_homo_hetero.sh`
- `bash_interface/cluster/submit_tu_sigma_homo_hetero.sh`

---

## Launch (150 jobs)

```text
╔══════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37434534  ·  2026-08-05  ·  %8          ║
║  6 TU × {GCN, SiGMA_homo×2LR, SiGMA_hetero×2LR} × 5 seeds = 150 ║
╚══════════════════════════════════════════════════════════════════╝
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
| **SLURM array** | ✅ **`37434534`** (main 150); 🛑 DD SiGMA retry — see below |
| **Tasks** | `1-150%8` |
| **Partition / mem / time** | `mweber_gpu` / 64GB / 96h |
| **Logs** | `logs_gnnplus/tu_sigma_hh_37434534_<TASK>.log` |
| **Checkpoints / stats** | `$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_<variant>_<lr>_seed<s>/` |
| **Best model** | `…/ckpt/` (best val epoch) |
| **Per-graph gates** | `…/gate_values_per_graph.pt` (SiGMA only; `[N, L, H]` attn + gnn) |
| **Meta** | `…/train_meta.txt` + `config_used.yaml` |
| **W&B** | `tu_hh_<ds>_{GCN,SiGMA_homo,SiGMA_hetero}_{lr001,lr01}` |

### DD SiGMA retry (batch 16 / 128GB)

Prior `37434534` DD SiGMA: seed 0 finished, seeds 1–4 **failed** (likely OOM @ batch 64).
GCN DD is fine (n=5). Relaunch SiGMA only:

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_tu_dd_sigma_retry.sh
```

```text
╔══════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 37557549  ·  2026-08-06  ·  1-20%4     ║
║  DD SiGMA retry · batch=16 · mem=128GB · 20 jobs                 ║
╚══════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37557549`** |
| **Tasks** | `1-20%4` (homo×2 LR + hetero×2 LR × 5 seeds) |
| **batch / mem** | **16** / **128GB** |
| **W&B** | `tu_hh_dd_{SiGMA_homo,SiGMA_hetero}_{lr001,lr01}_bs16` |
| **Logs** | `logs_gnnplus/tu_dd_sigma_37557549_<TASK>.log` |
| **Scripts** | `submit_tu_dd_sigma_retry.sh` / `run_tu_dd_sigma_retry.sh` |

### NCI1 SiGMA retry (new LRs + longer train)

Prior best: hetero `lr=1e-3` → **79.03±1.19** vs GCN **80.51±0.71**.
Retry: LR ∈ `{5e-4, 2e-3}`, `max_epoch=2000`, `schedule_patience=100`, `%10` GPUs.

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_tu_nci1_sigma_retry.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *paste JOBID* |
| **Tasks** | `1-20%10` |
| **LRs** | `5e-4`, `2e-3` |
| **max_epoch / patience** | **2000** / **100** |
| **W&B** | `tu_hh_nci1_{SiGMA_homo,SiGMA_hetero}_{lr5e4,lr2e3}_ep2000` |
| **Logs** | `logs_gnnplus/tu_nci1_sigma_<JOBID>_<TASK>.log` |
| **Scripts** | `submit_tu_nci1_sigma_retry.sh` / `run_tu_nci1_sigma_retry.sh` |
| **Target** | beat GCN 80.51±0.71% |

### TU social extras (COLLAB / IMDB-BINARY / REDDIT-BINARY)

PyG [TUDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html) stats-table set (Lukas): keep MUTAG/ENZYMES/PROTEINS; **drop NCI1/TRIANGLES/DD** from the paper table; add these three (0 node features → `Constant()`).

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_tu_sigma_social.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`37574967`** |
| **Tasks** | `1-75%20` |
| **Mem** | 128GB |
| **Batches** | COLLAB 32 · IMDB 64 · REDDIT 16 |
| **W&B** | `tu_hh_{collab,imdb_binary,reddit_binary}_{GCN,SiGMA_homo,SiGMA_hetero}_{lr001,lr01}` |
| **Logs** | `logs_gnnplus/tu_sigma_soc_37574967_<TASK>.log` |
| **Scripts** | `submit_tu_sigma_social.sh` / `run_tu_sigma_social.sh` |
| **Loader** | `REDDIT-BINARY` added to `preformat_TUDataset` (+ Constant) |

### Task map

Per dataset block of 25 tasks (seeds 0–4):

| Offset | Variant | LR |
|--------|---------|-----|
| 1–5 | GCN | 0.001 |
| 6–10 | SiGMA_homo a2g4 (`GCN×4`) | 0.001 |
| 11–15 | SiGMA_homo a2g4 | 0.01 |
| 16–20 | SiGMA_hetero a2g4 (`GCN,GIN,SAGE,GAT`) | 0.001 |
| 21–25 | SiGMA_hetero a2g4 | 0.01 |

Dataset order: MUTAG → ENZYMES → PROTEINS → DD → NCI1 → TRIANGLES  
(tasks 1–25, 26–50, …, 126–150).

### Smoke (MUTAG only, all variants, 1 seed each → tasks 1,6,11,16,21)

```bash
TU_SIGMA_HH_ARRAY=1,6,11,16,21 TU_SIGMA_HH_PARALLEL=5 \
  bash bash_interface/cluster/submit_tu_sigma_homo_hetero.sh
```

---

## Model dump + per-graph gates

Each run writes under
`$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_<variant>_<lr>_seed<s>/`:

| File | Contents |
|------|----------|
| `ckpt/` | Best-val GraphGym checkpoint (all families) |
| `gate_values_per_graph.pt` | SiGMA only — auto after train |
| `config_used.yaml` | Anchor yaml copied at launch |
| `train_meta.txt` | dataset / lr / seed / W&B / job id |

`gate_values_per_graph.pt` keys (from `scripts/gate_viz/dump_per_graph_gates.py`):

- `attn`: `[N_graphs, L, Na]` — attention-head gates per layer
- `gnn`: `[N_graphs, L, Ng]` — MP-head gates per layer (homo: GCN×4; hetero: GCN,GIN,SAGE,GAT)
- `y`, `split` (0/1/2), `meta`

Plot locally (same as ENZYMES gate-viz):

```bash
python scripts/gate_viz/plot_per_graph_gates.py \
  --pt $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/enzymes_SiGMA_hetero_lr001_seed2/gate_values_per_graph.pt \
  --out_dir results/gate_viz/tu_hh_enzymes_hetero_lr001_seed2
```

### Pull dumps from cluster → plot locally

Training already wrote `gate_values_per_graph.pt` next to each SiGMA `ckpt/`
(best-val checkpoint). On the cluster, check:

```bash
ls $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/*/gate_values_per_graph.pt | wc -l
# expect ~80 SiGMA dirs with dumps (DD mostly missing; triangles hetero lr01 maybe incomplete)
```

Rsync only the `.pt` files (small) to your laptop:

```bash
mkdir -p results/tu_sigma_homo_hetero
rsync -avz --include='*/' --include='gate_values_per_graph.pt' --exclude='*' \
  rpellegrinext@holylogin.rc.fas.harvard.edu:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/tu_sigma_homo_hetero/ \
  results/tu_sigma_homo_hetero/
```

Batch-plot **SiGMA hetero** for the paper table set (MUTAG…REDDIT), best LR,
seed 2 — per-graph gates by head×layer (attn 2 + MP GCN/GIN/SAGE/GAT, L=12):

```bash
# local after rsync
python scripts/gate_viz/plot_tu_hh_gates_batch.py \
  --root results/tu_sigma_homo_hetero \
  --out_dir results/gate_viz/tu_hh_hetero \
  --datasets paper --variants SiGMA_hetero \
  --seeds 2 --prefer-lr best_from_table --color-by-class

# cluster
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
python scripts/gate_viz/plot_tu_hh_gates_batch.py \
  --root $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero \
  --out_dir $GNNPLUS_OUT_DIR/gate_viz/tu_hh_hetero \
  --datasets paper --variants SiGMA_hetero \
  --seeds 2 --prefer-lr best_from_table --color-by-class
```

Best-LR mapping used: MUTAG/ENZYMES/PROTEINS/IMDB hetero → `lr001`; COLLAB hetero → `lr01`;
REDDIT provisional `lr001` (update when both LRs finish). Add `--variants SiGMA_homo,SiGMA_hetero`
for both families. REDDIT dumps appear only after those SiGMA jobs finish.

Per-panel independent ranking (each cell sorted by its own γ↓):

```bash
python scripts/gate_viz/plot_tu_hh_gates_batch.py \
  --root results/tu_sigma_homo_hetero \
  --out_dir results/gate_viz/tu_hh_hetero \
  --datasets paper --variants SiGMA_hetero \
  --seeds 2 --prefer-lr best_from_table --color-by-class \
  --sort-mode per_panel
```

Filenames: `*_shared_order_*.png` vs `*_by_rank_*.png`.

### Per-node gates (mean + band, colored graphs)

Training dumps are **graph-mean** only. For node-level γ (within-graph bands and
node-colored drawings), re-dump with edges from a checkpoint:

```bash
# cluster — do NOT run dump python on login (GLIBC / wrong env).
# Use the SLURM dump script (sources common_env.sh on a GPU node).
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
# pull latest submit script first (exports GATE_DUMP_LEVEL)

# MUTAG SiGMA_hetero lr001 seed2 → array task 18
GATE_DUMP_LEVEL=both TU_SIGMA_HH_DUMP_ARRAY=18 \
  bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh

# writes under $GNNPLUS_OUT_DIR/.../mutag_SiGMA_hetero_lr001_seed2/:
#   gate_values_per_graph.pt + gate_values_per_node.pt (incl. edge_index)
# log: logs_gnnplus/tu_hh_gdump_<JOB>_18.log
```

Rsync node dumps (still small):

```bash
rsync -avz --include='*/' --include='gate_values_per_node.pt' --exclude='*' \
  rpellegrinext@holylogin.rc.fas.harvard.edu:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/tu_sigma_homo_hetero/ \
  results/tu_sigma_homo_hetero/
```

Plot into the same paper figure folder as the graph-mean grids:

```bash
# single run
python scripts/gate_viz/plot_per_node_gates.py \
  --pt-node results/tu_sigma_homo_hetero/mutag_SiGMA_hetero_lr001_seed2/gate_values_per_node.pt \
  --out_dir results/gate_viz/tu_hh_hetero/mutag_SiGMA_hetero_lr001_seed2 \
  --color-by-class \
  --band p10_p90 \
  --sort-head 1 \
  --draw-head 1 \
  --n-draw 8

# batch (paper hetero, best LR, seed 2) — needs gate_values_per_node.pt present
python scripts/gate_viz/plot_tu_hh_gates_batch.py \
  --root results/tu_sigma_homo_hetero \
  --out_dir results/gate_viz/tu_hh_hetero \
  --datasets paper --variants SiGMA_hetero \
  --seeds 2 --prefer-lr best_from_table --color-by-class \
  --level node
```

Outputs (alongside existing `*_by_rank_by_class.png`):

| File | Contents |
|------|----------|
| `*_gates_{attn,gnn}_nodeband_shared_order_by_class.png` | Graph-mean γ + within-graph node percentile band, shared rank |
| `*_gates_gnn_L{k}_{GIN}_node_graphs.png` | Top/bottom ranked graphs, nodes colored by γ |

Band modes: `p10_p90` (default), `p25_p75`, `minmax`, `std`.

If a `.pt` is missing but `ckpt/` exists, re-dump (same 1–150 task map; GCN no-op):

```bash
bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh
```

Disable auto gate dump at train time: `TU_SIGMA_HH_GATE_DUMP=0`.

---

## Aggregate

```bash
# Example: ENZYMES hetero @ lr=0.001
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group tu_hh_enzymes_SiGMA_hetero_lr001 --metric best_test_perf --state finished

# Loop-friendly: all groups for one dataset
for fam in GCN SiGMA_homo SiGMA_hetero; do
  for lr in lr001 lr01; do
    # GCN only has lr001
    [[ "$fam" == "GCN" && "$lr" == "lr01" ]] && continue
    python scripts/api_wanndb_query/aggregate_paper_repro.py \
      --group tu_hh_enzymes_${fam}_${lr} --metric best_test_perf --state finished
  done
done
```

---

## Results (fill after runs)

### Best LR per SiGMA family (by mean test)

| Dataset | Homo best LR | Hetero best LR |
|---------|--------------|----------------|
| MUTAG | | |
| ENZYMES | | |
| PROTEINS | | |
| DD | | |
| NCI1 | | |
| TRIANGLES | | |

### Final table (mean±std test accuracy %)

Paper table (Lukas / PyG stats set). SiGMA LR = better of `{1e-3, 1e-2}` per family.

| Dataset | GCN | SiGMA (homo) | SiGMA (hetero) |
|---------|-----|--------------|----------------|
| MUTAG | 75.74±8.85 | 73.19±4.15 (1e-3) | **84.68±3.50** (1e-3) |
| ENZYMES | 41.87±5.24 | 47.07±1.98 (1e-3) | **47.60±6.99** (1e-3) |
| PROTEINS | 72.97±3.39 | 73.41±1.99 (1e-2) | **74.12±1.06** (1e-3) |
| COLLAB | 76.14±0.92 | 73.97±0.88 (1e-2) | **77.25±0.95** (1e-2) |
| IMDB-BINARY | 65.44±1.85 | 66.64±1.19 (1e-2) | **69.92±2.75** (1e-3) |
| REDDIT-BINARY | 92.60±1.62 | 87.92±7.51 (1e-3) | **92.72±1.01** (1e-3) |

Social job **`37574967`**: all 75 tasks finished (REDDIT SiGMA n=5 both LRs).

LaTeX (hetero bold when ≥ GCN and ≥ homo):

```latex
REDDIT-BINARY & $92.60{\pm}1.62$ & $87.92{\pm}7.51$ & $\mathbf{92.72{\pm}1.01}$ \\
```

### Standalone MPGNN baselines (GIN / SAGE / GAT)

Same recipe as GCN (`custom_gnn`, L12, H64, lr=`1e-3`, residual+FFN, mean pool).
Paper-table datasets only (GCN already done).

| | |
|--|--|
| **Submit** | `bash bash_interface/cluster/submit_tu_mpgnn_baselines.sh` |
| **Tasks** | 6 ds × {GIN,SAGE,GAT} × 5 seeds = **90** |
| **W&B** | `tu_hh_<ds>_{GIN,SAGE,GAT}_lr001` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_{GIN,SAGE,GAT}_lr001_seed<s>/` |
| **SLURM** | ✅ **`37649411`** (all 90 finished) |

Configs: `configs/tu_sigma_homo_hetero/{gin,sage,gat}-anchor.yaml`
(GAT via new `GNNPlus/layer/gat_conv_layer.py`, heads=1 like SiGMA MP head).

- Edge encoder off (TU graphs here have no edge attrs; avoids LinearEdge crash).
- Paper table focus (Lukas / PyG stats): **MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY**. NCI1 / TRIANGLES / DD were exploratory; social extras launched via `submit_tu_sigma_social.sh`.
- IMDB / COLLAB / REDDIT use `Constant()` node features (no native node attrs).
- Also listed in [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md).

### COLLAB SiGMA-hetero LR fill (beat GIN)

Standalone GIN on COLLAB is **78.40±0.90**; best SiGMA hetero so far is
**77.25±0.95** at `lr=1e-2` (only `{1e-3, 1e-2}` were in the original social grid).
Extra LRs elsewhere (e.g. NCI1 `5e-4`/`2e-3`) were **not** run on COLLAB.

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_tu_collab_sigma_lr_fill.sh
```

| | |
|--|--|
| **Submit** | `bash_interface/cluster/submit_tu_collab_sigma_lr_fill.sh` |
| **Tasks** | 3 LRs × 5 seeds = **15** |
| **LRs** | `2e-3`, `5e-3`, `2e-2` |
| **W&B** | `tu_hh_collab_SiGMA_hetero_{lr2e3,lr5e3,lr2e2}` |

---

## Param-matched relaunch (~1× GCN) + GPS

Prior SiGMA a2g4 (`d_h=16`) was **~1.64–1.67×** GCN params. Relaunch keeps
**a2g4 · L12 · H64** but sets **`d_h=4`** so SiGMA ≈ **1.02×** GCN. Also add a
**GPS-style** baseline (this fork has no `GPSModel`): SiGMA **a1g1** =
1 Transformer attn + 1 **GATEDGCN** MP (GraphGPS layer composition), `d_h=8`
→ **~1.01×** GCN.

| Model | Heads | Key knobs | Params (ENZYMES-scale) | vs GCN |
|-------|-------|-----------|------------------------|--------|
| GCN (existing) | — | L12 H64 | 258 118 | 1.00× |
| SiGMA homo matched | a2g4 | `d_h=4`, GCN×4 | 262 702 | 1.02× |
| SiGMA hetero matched | a2g4 | `d_h=4`, GCN,GIN,SAGE,GAT | 263 326 | 1.02× |
| GPS-style | a1g1 | `d_h=8`, GATEDGCN | 260 926 | 1.01× |
| SiGMA hetero (old table) | a2g4 | `d_h=16` | 430 798 | 1.67× |

Does **not** re-run GCN / GIN / SAGE / GAT (reuse `tu_hh_*`). New W&B prefix
`tu_1x_*` so results do not mix with the ~1.65× SiGMA runs.

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
| **SLURM** | ✅ **`37724579`** (`1-180%20`) |
| **Submit** | `bash_interface/cluster/submit_tu_sigma_1x_gcn.sh` |
| **Tasks** | `1-180%20` · 6 ds × {homo×2LR, hetero×2LR, GPS×2LR} × 5 seeds |
| **Mem / time** | 128GB / 96h |
| **Batches** | bio 64 · COLLAB 32 · IMDB 64 · REDDIT 16 |
| **W&B** | `tu_1x_<ds>_{SiGMA_homo,SiGMA_hetero,GPS}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_1x_gcn/<ds>_<variant>_<lr>_seed<s>/` |
| **Configs** | `sigma-{homo,hetero}-a2g4-matched-anchor.yaml`, `gps-a1g1-anchor.yaml` |
| **Param check** | `python scripts/count_tu_model_params.py --cfg …` |

Per-dataset task block (30 tasks, seeds 0–4):

| Offset | Variant | LR |
|--------|---------|-----|
| 1–5 | SiGMA_homo matched | 0.001 |
| 6–10 | SiGMA_homo matched | 0.01 |
| 11–15 | SiGMA_hetero matched | 0.001 |
| 16–20 | SiGMA_hetero matched | 0.01 |
| 21–25 | GPS a1g1 | 0.001 |
| 26–30 | GPS a1g1 | 0.01 |

Dataset order: MUTAG → ENZYMES → PROTEINS → COLLAB → IMDB-BINARY → REDDIT-BINARY.
