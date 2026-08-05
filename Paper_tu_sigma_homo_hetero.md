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
║  🛑🛑🛑  TO RUN  ·  paste SLURM JOBID below after submit  🛑🛑🛑 ║
║  6 TU × {GCN, SiGMA_homo×2LR, SiGMA_hetero×2LR} × 5 seeds = 150  ║
╚══════════════════════════════════════════════════════════════════╝
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_tu_sigma_homo_hetero.sh
```

| Field | Value |
|-------|-------|
| **SLURM array** | 🛑 *paste JOBID here* |
| **Tasks** | `1-150%8` (override with `TU_SIGMA_HH_ARRAY` / `TU_SIGMA_HH_PARALLEL`) |
| **Partition / mem / time** | `mweber_gpu` / 64GB / 96h |
| **Logs** | `logs_gnnplus/tu_sigma_hh_<JOBID>_<TASK>.log` |
| **Checkpoints / stats** | `$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_<variant>_<lr>_seed<s>/` |
| **Best model** | `…/ckpt/` (best val epoch) |
| **Per-graph gates** | `…/gate_values_per_graph.pt` (SiGMA only; `[N, L, H]` attn + gnn) |
| **Meta** | `…/train_meta.txt` + `config_used.yaml` |
| **W&B** | `tu_hh_<ds>_{GCN,SiGMA_homo,SiGMA_hetero}_{lr001,lr01}` |

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

Offline re-dump (if `.pt` missing; same 1–150 task map; GCN slots no-op):

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

### Final table (mean±std test accuracy)

| Dataset | GCN | SiGMA (homo) | SiGMA (hetero) |
|---------|-----|--------------|----------------|
| MUTAG | | | |
| ENZYMES | | | |
| PROTEINS | | | |
| DD | | | |
| NCI1 | | | |
| TRIANGLES | | | |

---

## Notes

- Edge encoder off (TU graphs here have no edge attrs; avoids LinearEdge crash).
- IMDB / COLLAB not included (degree features / different loader path); add later if needed.
- Also listed in [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md).
