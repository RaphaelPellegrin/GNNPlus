# Synthetic GCN vs GIN operator-routing benchmark

Controlled benchmark to test whether **SiGMA headwise gating routes to the correct
message-passing operator per graph** — addressing reviewer requests for (i) a direct
link between gating and graph-level heterogeneity / operator need, and (ii) empirical
validation that the model adapts differently across graphs in one dataset.

**Related:** [`Paper_heterogeneity.md`](Paper_heterogeneity.md) (real-data heterogeneity
profiles) · [`rebuttal.md`](rebuttal.md) (reviewer log)

**W&B project:** [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
**Planned tag:** `gcn_gin_routing_synthetic`  
**Planned groups:** `paper_gcn_gin_routing_{toy,sigma}_*`

---

## Reviewer targets

| Concern | What this benchmark shows |
|---------|---------------------------|
| Gates vary but are not tied to graph “needs” | Labels are **defined** by which aggregation rule is correct; gates should align |
| No proof of per-graph adaptation | Mixed dataset + **opposite-sign pairs** force graph-specific routing |
| Request for synthetic controlled test | One-layer star graphs, binary labels, interpretable GCN vs GIN targets |

**Scope note (for paper text):** This isolates **operator routing**. Real-data gate
variation and TMD / heterogeneity-profile links remain complementary evidence
(see §Future bridge below).

---

## Graph construction

Each graph is a **rooted star** (directed or undirected — fix in generator v1):

```
        u₁ (x₁ ∈ {-1,+1})
       /
  r ──┼── u₂ … uₖ     k ∈ {4,…,8} neighbors
       \
        uₖ
```

- **Root** `r`: degree `d_r = k`. Feature layout TBD in §Feature layout.
- **Neighbors** `u`: scalar signal `x_u ∈ {-1, +1}`.
- **Dummy leaves:** each `u` has `ℓ_u` zero-feature leaves so **neighbor degree**
  `d_u` varies (GCN normalization matters; blocks degree-counting shortcuts).
- **Graph type** `τ ∈ {0, 1}`: `τ=0` → GCN-type label rule; `τ=1` → GIN-type.

**Dataset size (initial):**

| Split | Per type | Total | Notes |
|-------|---------:|------:|-------|
| Train | 5 000 | 10 000 | |
| Val | 1 000 | 2 000 | |
| Test | 1 000 | 2 000 | |
| **Opposite-sign pairs** | ≥ 20% of train | | same `(x, degrees)` pattern, different `τ` and `y` |

Generator: `scripts/synthetic/generate_gcn_gin_routing_dataset.py`  
Core module: `GNNPlus/loader/dataset/gcn_gin_routing.py`  
On-disk name: `gcn_gin_routing_v1` under `$GNNPLUS_DATASET_DIR/synthetic/` (or `results/gcn_gin_routing/data/` locally)

**Example figures (one per graph):** `results/gcn_gin_routing/examples/`  
| # | File |
|---|------|
| 1 | [`fig_example_01_easy_aligned_gcn.png`](results/gcn_gin_routing/examples/fig_example_01_easy_aligned_gcn.png) |
| 2 | [`fig_example_02_easy_gin_vote.png`](results/gcn_gin_routing/examples/fig_example_02_easy_gin_vote.png) |
| 3 | [`fig_example_03_medium_degree_imbalance.png`](results/gcn_gin_routing/examples/fig_example_03_medium_degree_imbalance.png) |
| 4 | [`fig_example_04_hard_opposite_sign_gcn.png`](results/gcn_gin_routing/examples/fig_example_04_hard_opposite_sign_gcn.png) |
| 5 | [`fig_example_05_hard_opposite_sign_gin.png`](results/gcn_gin_routing/examples/fig_example_05_hard_opposite_sign_gin.png) |
| 6 | [`fig_example_06_hard_near_threshold.png`](results/gcn_gin_routing/examples/fig_example_06_hard_near_threshold.png) |
| 7 | [`fig_example_07_medium_many_neighbors.png`](results/gcn_gin_routing/examples/fig_example_07_medium_many_neighbors.png) |

Regenerate: `python scripts/synthetic/plot_gcn_gin_routing_examples.py`  
Optional combined grid: add `--combined` → `fig_example_graphs.png`

---

## Label rules

Binary label `y \in \{0, 1\}` from root readout (threshold at 0).

> **Graph type τ — label rules**
>
> | τ | Type | Label rule |
> |---|------|------------|
> | **0** | GCN-type | \(y = \mathbb{1}\!\left[\sum_{u \in N(r)} \frac{x_u}{\sqrt{(d_r+1)(d_u+1)}} > 0\right]\) |
> | **1** | GIN-type | \(y = \mathbb{1}\!\left[\sum_{u \in N(r)} x_u > 0\right]\) |
>
> - **τ** — binary type at the root (which aggregation rule defines the label)  
> - **\(d_r\)** — root degree (= number of signal neighbors)  
> - **\(d_u\)** — degree of neighbor \(u\) (includes dummy leaves)  
> - **\(x_u \in \{-1,+1\}\)** — scalar feature on signal neighbors  
> - **\(y \in \{0,1\}\)** — graph-level class label the model must predict (from the τ-selected rule)

> **Node features (2-dim `x`)**
>
> | Node | Signal feature \(x\) | Type channel (2nd dim) |
> |------|---------------------:|------------------------|
> | Root `r` | **0** | **τ** (0 or 1) |
> | Signal neighbors | **+1 or −1** | **0** |
> | Dummy leaves (gray) | **0** | **0** |

**Opposite-sign cases:** sample neighbor features and degrees such that the GCN sum
and GIN sum have **opposite signs**. Same local pattern, different `τ` → different
`y`. Requires graph-level type information (not recoverable from local structure
alone).

---

## Feature layout (open design — resolve in Track A)

| Option | Root `x_r` | Neighbor `x_u` | Pros | Cons |
|--------|----------|----------------|------|------|
| **A (recommended)** | `[τ, 0]` | `[x_u, 0]` | Type visible at root for gates | GIN self-term sees `τ` unless conv input masked |
| **B** | `[0]` | `[x_u]` | Clean conv on neighbors | Gates at root constant unless type injected elsewhere |
| **C** | `[τ]` only in `graph.type` | `[x_u]` | Clean MP | Needs small fork: `gate_proj` reads `type`, conv does not |

**Decision:** _TBD after Track A toy runs_  
**Track A fix:** use **custom sum / norm-sum convs** with **no self-loop** on root;
type bit feeds **gate + classifier only** (not aggregation).

---

## Two-track protocol

We run **both** tracks and report the same metrics on each.

### Track A — Toy (label-faithful convs)

**Goal:** Labels match the forward pass **exactly** (cleanest causal story).

| Knob | Value |
|------|--------|
| Architecture | 1× hybrid layer, **a0g2**, `GIN,GCN`, **no attention** |
| Conv | Custom **`NormSumConv`** (GCN rule) + **`SumConv`** (GIN rule) |
| Self-loops | **Off** on MP edges (or root excluded from self term) |
| `d_h` | **1** (scalar channels) |
| Nonlinearity | Identity or single shared `ReLU` **after** gated fuse |
| Gate | `mp_gate: headwise` on both MP heads |
| Readout | **Root node** embedding → linear classifier |
| Train | Adam, lr sweep `{1e-3, 1e-2}`, 5 seeds |

Config (planned): `configs/synthetic/gcn_gin_routing_toy_a0g2.yaml`

### Track B — Full SiGMA (production heads)

**Goal:** Show **qualitative routing** reproduces with real `GINConv` / `GCNConv`
heads from [`GNNPlus/layer/gated_hybrid_layer.py`](GNNPlus/layer/gated_hybrid_layer.py).

| Knob | Value |
|------|--------|
| Architecture | 1× `GatedHybridGraphLayer`, **a0g2**, `GIN,GCN`, no attention |
| Conv | PyG **`GINConv`** (MLP inside) + **`GCNConv`** |
| `d_h` | **4** (small; linear-ish init) |
| Gate | `mp_gate: headwise` |
| Self-loops | Default PyG; document mismatch vs labels in §Alignment notes |
| Readout | Root node → MLP or linear head |
| Train | Same recipe as Track A |

Config (planned): `configs/synthetic/gcn_gin_routing_sigma_a0g2.yaml`

**Success criterion (Track B):** same **directional** gate pattern as Track A
(γ_GCN ↑ on GCN-type, γ_GIN ↑ on GIN-type) and masking ablation with same
**asymmetry** — exact label match not required.

---

## Baselines & ablations

**Yes — run GCN-only and GIN-only on the same synthetic data** (both Track A toy
and Track B full SiGMA). This was always in the plan; it is **essential** for the
reviewer story:

| Baseline | What it shows |
|----------|----------------|
| **GCN-only** (`a0g1`, head `GCN`) | Strong on τ=0 graphs, ~chance on τ=1 → **one operator is not enough** |
| **GIN-only** (`a0g1`, head `GIN`) | Strong on τ=1 graphs, ~chance on τ=0 → symmetric |
| **SiGMA gated** (`a0g2`, `GIN,GCN`) | High on **both** types → **routing** beats either single head |
| **SiGMA ungated** | Both heads always on → worse than gated on mixed data |
| **Mask ablation** | Causal complement: dropping the right head hurts the matching type |

On a **balanced mixed** test set (50% τ=0, 50% τ=1), GCN-only and GIN-only should
land near **~50% overall** but **high on their matching type** — the key figure is
a **2×2 per-type accuracy table**, not pooled accuracy alone.

### Model matrix (run all on same train/val/test split)

| ID | Track | Arch | Heads | Gate | Config (planned) |
|----|-------|------|-------|------|------------------|
| **M1** | A toy | `a0g2` | `GIN,GCN` (SumConv, NormSumConv) | headwise | `gcn_gin_routing_toy_a0g2_gated.yaml` |
| **M2** | A toy | `a0g2` | `GIN,GCN` | none | `gcn_gin_routing_toy_a0g2_ungated.yaml` |
| **M3** | A toy | `a0g1` | `GCN` | n/a | `gcn_gin_routing_toy_a0g1_gcn.yaml` |
| **M4** | A toy | `a0g1` | `GIN` | n/a | `gcn_gin_routing_toy_a0g1_gin.yaml` |
| **M5** | B σ | `a0g2` | `GIN,GCN` (PyG) | headwise | `gcn_gin_routing_sigma_a0g2_gated.yaml` |
| **M6** | B σ | `a0g2` | `GIN,GCN` | none | `gcn_gin_routing_sigma_a0g2_ungated.yaml` |
| **M7** | B σ | `a0g1` | `GCN` | n/a | `gcn_gin_routing_sigma_a0g1_gcn.yaml` |
| **M8** | B σ | `a0g1` | `GIN` | n/a | `gcn_gin_routing_sigma_a0g1_gin.yaml` |

Same hyperparams across M1–M8: 1 layer, root readout, Adam lr ∈ `{1e-3, 1e-2}`,
5 seeds, 14k graphs (10k/2k/2k). **Mask ablation** = eval-only on best gated run
(M1 or M5).

### Expected test accuracy (mixed set)

| Model | Test all | τ=0 (GCN-type) | τ=1 (GIN-type) |
|-------|---------:|---------------:|---------------:|
| SiGMA gated (main) | **high** | **high** | **high** |
| SiGMA ungated | moderate | moderate | moderate |
| **GCN-only** | **~50%** | **high** | **low** |
| **GIN-only** | **~50%** | **low** | **high** |

Record **per-type** accuracy (not pooled only).

---

## Metrics & figures

### Primary (both tracks)

| Metric | Definition | Target |
|--------|------------|--------|
| **Test acc (all)** | Correct / N | > 95% (Track A); > 90% (Track B) |
| **Test acc GCN-type** | `τ=0` only | high |
| **Test acc GIN-type** | `τ=1` only | high |
| **Δγ_GCN** | `mean(γ_GCN \| τ=0) − mean(γ_GCN \| τ=1)` at **root** | **> 0** |
| **Δγ_GIN** | `mean(γ_GIN \| τ=1) − mean(γ_GIN \| τ=0)` at **root** | **> 0** |
| **Mask asymmetry** | `acc_drop(GCN-type \| mask GCN) − acc_drop(GCN-type \| mask GIN)` | **> 0** |
| | `acc_drop(GIN-type \| mask GIN) − acc_drop(GIN-type \| mask GCN)` | **> 0** |

Gate extraction: `HybridGNN.extract_gate_stats()` / layer `return_gate_stats=True` —
report **root node** only (ignore dummy leaves).

### Planned figures

| Fig | Content | Path (planned) |
|-----|---------|----------------|
| 1 | Boxplots: root γ_GCN, γ_GIN by `τ` (toy vs sigma side-by-side) | `results/gcn_gin_routing/fig_gate_by_type.pdf` |
| 2 | **2×2 per-type acc** (SiGMA vs GCN-only vs GIN-only vs ungated) | `results/gcn_gin_routing/fig_baseline_per_type.pdf` |
| 3 | Masking ablation (gated SiGMA) | `results/gcn_gin_routing/fig_mask_ablation.pdf` |
| 3 | Example opposite-sign pair (structure + labels) | [`results/gcn_gin_routing/fig_example_graphs.png`](results/gcn_gin_routing/fig_example_graphs.png) |

Plot scripts: `scripts/synthetic/plot_gcn_gin_routing_examples.py` (examples) · `plot_gcn_gin_routing_results.py` (results, TBD)

---

## Results tracker

**Last updated:** _2026-08-27 — design only, no runs yet_

### Track A — Toy (label-faithful)

| Seed | lr | Test all | GCN-type | GIN-type | Δγ_GCN | Δγ_GIN | W&B run |
|-----:|----|---------:|---------:|---------:|-------:|-------:|---------|
| 0 | 1e-3 | — | — | — | — | — | |
| 1 | 1e-3 | — | — | — | — | — | |
| 2 | 1e-3 | — | — | — | — | — | |
| 3 | 1e-3 | — | — | — | — | — | |
| 4 | 1e-3 | — | — | — | — | — | |
| **mean±std** | | — | — | — | — | — | |

**Best seed summary:** _TBD_

### Track B — Full SiGMA

| Seed | lr | Test all | GCN-type | GIN-type | Δγ_GCN | Δγ_GIN | W&B run |
|-----:|----|---------:|---------:|---------:|-------:|-------:|---------|
| 0 | 1e-3 | — | — | — | — | — | |
| 1 | 1e-3 | — | — | — | — | — | |
| 2 | 1e-3 | — | — | — | — | — | |
| 3 | 1e-3 | — | — | — | — | — | |
| 4 | 1e-3 | — | — | — | — | — | |
| **mean±std** | | — | — | — | — | — | |

**Qualitative match to Track A?** _TBD (yes / partial / no)_

### Single-head baselines — Track A (toy)

| Model | Seed | lr | Test all | τ=0 | τ=1 | W&B run |
|-------|-----:|----|---------:|----:|----:|---------|
| GCN-only | 0–4 | best | — | — | — | |
| GIN-only | 0–4 | best | — | — | — | |
| Ungated a0g2 | 0–4 | best | — | — | — | |
| SiGMA gated | 0–4 | best | — | — | — | |

### Single-head baselines — Track B (full SiGMA)

| Model | Seed | lr | Test all | τ=0 | τ=1 | W&B run |
|-------|-----:|----|---------:|----:|----:|---------|
| GCN-only | 0–4 | best | — | — | — | |
| GIN-only | 0–4 | best | — | — | — | |
| Ungated a0g2 | 0–4 | best | — | — | — | |
| SiGMA gated | 0–4 | best | — | — | — | |

### Summary (best lr, mean ± std over 5 seeds)

| Model | Track | Test all | τ=0 | τ=1 | Notes |
|-------|-------|---------:|----:|----:|-------|
| GCN-only | A | — | — | — | should ace τ=0, fail τ=1 |
| GIN-only | A | — | — | — | opposite |
| Ungated a0g2 | A | — | — | — | |
| SiGMA gated | A | — | — | — | main |
| GCN-only | B | — | — | — | |
| GIN-only | B | — | — | — | |
| Ungated a0g2 | B | — | — | — | |
| SiGMA gated | B | — | — | — | main |

### Masking ablation (eval-only, gated model)

| Masked head | GCN-type acc | GIN-type acc | Δ vs full |
|-------------|-------------:|-------------:|----------:|
| none (full) | — | — | — |
| GCN | — | — | |
| GIN | — | — | |

_Fill separately for Track A and Track B._

---

## Alignment notes (Track B)

Document known mismatches between labels and production heads:

| Issue | Label uses | SiGMA GIN head | SiGMA GCN head | Mitigation |
|-------|------------|----------------|----------------|------------|
| GIN aggregation | `Σ x_u` | `MLP((1+ε)x + Σ x_j)` | — | small `d_h`, init near identity; report routing not exact fit |
| GCN norm | `1/√((d_r+1)(d_u+1))` | PyG symmetric norm + self-loop | — | verify `add_self_loops`; optional `GCNConv(cached=False)` |
| Type at root | not in sum | self term may include `τ` | self term may include `τ` | Option C in §Feature layout |
| Readout depth | 1-hop agg | 1 layer but with MLP/BN | 1 layer + weights | Track A as ground truth |

---

## Implementation checklist

- [x] `GNNPlus/loader/dataset/gcn_gin_routing.py` — star builder, scores, `GcnGinRoutingDataset`
- [x] `scripts/synthetic/generate_gcn_gin_routing_dataset.py`
- [x] `scripts/synthetic/plot_gcn_gin_routing_examples.py` — curated examples figure
- [x] `GNNPlus/layer/routing_sum_conv.py` — Track A convs (`RoutingSumConv`, `RoutingNormSumConv`)
- [x] Wire `ROUTING_SUM` / `ROUTING_NORMGCN` into `_ProjectedMPHead` (`gated_hybrid_layer.py`)
- [x] `configs/synthetic/gcn_gin_routing_toy_a0g2_gated.yaml`
- [x] `configs/synthetic/gcn_gin_routing_toy_a0g2_ungated.yaml`
- [x] `configs/synthetic/gcn_gin_routing_toy_a0g1_gcn.yaml` — **GCN-only baseline**
- [x] `configs/synthetic/gcn_gin_routing_toy_a0g1_gin.yaml` — **GIN-only baseline**
- [x] `configs/synthetic/gcn_gin_routing_sigma_a0g2_gated.yaml`
- [x] `configs/synthetic/gcn_gin_routing_sigma_a0g2_ungated.yaml`
- [x] `configs/synthetic/gcn_gin_routing_sigma_a0g1_gcn.yaml` — **GCN-only baseline**
- [x] `configs/synthetic/gcn_gin_routing_sigma_a0g1_gin.yaml` — **GIN-only baseline**
- [ ] `scripts/synthetic/eval_gcn_gin_routing_masks.py` — head-masking eval
- [ ] `scripts/synthetic/plot_gcn_gin_routing_results.py`
- [x] `bash_interface/cluster/submit_gcn_gin_routing.sh` + `run_gcn_gin_routing.sh`
- [ ] Add row to [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) when submitted (paste JOBIDs)

---

## Cluster (ready to launch)

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull   # after you push local changes

# 1) Generate data once (~1 min CPU, 10k/2k/2k graphs)
python scripts/synthetic/generate_gcn_gin_routing_dataset.py \
  --root "$GNNPLUS_DATASET_DIR/GcnGinRouting"

# 2) Submit both tracks (4 models × 2 LRs × 5 seeds = 80 tasks each)
bash bash_interface/cluster/submit_gcn_gin_routing.sh both
# Or: toy | sigma | both
# Optional auto-prep if data missing: GCN_GIN_ROUTING_PREP_DATA=1 bash .../submit_gcn_gin_routing.sh both
```

| Job | SLURM JOBID | Array | Tasks | Status | Logs |
|-----|------------:|-------|------:|--------|------|
| toy (Track A) | — | `1-80%10` | 80 | 🛑 ready | `logs_gnnplus/gcn_gin_route_toy_<JOBID>_<TASK>.log` |
| sigma (Track B) | — | `1-80%10` | 80 | 🛑 ready | `logs_gnnplus/gcn_gin_route_sigma_<JOBID>_<TASK>.log` |

**W&B:** tag `gcn_gin_routing_synthetic` · groups `paper_gcn_gin_routing_{toy,sigma}_<model>_<lr>`  
**Out:** `$GNNPLUS_OUT_DIR/gcn_gin_routing/<track>/<model>_<lr>_seed<s>/`

Local smoke (verified 2026-08-27):

```bash
python main.py --cfg configs/synthetic/gcn_gin_routing_toy_a0g2_gated.yaml \
  --repeat 1 seed 0 wandb.use False optim.max_epoch 2 dataset.dir results/gcn_gin_routing/data
```

---

## Future bridge (real data — optional)

After synthetic results, one lightweight real-data panel:

- Dataset: MUTAG or ENZYMES (heterogeneity pickles exist)
- x-axis: TMD class or per-graph hardness (1 − avg accuracy)
- y-axis: `γ_GCN − γ_GIN` at readout node / graph mean
- Claim: **routing correlates with heterogeneity**, not only on synthetic stars

Tracker: add subsection here or link from [`Paper_heterogeneity.md`](Paper_heterogeneity.md).

---

---

## Changelog

| Date | Event |
|------|--------|
| 2026-08-27 | Toy dataset module + 7-panel example figure (`fig_example_graphs.png`) |
| 2026-08-27 | 8 configs, cluster scripts, local smoke (toy + sigma); `pair_id` collate fix |
