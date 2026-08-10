# Attention sinks in Graph Transformers — tracking

Paper: [arXiv:2606.08105](https://arxiv.org/pdf/2606.08105) (Fesser et al.) — NOP vs broadcast sinks; gating ≈ NOP intervention; registers ≈ broadcast.

Goal: **(1) find AS** in ungated graph transformers; **(2) study which node traits** sinks prefer; gated contrast + ‖v‖ for NOP vs broadcast.

---

## Research plan (locked)

### Phase A — Existence (order-free)
Over many graphs × layers × heads: \(\max\alpha\) vs \(1/n\), τ·μ sink rate, ε-sink rate, exemplar stripes (degree-sort **display only**).  
Primary: **ungated attn**. Contrast: **gated**.

### Phase B — Localization (pre-specified traits)
degree, PageRank, eigenvector, closeness, clustering, k-core:
- rank of argmax-α node (is sink = top hub?)
- Spearman \(\rho(\alpha, c)\) within graph, mean over graphs  
Offline: HP figures `17`–`25` from `plot_attention_sinks_aggregate.py`.

### Phase C — Mechanism (Fesser)
- ``‖v_sink‖ / mean‖v‖``: ≪1 ⇒ **NOP**; ~1 ⇒ not NOP
- ``stable_rank(AV)`` + mean row-cosine of ``AV``: ~1 + high cosine ⇒ **broadcast**
- Saved in mid-train ``attention_batch_ep*.pt`` and full dumps: ``value_norms``, ``head_outputs`` (``AV``), ``attn_gates``
- Offline: ``scripts/attention_sinks/summarize_nop_broadcast.py``
- Optional later: virtual nodes / registers


---

## Status

| Step | Status | Notes |
|------|--------|-------|
| Code: dump + ‖v‖ + W&B panels | ✅ | existence (Phase A) mid-train |
| Code: offline + centrality | ✅ | Phase B aggregate plotter |
| MUTAG SiGMA (gated/ungated) | ✅ | JOB 37966868 · W&B `ibm24bvj` / `tqe7n2j4` |
| MUTAG GPS | ✅ | JOB **37969759** · W&B `jj98ytzn` / `h7ut8lsp` |
| ENZYMES + paper TU scale-up | ⏳ | scripts expanded to 6 ds; submit GPS ungated first |
| Full-dataset dump → Phase B figs | ✅ MUTAG | ENZYMES/COLLAB/… after train |
| Paper write-up | ⏳ | |

### First MUTAG findings (ep999 train batch, strongest head)

| Run | Head | mean α / uniform | sink = max-deg hub? | mean deg-rank of sink | vnr |
|-----|------|------------------|---------------------|------------------------|-----|
| **GPS ungated** | L0 h0 | **~4.9×** | **0%** | **~15** (leaves) | ~1.05 |
| GPS gated | L0 h0 | ~5.2× | 0% | ~15 (leaves) | ~1.05 |
| SiGMA ungated | L2 h0 | ~4.0× | 4% | ~9 | ~1.15 |
| SiGMA gated | L4 h1 | ~3.9× | 7% | ~8 | ~1.21 |

**Takeaway (batch):** clear AS (τ·μ on argmax, α ≫ 1/n). GPS L0 sinks looked like **low-degree periphery**. ‖v‖ ratio ~1 → not classic NOP.

Plots: `results/tu_attention_sinks/sink_graph_plots/` · W&B panels: `results/tu_attention_sinks/wandb_panels/`

### Dataset-wide MUTAG (Phase B — all 188 graphs)

Dumped via `dump_attention_maps.py` (CPU `map_location`) from **best-val ckpts** (not ep999): GPS ungated **ep65**, GPS gated **ep68**, SiGMA ungated **ep11**, SiGMA gated **ep156**.  
HP aggregate: `Heterogeneity_Profile/visualizations/attention_sinks/mutag_{GPS,SiGMA}_{un,}gated_full/` (figs **17–25**, `records.csv`, `summary_layer_head.txt`).

| Run (ckpt) | Strong AS heads (rate≥0.5) | On those heads: hub% / deg≤2 / ρ(α,deg) | L0 specifically |
|------------|----------------------------|------------------------------------------|-----------------|
| **GPS ungated ep65** | **L0H0 only** (rate=1.00, max_ratio≈3.8×) | **0% / 100% / −0.63** | periphery confirmed |
| GPS gated ep68 | 5 heads; L0 still leaves; L4 strong AS | mixed (L0: 0%/100%/−0.50; L4: hub%≈15, ρ=+0.34) | L0 periphery; later layers not |
| SiGMA ungated ep11 | L1H0, L3H1 | **hub-leaning** (~11% hub, deg≈3.0, ρ≈+0.69) | L0 flat (no AS) |
| SiGMA gated ep156 | 7 heads | mixed (L1H0: deg≈1.8, ρ≈−0.37; L0H1: ρ≈+0.61) | not a clean periphery story |

**Verdict:** “periphery sinks” **holds dataset-wide for GPS ungated L0** (all 188 graphs: mean sink degree **1.06**, never the max-degree hub, Spearman ρ(α, degree/PageRank) ≈ **−0.64**). It is **not** a universal claim across layers or SiGMA — strong SiGMA sinks lean **higher** degree/PageRank. Mid-depth GPS layers rarely sink (τ-rate ≪ 0.5) and, when they do, track centrality more.

---

## Models (4 variants × MUTAG + ENZYMES, seed 2)

| Variant | Arch | Gate | Role |
|---------|------|------|------|
| `SiGMA_hetero_ungated_attn` | a2g4 matched | attn **none**, MP headwise | Primary AS evidence |
| `SiGMA_hetero_gated` | a2g4 matched | headwise | Contrast (sinks should weaken) |
| `GPS_ungated_attn` | a1g1 GATEDGCN+attn | attn **none**, MP headwise | Lit-style GT |
| `GPS_gated` | a1g1 | headwise | GPS + gating contrast |

- `attn_mask=full` (all pairs **within** graph)
- Out: `$GNNPLUS_OUT_DIR/tu_attention_sinks/<ds>_<variant>_lr001_seed2/`
- W&B group: `tu_as_<ds>_<variant>`

---

## W&B media (logged during training)

Enabled via `gnn.hybrid.log_attention_sinks True`.

**When:** epoch `0`, every `attention_sink_every` (default **50**), and **last** epoch.  
Not every epoch (too heavy).

| W&B key | Content |
|---------|---------|
| `attn_sinks/panel_by_layer_head` | Grid L×H attention heatmaps (graph 0, **degree-sorted**); red line = argmax α |
| `attn_sinks/panel_mean_over_heads` | Per-layer mean over heads |
| `attn_sinks/panel_sink_rate_LxH` | τ·μ sink present (0/1) heatmap |
| `attn_sinks/panel_max_alpha_LxH` | max column-mean α |
| `attn_sinks/panel_sink_vnorm_ratio_LxH` | ‖v_sink‖ / mean‖v‖ (NOP ≪ 1, broadcast ~ 1) |
| `attn_sinks/panel_av_stable_rank_LxH` | stable_rank(AV) (~1 ⇒ broadcast) |
| `attn_sinks/panel_av_row_cosine_LxH` | mean row-cosine of AV (broadcast ⇒ high) |
| `attn_sinks/panel_mechanism_LxH` | heuristic 0=NOP / 0.5=amb / 1=broadcast |
| `attn_sinks/*` scalars | sink rate, max α, vnorm, stable_rank, row_cos, mech fracs |

Also on disk: `<run_dir>/attention_sinks/epXXXX/*.png` + `attention_batch_epXXXX.pt`.

---

## Git (run locally — do **not** ask the agent to `git add` / `push`)

### GNNPlus

```bash
cd /Users/pellegrinraphael/Desktop/Academic_Research/Repos_GNN/GNNPlus

git status
git diff --stat

git add \
  GNNPlus/attention_sink_tracking.py \
  GNNPlus/config/gated_hybrid_config.py \
  GNNPlus/train/custom_train.py \
  GNNPlus/layer/gated_hybrid_layer.py \
  GNNPlus/network/hybrid_gnn.py \
  scripts/attention_sinks/dump_attention_maps.py \
  bash_interface/cluster/run_tu_attention_sinks.sh \
  bash_interface/cluster/submit_tu_attention_sinks.sh \
  Paper_attention_sinks.md

git commit -m "$(cat <<'EOF'
Add attention-sink W&B panels and TU AS train campaign.

Log sparse-epoch Fesser-style attn heatmaps (per head, mean-over-heads,
sink-rate / max-α / v-norm) for ungated SiGMA/GPS vs gated contrast on
MUTAG and ENZYMES.
EOF
)"

git status
# push yourself when ready:
# git push -u origin HEAD
```

### Heterogeneity_Profile (offline plots + restored sink package)

```bash
cd /Users/pellegrinraphael/Desktop/Academic_Research/Repos_GNN/Heterogeneity_Profile

git status
git diff --stat

git add \
  src/graph_moes/attention_sinks/ \
  src/graph_moes/utils/attention_sink_post.py \
  scripts/plots/plot_attention_sinks_aggregate.py \
  scripts/plots/plot_attention_single_graph.py \
  tests/test_attention_sinks.py \
  tests/test_attention_sink_post.py \
  bash_interface/sweeps/run_attention_sink_analysis.sh

git commit -m "$(cat <<'EOF'
Restore attention-sink analysis package and aggregate plot scripts.

Rebuild core/node_features for graph AS metrics (τ·μ, ε-sinks, centrality)
and wire offline figure generation used with GNNPlus attention dumps.
EOF
)"

git status
# git push -u origin HEAD
```

---

## Cluster (after push + Duo)

Paper TU set (6): MUTAG · ENZYMES · PROTEINS · COLLAB · IMDB-BINARY · REDDIT-BINARY  
(4 variants × 6 = **24** tasks). COLLAB uses `lr=0.01`; others `0.001`.

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
# if scripts not pushed yet: scp run/submit from laptop

# Cheapest ×uniform vs |V| test — GPS ungated on all 6 ds:
AS_ARRAY=4,8,12,16,20,24 AS_PARALLEL=6 \
  bash bash_interface/cluster/submit_tu_attention_sinks.sh

# Skip MUTAG (done), all 4 variants on remaining 5 ds:
AS_ARRAY=5-24 AS_PARALLEL=10 \
  bash bash_interface/cluster/submit_tu_attention_sinks.sh
```

Record JOBIDs:
- `37966868` — MUTAG smoke 1–4 (2026-08-09)
- `37969759` — MUTAG GPS 3–4 retry
- **`37971764`** — GPS ungated × 6 paper TU (tasks 4,8,12,16,20,24) · 2026-08-09
- **`37971765`** — ENZYMES variants 5–7 (SiGMA gated/ungated + GPS gated)

Task map: `ds×4 + variant` with ds order  
`mutag, enzymes, proteins, collab, imdb_binary, reddit_binary` · variants 0..3 as before.  
GPS ungated = **4,8,12,16,20,24**.

**Hypothesis note (MUTAG L0 within-ds):** α **falls** with \(n\) (corr ≈ −0.59) so ×uniform only weakly rises (corr ≈ +0.24). Cross-dataset (ENZYMES/COLLAB/REDDIT) is needed to see if larger graphs give bigger multiples.

Monitor:
```bash
squeue -u $USER | grep tu_attn
```
W&B: groups `tu_as_<ds>_*`

---

## After train — rsync + offline aggregate

```bash
rsync -avz --progress \
  fasrc:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/tu_attention_sinks/ \
  results/tu_attention_sinks/
```

# Full dump from best ckpt (local CPU OK — dump uses map_location=cpu):
# (see earlier MUTAG GPS example; swap run_dir / dataset.name)

# HP centrality panels 17–25:
# cd ../Heterogeneity_Profile && PYTHONPATH=src python scripts/plots/plot_attention_sinks_aggregate.py ...

Done locally for all 4 MUTAG variants → `visualizations/attention_sinks/mutag_{GPS,SiGMA}_{un,}gated_full/`.

---

## Paper figure checklist

- [x] Ungated SiGMA: vertical stripes (degree-sorted exemplars) — fig 16 in HP outs
- [x] Sink-rate L×H heatmap — HP figs 01–03 family
- [x] Sink centrality (hub vs leaf) — offline aggregate figs **17–25** (MUTAG full)
- [x] ‖v_sink‖ / mean‖v‖ + stable_rank(AV) + row-cosine (NOP vs broadcast) — logged + dumped
- [x] Gated contrast (same arch)
- [x] GPS ungated (+ gated)
- [ ] Epoch evolution panels from W&B (ep0 / ep50 / … / last)
- [ ] ENZYMES / COLLAB / paper-TU GPS ungated + α vs \(n\) across datasets
- [ ] ENZYMES full dump + aggregate

---

## Design notes

- **No position-0:** PyG index is arbitrary; panels sort by **degree**.
- **Variable |V|:** dense attn only within-graph; truncate batch to `attention_sink_max_nodes`.
- **Gating:** Fesser: gating targets NOP. Ungated attn is required to *see* AS; gated runs are the intervention contrast.
- **REDDIT:** full `.pt` dump skipped unless `AS_DUMP_REDDIT=1`; rely on W&B mid-train panels (max_nodes=512).

