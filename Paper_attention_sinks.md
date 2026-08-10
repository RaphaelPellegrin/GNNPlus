# Attention sinks in Graph Transformers — tracking

Paper: [arXiv:2606.08105](https://arxiv.org/pdf/2606.08105) (Fesser et al.) — NOP vs broadcast sinks; gating ≈ NOP intervention; registers ≈ broadcast.

Goal: show AS evidence on **ungated** graph attention (SiGMA / GPS), with gated contrast + ‖v_sink‖ diagnostics.

---

## Status

| Step | Status | Notes |
|------|--------|-------|
| Code: dense attn dump + ‖v‖ | ✅ | `collect_attention_maps`, `dump_attention_maps.py` |
| Code: train-time W&B sink PNGs | ✅ | `attention_sink_tracking.py` (sparse epochs) |
| Code: offline aggregate plots | ✅ | Heterogeneity_Profile `plot_attention_sinks_aggregate.py` |
| Cluster train MUTAG+ENZYMES | ⏳ | submit below after push |
| Rsync + paper figures | ⏳ | after jobs finish |
| Paper write-up | ⏳ | |

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
| `attn_sinks/*` scalars | mean sink rate, max α, vnorm ratio, per L/h |

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

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# Full 8 jobs (MUTAG+ENZYMES × 4 variants). W&B sink PNGs every 50 epochs.
bash bash_interface/cluster/submit_tu_attention_sinks.sh

# MUTAG-only smoke (tasks 1–4), denser panels every 25 epochs:
AS_ARRAY=1-4 AS_PARALLEL=4 AS_SINK_EVERY=25 \
  bash bash_interface/cluster/submit_tu_attention_sinks.sh
```

Record JOBID here: `________________`

---

## After train — rsync + offline aggregate (optional)

```bash
rsync -avz --progress \
  fasrc:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/tu_attention_sinks/ \
  results/tu_attention_sinks/

# Offline multi-batch aggregate (Heterogeneity_Profile):
cd ../Heterogeneity_Profile
PYTHONPATH=src python scripts/plots/plot_attention_sinks_aggregate.py \
  --input-dir ../GNNPlus/results/tu_attention_sinks/mutag_SiGMA_hetero_ungated_attn_lr001_seed2/attention_sinks \
  --output-dir visualizations/attention_sinks/mutag_ungated_attn \
  --tau 1.5 --epsilon 0.3
```

If mid-train `.pt` files are nested under `epXXXX/`, point `--input-dir` at the run’s `attention_sinks/` tree or flatten; end-of-run dump via `AS_DUMP_ATTN=1` / `dump_attention_maps.py` writes flat `attention_matrices/`.

---

## Paper figure checklist

- [ ] Ungated SiGMA: vertical stripes (degree-sorted exemplars)
- [ ] Sink-rate L×H heatmap
- [ ] Sink centrality (hub vs leaf) — offline aggregate
- [ ] ‖v_sink‖ / mean‖v‖ (NOP vs broadcast)
- [ ] Gated contrast (same arch)
- [ ] GPS ungated (+ gated)
- [ ] Epoch evolution panels from W&B (ep0 / ep50 / … / last)

---

## Design notes

- **No position-0:** PyG index is arbitrary; panels sort by **degree**.
- **Variable |V|:** dense attn only within-graph; truncate batch to `attention_sink_max_nodes`.
- **Gating:** Fesser: gating targets NOP. Ungated attn is required to *see* AS; gated runs are the intervention contrast.
