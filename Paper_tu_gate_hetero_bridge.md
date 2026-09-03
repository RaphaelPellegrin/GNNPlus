# TU gate–operator bridge (MUTAG + ENZYMES)

```text
╔══════════════════════════════════════════════════════════════════╗
║  Xu L=4 SiGMA a2g4 × 5 seeds  🔄 44258255  (gpu_h200 1-10%10)   ║
║  GAT specialist prefs (≥100)  🛑 TO SUBMIT after git push        ║
║  L=12 a2g4 join: all 12 MP layers scanned — no routing signal   ║
╚══════════════════════════════════════════════════════════════════╝
```

## GAT specialist preference (Xu recipe)

Preference figure currently has **GCN / GIN / SAGE only**. SiGMA `a2g4` also
has a **GAT** MP head, so we add standalone GAT hetero profiles (same Xu HPs
as SAGE: L=4, H=64, 350 ep, sum pool).

| Field | Value |
|-------|-------|
| **Jobs** | 2 (mutag_gat, enzymes_gat) |
| **Configs** | `configs/heterogeneity/powerful_gnns/{mutag,enzymes}-gat.yaml` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/{mutag,enzymes}_gat/` |
| **Join** | `--operators GCN,GIN,SAGE,GAT` after pickles land |

```bash
HETERO_DATASETS=mutag,enzymes HETERO_MODELS=gat \
  HETERO_NUM_TASKS=2 HETERO_ARRAY=1-2 \
  HETERO_PARTITION=gpu_h200 HETERO_PARALLEL=2 \
  bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
```


## Xu-protocol SiGMA a2g4 (5 seeds, checkpoint + gates)

Match **Xu HPs** (L=4, lr=0.01, 350 ep, sum pool) with the **a2g4 mix**
(GCN,GIN,SAGE,GAT) so last-layer γ can be joined to the existing specialist
pickles. Not the old `mutag-sigma.yaml` `GIN,GIN` a2g2.

| Field | Value |
|-------|-------|
| **Jobs** | 10 (mutag seeds 0–4, enzymes seeds 0–4) |
| **Submit** | `bash bash_interface/cluster/submit_heterogeneity_xu_sigma_a2g4_ckpt.sh` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/<ds>_SiGMA_hetero_xu_seed<s>/` |
| **SLURM** | 🔄 **`44258255`** `gpu_h200` `1-10%10` (2026-09-03 ~18:54 ET) · cancelled `44229226` |

```bash
squeue -u $USER -j 44258255
head -40 logs_gnnplus/xu_sigma_a2g4_44258255_1.log   # mutag seed 0
# first lines: Training / Trial, not import errors
```

When all 10 finish, each run dir should have `ckpt/` + `gate_values_per_graph.pt`. Then pull and re-join:

```bash
rsync -avz --include='*/' --include='gate_values_per_graph.pt' --include='config_used.yaml' --exclude='*' \
  rpellegrinext@holylogin.rc.fas.harvard.edu:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/ \
  results/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/
```

---

### Live status (2026-09-03 ~14:26 ET)

| Array | Task | State | Notes |
|-------|------|-------|-------|
| **43789365** | 1 mutag_gcn | ✅ | ≥100 apps |
| **44164801** | 2 mutag_gin | ✅ | ≥100 apps |
| **44164801** | 3 mutag_sage | ✅ | ≥100 apps |
| **44164801** | 4 mutag_gatedgcn | ❌ then 🔄 **`44218244`** | LinearEdge `times_func` + 4-D edges |
| **44164801** | 5 enzymes_gcn | ✅ | ≥100 apps |
| **44164801** | 6 enzymes_gin | ✅ | ≥100 apps |
| **44164801** | 7 enzymes_sage | ✅ | ≥100 apps |
| **44164801** | 8 enzymes_gatedgcn | ❌ then 🔄 **`44218244`** | ones-edge + `times_func` |

**GatedGCN fix (synced):** yaml `posenc_RWSE.kernel.times_func` (MUTAG `range(1,5)`, ENZYMES `range(1,2)`) + `master_loader` ones `edge_attr` when `num_edge_features==0`.

```bash
squeue -u $USER -j 44218244
# first lines must NOT be LinearEdgeEncoder ValueError
head -60 logs_gnnplus/hetero_gate_bridge_44218244_4.log
head -60 logs_gnnplus/hetero_gate_bridge_44218244_8.log
```

### Task progress

| Task | Dataset | Operator | Status |
|------|---------|----------|--------|
| 1 | mutag | gcn | ✅ |
| 2 | mutag | gin | ✅ |
| 3 | mutag | sage | ✅ |
| 4 | mutag | gatedgcn | 🔄 `44218244` |
| 5 | enzymes | gcn | ✅ |
| 6 | enzymes | gin | ✅ |
| 7 | enzymes | sage | ✅ |
| 8 | enzymes | gatedgcn | 🔄 `44218244` |

**Done when** each out dir contains `*_graph_dict.pickle` + `test_appearances.csv`:
`$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/<ds>_<model>/`

**Related docs:** [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) ·
[`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) (gate dumps) ·
[`Paper_heterogeneity.md`](Paper_heterogeneity.md) (hetero protocol)

Connect **Appendix F** SiGMA gate plots (`results/gate_viz/tu_hh_hetero/`) with
**Tables 1–2** operator-preference data (per-graph GCN / GIN / SAGE / GatedGCN
heterogeneity profiles).

## Scientific goal

> Graphs on which standalone **GCN** (resp. GIN, SAGE) is the best specialist
> should receive higher **GCN** (resp. GIN, SAGE) gate mass in the trained
> SiGMA hetero model (`a2g4`: MP heads GCN,GIN,SAGE,GAT).

GatedGCN is a fourth **baseline operator** in Tables 1–2; SiGMA hetero has no
GatedGCN head (proxy: compare GatedGCN preference vs GCN gate or skip in fig).

---

## Results so far (L=12 `a2g4` dumps, 2026-09-03)

Join: Xu-style pickles (GCN/GIN/SAGE, ≥100 appearances) ↔ 5-seed
`results/tu_sigma_homo_hetero/{mutag,enzymes}_SiGMA_hetero_lr001_seed{0–4}/`.
Alignment is **TUDataset `graph_idx`**, not dump-row `i` (train loader is
shuffled). Default join uses **val+test** via GraphGym `ShuffleSplit`.

Figures: `results/heterogeneity/tu_gate_bridge_analysis/paper_figures/`  
Layer CSV: `results/heterogeneity/tu_gate_bridge_analysis/layer_delta_gamma.csv`

**Last layer (index 11):** weak / slightly **anti-aligned**. MUTAG n=184,
ENZYMES n=578 (union of val+test across seeds). |\Deltaγ| ≲ 0.03, error bars
through 0. Pearson r(Δacc, Δγ) ≈ −0.19 / −0.13 vs GIN. Last-layer GIN γ is
high regardless of specialist. SAGE same story.

**Every MP layer (0–11):** `--scan-layers` (default on). Same Δγ contrast at
each depth. Nothing interesting: |\Deltaγ| ≲ 0.03 at all layers, usually
inside 5-seed std; r vs Δacc ~ 0. ENZYMES GIN Δγ is slightly **negative**
mid-depth (L5–L8, down to −0.028 ± 0.020). MUTAG SAGE L8 +0.022 ± 0.042 is
noise. Last-layer was not hiding a shallow specialist map.

Why this is unsurprising on these dumps: Xu L=4 specialists vs SiGMA **L=12**
`a2g4`; preference ≠ graph type; GAT specialist pickles **pending** (see banner);
MUTAG ~48% ties.

**Next:** when **`44258255`** finishes, pull `tu_xu_sigma_a2g4` dumps (L=4,
same recipe as pickles) and re-join + re-scan 4 layers.

```bash
python scripts/heterogeneity/join_tu_gate_operator_preference.py \
  --datasets mutag,enzymes \
  --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \
  --gate-root results/tu_sigma_homo_hetero \
  --lr-tag lr001 \
  --seeds 0,1,2,3,4 \
  --operators GCN,GIN,SAGE \
  --splits val,test \
  --out-dir results/heterogeneity/tu_gate_bridge_analysis
```

---

## What already exists

| Artifact | Location |
|----------|----------|
| SiGMA hetero training + `gate_values_per_graph.pt` | `$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/` |
| Appendix F gate PNGs | `results/gate_viz/tu_hh_hetero/` |
| Gate batch plotter | `scripts/gate_viz/plot_tu_hh_gates_batch.py` |

## What this campaign adds

| Artifact | Location |
|----------|----------|
| Hetero profiles (4 operators × 2 datasets) | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/` |
| Join CSV + diagnostic plots | `results/heterogeneity/tu_gate_bridge_analysis/<ds>/` |

Configs: `configs/heterogeneity/powerful_gnns/{mutag,enzymes}-{gcn,gin,sage,gatedgcn}.yaml`  
(Xu et al. ICLR 2019 recipe; GatedGCN uses edge encoder + layer ffn/residual.)

---

## Launch on cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# Recommended first: pilot (10 appearances) — ~hours not days
HETERO_REQUIRED_TEST_APPEARANCES=10 HETERO_MAX_TRIALS=200 \
  bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh

# Paper target (≥100 appearances) — heavy; expect days per dataset
bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
```

### Submitted runs

| Campaign | SLURM array | Appearances | Max trials | Parallel | Logs |
|----------|-------------|-------------|------------|----------|------|
| **Pilot** | ✅ **`43789364`** | ≥10 | 200 | `1-8%4` | `logs_gnnplus/hetero_gate_bridge_43789364_<TASK>.log` |
| **Full (paper)** | ⚠️ **`43789365`** | ≥100 | 2000 | `1-8%4` | task 1 ✅; **2–8 failed** (`num_tasks=1` env bug) |
| **Full retry 2–8** | ❌ **`44100206`** | ≥100 | 2000 | `2-8%4` | same `--export` comma bug |
| **Full retry 2–8 (fixed)** | ✅ **`44164801`** | ≥100 | 2000 | `2-8%4` | GCN/GIN/SAGE ✅; gatedgcn ❌ |
| **GatedGCN retry 4,8** | 🔄 **`44218244`** | ≥100 | 2000 | `4,8%4` | `times_func` + ones-edge |

| Field | Value |
|-------|-------|
| **Partition / mem / time** | `mweber_gpu` / 64GB / 192h |
| **Scripts** | `submit_heterogeneity_tu_gate_bridge.sh` → `run_heterogeneity_tu_gate_bridge.sh` |
| **W&B groups** | `building_hetero_profile_<ds>_tu_gate_bridge` |
| **W&B run names** | `<ds>_<model>_tu_gate_bridge` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/<ds>_<model>/` |

⚠️ **Out dirs shared** under `tu_gate_bridge/`. Paper source of truth:
**`43789365_1`** (`mutag_gcn`) + **`44100206`** (tasks 2–8). Always set
`HETERO_DATASETS=mutag,enzymes` and `HETERO_MODELS=gcn,gin,sage,gatedgcn`
explicitly on submit (or `unset` them) — leftover smoke env caused
`num_tasks=1` and killed 43789365 tasks 2–8.

**Monitor:**
```bash
squeue -u $USER -j 44100206
# once running:
head -40 logs_gnnplus/hetero_gate_bridge_44100206_2.log   # expect task 2/8
tail -f logs_gnnplus/hetero_gate_bridge_44100206_2.log
```

| Field | Value |
|-------|-------|
| **Tasks** | `2-8%4` retry (+ task 1 already done) |

### Task map

| Task | Dataset | Operator |
|------|---------|----------|
| 1 | mutag | gcn |
| 2 | mutag | gin |
| 3 | mutag | sage |
| 4 | mutag | gatedgcn |
| 5 | enzymes | gcn |
| 6 | enzymes | gin |
| 7 | enzymes | sage |
| 8 | enzymes | gatedgcn |

### Smoke (login-node friendly check)

```bash
HETERO_ARRAY=1 HETERO_NUM_TASKS=1 \
  HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=10 \
  bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
```

---

## Pull results → join locally

```bash
# hetero pickles
bash bash_interface/local/pull_tu_gate_bridge_hetero.sh

# gate dumps (if not already local)
mkdir -p results/tu_sigma_homo_hetero
rsync -avz --include='*/' --include='gate_values_per_graph.pt' --exclude='*' \
  rpellegrinext@holylogin.rc.fas.harvard.edu:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/tu_sigma_homo_hetero/ \
  results/tu_sigma_homo_hetero/

# 5-seed join + last-layer paper figs + all-layer Δγ scan
python scripts/heterogeneity/join_tu_gate_operator_preference.py \
  --datasets mutag,enzymes \
  --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \
  --gate-root results/tu_sigma_homo_hetero \
  --lr-tag lr001 \
  --seeds 0,1,2,3,4 \
  --operators GCN,GIN,SAGE \
  --splits val,test \
  --out-dir results/heterogeneity/tu_gate_bridge_analysis
```

Outputs under `results/heterogeneity/tu_gate_bridge_analysis/`:

- join CSVs + `paper_figures/` (preference bars, gate-by-pref, Δγ, scatters, ranked gates)
- `layer_delta_gamma.csv` + `paper_figures/fig_delta_gamma_by_layer.png`

---

## Tables 1–2 (to fill after hetero jobs)

**Table 1 (operator preference fractions):** from join CSV — fraction of graphs
where each operator is argmax accuracy (optionally require margin > ε).

**Table 2 (gate alignment):** mean SiGMA gate on head H when operator H preferred
vs not (printed by join script; extend for paper LaTeX).

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-03 | Added Xu-recipe **GAT** specialist yamls; ready to submit 2-task bridge array |
| 2026-09-03 | Resubmitted Xu SiGMA a2g4 on **`gpu_h200`** **`44258255`**; cancelled PD **`44229226`** |
| 2026-09-03 | All 12 MP layers scanned on L=12 dumps: no routing (|\Deltaγ| ≲ 0.03, r~0) |
| 2026-09-03 | Submitted Xu L=4 SiGMA a2g4 ckpt **`44229226`** `1-10%5` (later cancelled) |
| 2026-09-03 | Submitted GatedGCN retry **`44218244`** tasks 4,8 (holylogin08) |
| 2026-09-03 | **44164801**: 6/8 ✅ (GCN/GIN/SAGE); gatedgcn ❌ (`LinearEdge` / ENZYMES no edges) |
| 2026-09-03 | Fix gatedgcn yaml `times_func` + master_loader ones-edge for empty edge_attr |
| 2026-09-03 | **44164801** running: tasks 2/3/5/6 R; verified `2×4=8` + mutag_gin Trial 1 |
| 2026-09-03 | Submitted fixed retry **44164801** tasks 2–8 (boslogin08); banner lists OK |
| 2026-09-03 | Root cause: SLURM `--export` comma-split; fixed submit → `--export=ALL` |
| 2026-09-03 | **44100206** all FAILED again (`out of range 1..1`) |
| 2026-09-02 | **44100206** confirmed PD (Priority); shell `HETERO_*` unset ✅ |
| 2026-09-02 | Resubmit **44100206** tasks 2–8 (boslogin06) |
| 2026-09-02 | **43789365**: task 1 `mutag_gcn` ✅; tasks 2–8 ❌ `out of range (1..1)` (HETERO_* env) |
| 2026-09-02 | First `squeue`: full **43789365** tasks 1,7,8 + pilot **43789364** task 1 running |
| 2026-09-02 | Submitted pilot **43789364** (≥10 app) + full **43789365** (≥100 app) on holylogin05 |
| 2026-09-01 | Added gate-bridge configs, SLURM scripts, join script, pull helper (`06c47f0`) |
