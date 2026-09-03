# TU gate–operator bridge (MUTAG + ENZYMES)

```text
╔══════════════════════════════════════════════════════════════════╗
║  Pref–gate join: SAGE Δγ added (L=12 a2g4, 5 seeds)             ║
║  NEXT: Xu L=4 SiGMA a2g4 × 5 seeds + ckpt + gate dump (10 jobs) ║
╚══════════════════════════════════════════════════════════════════╝
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
| **SLURM** | _paste JOBID after submit_ |

```bash
# local git first, then on cluster: git pull && bash bash_interface/cluster/submit_heterogeneity_xu_sigma_a2g4_ckpt.sh
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

Layer-wise gate analysis → later (same join script, `--gate-layer`).

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

# join + plots
python scripts/heterogeneity/join_tu_gate_operator_preference.py \
  --dataset mutag \
  --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \
  --gate-pt results/tu_sigma_homo_hetero/mutag_SiGMA_hetero_lr001_seed2/gate_values_per_graph.pt \
  --out-dir results/heterogeneity/tu_gate_bridge_analysis/mutag

python scripts/heterogeneity/join_tu_gate_operator_preference.py \
  --dataset enzymes \
  --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \
  --gate-pt results/tu_sigma_homo_hetero/enzymes_SiGMA_hetero_lr001_seed2/gate_values_per_graph.pt \
  --out-dir results/heterogeneity/tu_gate_bridge_analysis/enzymes
```

Outputs per dataset:

- `<ds>_operator_gate_join.csv` — graph_idx, preferred operator, accuracies, gates
- `<ds>_preferred_operator_gate_boxplot.png`
- `<ds>_operator_acc_vs_sigma_gate_scatter.png`

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
