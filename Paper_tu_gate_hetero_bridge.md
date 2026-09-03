# TU gate–operator bridge (MUTAG + ENZYMES)

```text
╔══════════════════════════════════════════════════════════════════╗
║  ❌  44100206 FAILED again · same 1..1 bug · root cause found   ║
║  Cause: SLURM --export=… splits on commas → only mutag + gcn    ║
║  Fix: submit script now uses --export=ALL (shell env)           ║
║  Keep: 43789365_1 mutag_gcn ✅ · resubmit 2–8 after git sync    ║
╚══════════════════════════════════════════════════════════════════╝
```

### Live status (2026-09-03)

| Array | Task | State | Notes |
|-------|------|-------|-------|
| **43789365** | 1 | ✅ COMPLETED | `mutag_gcn` ≥100 apps on disk |
| **43789365** | 2–8 | ❌ FAILED | `out of range (1..1)` |
| **44100206** | 2–8 | ❌ FAILED | same bug — `--export` ate commas |
| **next** | 2–8 | 🛑 TO SUBMIT | after syncing fixed `submit_heterogeneity_tu_gate_bridge.sh` |

**Root cause:** `--export=ALL,...,HETERO_DATASETS=mutag,enzymes,HETERO_MODELS=gcn,gin,...`
is comma-split by SLURM → job only sees `HETERO_DATASETS=mutag` and
`HETERO_MODELS=gcn` → `num_tasks=1`.

**Fix (local):** submit script exports lists via shell + `--export=ALL` only.

### Task progress

| Task | Dataset | Operator | Status | Log | W&B name |
|------|---------|----------|--------|-----|----------|
| 1 | mutag | gcn | ✅ done (`43789365`) | `…_43789365_1.log` | `mutag_gcn_tu_gate_bridge` |
| 2 | mutag | gin | ❌ then 🛑 resubmit | — | `mutag_gin_tu_gate_bridge` |
| 3 | mutag | sage | ❌ then 🛑 resubmit | — | `mutag_sage_tu_gate_bridge` |
| 4 | mutag | gatedgcn | ❌ then 🛑 resubmit | — | `mutag_gatedgcn_tu_gate_bridge` |
| 5 | enzymes | gcn | ❌ then 🛑 resubmit | — | `enzymes_gcn_tu_gate_bridge` |
| 6 | enzymes | gin | ❌ then 🛑 resubmit | — | `enzymes_gin_tu_gate_bridge` |
| 7 | enzymes | sage | ❌ then 🛑 resubmit | — | `enzymes_sage_tu_gate_bridge` |
| 8 | enzymes | gatedgcn | ❌ then 🛑 resubmit | — | `enzymes_gatedgcn_tu_gate_bridge` |

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
| **Full retry 2–8 (fixed)** | 🛑 after syncing submit fix | ≥100 | 2000 | `2-8%4` | use `--export=ALL` only |

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
| 2026-09-03 | Root cause: SLURM `--export` comma-split; fixed submit → `--export=ALL` |
| 2026-09-03 | **44100206** all FAILED again (`out of range 1..1`) |
| 2026-09-02 | **44100206** confirmed PD (Priority); shell `HETERO_*` unset ✅ |
| 2026-09-02 | Resubmit **44100206** tasks 2–8 (boslogin06) |
| 2026-09-02 | **43789365**: task 1 `mutag_gcn` ✅; tasks 2–8 ❌ `out of range (1..1)` (HETERO_* env) |
| 2026-09-02 | First `squeue`: full **43789365** tasks 1,7,8 + pilot **43789364** task 1 running |
| 2026-09-02 | Submitted pilot **43789364** (≥10 app) + full **43789365** (≥100 app) on holylogin05 |
| 2026-09-01 | Added gate-bridge configs, SLURM scripts, join script, pull helper (`06c47f0`) |
