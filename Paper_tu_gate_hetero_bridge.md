# TU gate–operator bridge (MUTAG + ENZYMES)

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

| Field | Value |
|-------|-------|
| **SLURM array** | _(paste after submit)_ |
| **Tasks** | `1-8%4` (mutag/enzymes × gcn/gin/sage/gatedgcn) |
| **Scripts** | `submit_heterogeneity_tu_gate_bridge.sh` → `run_heterogeneity_tu_gate_bridge.sh` |
| **Logs** | `logs_gnnplus/hetero_gate_bridge_<JOBID>_<TASK>.log` |
| **W&B groups** | `building_hetero_profile_<ds>_tu_gate_bridge` |
| **Outs** | `$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/<ds>_<model>/` |

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
| 2026-09-01 | Added gate-bridge configs, SLURM scripts, join script, pull helper |
