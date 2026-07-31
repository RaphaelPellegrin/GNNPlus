# ENZYMES — port of Heterogeneity_Profile best HybridGated (`ogpkubk9`)

Source run: [`ogpkubk9`](https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/ogpkubk9)  
Project: `MOE_6` · best/test_acc ≈ **0.607** · a4g4 `GCN,GIN,SAGE,GAT` · plateau · L12 / H64 / dh16

---

## Frozen architecture

| Knob | Value |
|------|-------|
| Model | `hybrid_gnn` (SiGMA) |
| Heads | a4g4 — `GCN,GIN,SAGE,GAT` |
| `d_h` | 16 |
| Gate / norm / mask | headwise / layernorm / full |
| Depth / width | L=12, H=64 |
| FFN + residual | True |
| LR | 0.001 |
| WD | 0 |
| Batch | 64 |
| Split | random 50/25/25 (HP-style, not 10-fold) |

Configs:

- Plateau (source scheduler): `configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml`
- Cosine (non-plateau): `configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-cosine-anchor.yaml`

---

## 1. Seed grids (10 jobs)

```text
╔══════════════════════════════════════════════════════════════════╗
║  ✅  RELAUNCH  ·  SLURM 34081517  ·  2026-07-22  ·  %5           ║
║  ❌ 34076119 plateau: ReduceLROnPlateau _last_lr (ck2dwdc7)      ║
║  ❌ 34070247 LinearEdge; 33651466 inode quota                    ║
║  🧬 ENZYMES ogpkubk9 · plateau×5 + cosine×5 = 10 jobs            ║
║  📒 also listed in CLUSTER_LAUNCHES.md                           ║
╚══════════════════════════════════════════════════════════════════╝
```

5 seeds × {plateau, cosine}:

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_enzymes_ogpkubk9_seed_grids.sh
```

| Field | Value |
|-------|-------|
| **SLURM array** | ✅ **`34081517`** (scheduler + out_dir fixes). Priors: ❌ `34076119` `_last_lr`; ❌ `34070247` edge_encoder; `33651466` inode |
| **Tasks** | `1-10%5` |
| **W&B** | `enzymes_ogpkubk9_a4g4_plateau_seeds` / `enzymes_ogpkubk9_a4g4_cosine_seeds` |
| **Logs** | `logs_gnnplus/enz_ogpkubk9_34081517_<TASK>.log` |

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group enzymes_ogpkubk9_a4g4_plateau_seeds --metric best_test_perf --state finished
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group enzymes_ogpkubk9_a4g4_cosine_seeds --metric best_test_perf --state finished
```

---

## 2. Centered sweep (lr × #gates × d_h)

```text
╔══════════════════════════════════════════════════════════════════╗
║  🛑🛑🛑  TO RUN  ·  W&B sweep (create + agents)  🛑🛑🛑          ║
║  🎯 lr · num_attn/MP heads · d_h  around ogpkubk9                ║
╚══════════════════════════════════════════════════════════════════╝
```

Varies around ogpkubk9:

- `optim.base_lr` ∈ log-uniform [2e-4, 5e-3]
- `num_attn_heads` ∈ {2,4,6,8} (gated attn heads)
- `num_gnn_heads` ∈ {2,4,6,8} (gated MP heads)
- `d_h` ∈ {8,16,32,64}

YAML: `bash_interface/sweeps/enzymes_ogpkubk9_centered_sweep.yaml`

```bash
source ~/.gnnplus_env
export WANDB_PROJECT=GNNPlus
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# 🚀🚀🚀 CREATE SWEEP then paste printed sbatch agents
bash bash_interface/sweeps/create_sweep.sh \
  bash_interface/sweeps/enzymes_ogpkubk9_centered_sweep.yaml
```

| Field | Value |
|-------|-------|
| **Sweep ID** | 🛑 *TO RUN — not created* |
| **Agent job** | 🛑 *TO RUN — not submitted* |

---

## 3. Gate-viz (per-graph / per-head γ from checkpoint)

W&B `gates/...` during training is **batch-mean** only. Prior seed grids used
`train.enable_ckpt: False`, so those runs cannot be reloaded for per-graph gates.

**Yes — saving the model checkpoint is enough.** Gates are computed from the
forward pass (learned projections + activations); reload ckpt → forward → dump.

```text
╔══════════════════════════════════════════════════════════════════╗
║  ✅  SUBMITTED  ·  SLURM 36148089  ·  2026-07-29                 ║
║  plateau · seed 2 · ckpt every 50 ep                             ║
║  out: …/gate_viz_enzymes_ogpkubk9_plateau_seed2                  ║
║  log: logs_gnnplus/enz_gate_viz_36148089.log                     ║
╚══════════════════════════════════════════════════════════════════╝
```

```bash
# ✅ already submitted — do not re-run unless re-launching
# bash bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | ✅ **`36148089`** |
| **Submit** | `bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh` |
| **Config** | `enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml` |
| **out_dir** | `$GNNPLUS_OUT_DIR/gate_viz_enzymes_ogpkubk9_plateau_seed2` |
| **Ckpt** | `train.enable_ckpt True` · `ckpt_clean False` · period 50 |
| **W&B** | `enzymes_ogpkubk9_gate_viz` / `enzymes_gate_viz_plateau_seed2` |
| **Logs** | `logs_gnnplus/enz_gate_viz_36148089.log` |
| **Dump** | `submit_dump_enzymes_ogpkubk9_gates.sh` → `gate_values_per_graph.pt` |

```bash
# after training ckpts exist (preferred — uses common_env / gnnplus on GPU):
bash bash_interface/cluster/submit_dump_enzymes_ogpkubk9_gates.sh
# optional: GATE_DUMP_EPOCH=999 bash bash_interface/cluster/submit_dump_enzymes_ogpkubk9_gates.sh
```

Output tensors: `attn` `[N, L, Na]`, `gnn` `[N, L, Ng]` (per-graph mean of node γ).
