# July 2026 W&B sweeps (GNNPlus)

Entity/project: `weber-geoml-harvard-university/GNNPlus`

Cluster repo: `/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus`  
Branch: `paper_repro_m04v86sm`

---

## Active / use these

### Peptides-func gated MP sweep

| Field | Value |
|-------|-------|
| **Sweep ID** | `nuvkhnfr` |
| **Full path** | `weber-geoml-harvard-university/GNNPlus/nuvkhnfr` |
| **YAML** | `bash_interface/sweeps/peptides_func_hybrid_gated_mp_sweep.yaml` |
| **Metric** | maximize `best/val_ap` |
| **W&B** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/nuvkhnfr |
| **SLURM jobs** | `27867941` (1st launch); `28043288`, `28043588` (relaunches ×2) |
| **GPU cap** | 4 parallel per launch (`SWEEP_ARRAY_PARALLEL=4`); 8 total with struct pair |

Relaunch:

```bash
SWEEP_SLURM_TIME=240:00:00 SWEEP_ARRAY_TASKS=16 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=3 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    peptides_func weber-geoml-harvard-university/GNNPlus/nuvkhnfr
```

Or both func + struct (8 GPUs total):

```bash
bash bash_interface/sweeps/launch_peptides_gated_mp_sweeps.sh
```

---

### Peptides-struct gated MP sweep

| Field | Value |
|-------|-------|
| **Sweep ID** | `dz1j1uu7` |
| **Full path** | `weber-geoml-harvard-university/GNNPlus/dz1j1uu7` |
| **YAML** | `bash_interface/sweeps/peptides_struct_hybrid_gated_mp_sweep.yaml` |
| **Metric** | minimize `best/val_mae` |
| **W&B** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/dz1j1uu7 |
| **SLURM jobs** | `27867942` (1st launch); `28043289`, `28043610` (relaunches ×2) |

Relaunch:

```bash
SWEEP_SLURM_TIME=240:00:00 SWEEP_ARRAY_TASKS=16 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=3 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    peptides_struct weber-geoml-harvard-university/GNNPlus/dz1j1uu7
```

---

### Peptides-struct rholn782 MP sweep (LR × attn × MP presets × VN on/off)

| Field | Value |
|-------|-------|
| **Sweep ID** | **`o8wijtq7`** ← **current** |
| **Full path** | `weber-geoml-harvard-university/GNNPlus/o8wijtq7` |
| **YAML** | `bash_interface/sweeps/peptides_struct_hybrid_rholn782_mp_sweep.yaml` |
| **Anchor** | `configs/gated_hybrid/peptides-struct-hybrid-rholn782-anchor.yaml` |
| **Metric** | minimize `best/val_mae` |
| **W&B name** | `GNNplus_hybrid_rholn782-peptides_struct-lr_attn_mp_vn` |
| **W&B** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/o8wijtq7 |
| **SLURM job** | `28079484` (16 tasks, `%4` parallel, 3 runs/agent) |
| **Swept** | `optim.base_lr`; `hybrid_num_attn_heads` ∈ {0,1,2,4,8}; `add_virtual_nodes` ∈ {true,false}; `hybrid_gnn_types` ∈ {GINE,GGNN · GINE,GINE,GGNN · GINE×7,GGNN · GINE×4,GGNN×4} |

Relaunch:

```bash
SWEEP_SLURM_TIME=240:00:00 SWEEP_ARRAY_TASKS=16 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=3 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    peptides_struct weber-geoml-harvard-university/GNNPlus/o8wijtq7
```

Logs: `logs_gnnplus/sweep_agent_28079484_<TASK>.log`

---

## Superseded / do not use

| Sweep ID | Why |
|----------|-----|
| `45eazr1b` | rholn782 v1 — all runs failed (`KeyError: dataset.add_virtual_nodes` before VN code on branch) |
| `ggxkyl39` | rholn782 — created before successful `git pull` (no `add_virtual_nodes` sweep param) |

---

## Quick reference — relaunch commands

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Gated MP pair (8 GPUs = 4 func + 4 struct)
FUNC_SWEEP_ID=weber-geoml-harvard-university/GNNPlus/nuvkhnfr \
STRUCT_SWEEP_ID=weber-geoml-harvard-university/GNNPlus/dz1j1uu7 \
  bash bash_interface/sweeps/launch_peptides_gated_mp_sweeps.sh

# rholn782 MP + VN sweep
SWEEP_SLURM_TIME=240:00:00 SWEEP_ARRAY_TASKS=16 SWEEP_ARRAY_PARALLEL=4 RUNS_PER_AGENT=3 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    peptides_struct weber-geoml-harvard-university/GNNPlus/o8wijtq7
```

Monitor: `squeue -u $USER | grep gnnplus_sweep`

---

## Paper repro cohorts (5 seeds)

### Peptides-struct bvw3v272 a2g3 (from sweep dz1j1uu7)

| Field | Value |
|-------|-------|
| **Source run** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/bvw3v272 |
| **W&B group** | `paper_bestmodel_v2_peptides_struct_bvw3v272_a2g3_ep250` |
| **Config** | `configs/gated_hybrid/peptides-struct-hybrid-bvw3v272-a2g3-anchor.yaml` |
| **Architecture** | a2g3: 2 attn + 3× GINE, d_h=64, lr≈4.53e-4, ep=250 |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_hybrid_bvw3v272_a2g3_paper_repro.sh` |
| **Seeds** | 0–4 (task_id 1–5) |
| **SLURM job** | `28133112` |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_struct_hybrid_bvw3v272_a2g3_paper_repro.sh
```

### Peptides-struct 48z1z9zi a1g1 (from sweep dz1j1uu7)

| Field | Value |
|-------|-------|
| **Source run** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/48z1z9zi |
| **W&B group** | `paper_bestmodel_v2_peptides_struct_48z1z9zi_a1g1_ep400` |
| **Config** | `configs/gated_hybrid/peptides-struct-hybrid-48z1z9zi-a1g1-anchor.yaml` |
| **Architecture** | a1g1: 1 attn + 1× GINE, d_h=64, lr≈5.89e-4, ep=400 |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_hybrid_48z1z9zi_a1g1_paper_repro.sh` |
| **Seeds** | 0–4 (task_id 1–5) |
| **SLURM job** | `28133099` |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_struct_hybrid_48z1z9zi_a1g1_paper_repro.sh
```

### Peptides-struct fn30nnxg a8g3 (from sweep o8wijtq7 / rholn782 lineage)

| Field | Value |
|-------|-------|
| **Source run** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fn30nnxg |
| **W&B group** | `paper_bestmodel_v2_peptides_struct_fn30nnxg_a8g3_ep250` |
| **Config** | `configs/gated_hybrid/peptides-struct-hybrid-fn30nnxg-a8g3-anchor.yaml` |
| **Architecture** | a8g3: 8 attn + GINE,GINE,GGNN, d_h=16, VN=4, lr≈2.02e-4, ep=250 |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_hybrid_fn30nnxg_a8g3_paper_repro.sh` |
| **Seeds** | 0–4 (task_id 1–5) |
| **SLURM job** | `28145547` |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_struct_hybrid_fn30nnxg_a8g3_paper_repro.sh
```

### Peptides-func o5cdk766 a1g1 (from sweep nuvkhnfr / zc371e1n lineage)

| Field | Value |
|-------|-------|
| **Source run** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o5cdk766 |
| **W&B group** | `paper_bestmodel_v2_peptides_func_o5cdk766_a1g1_ep900` |
| **Config** | `configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml` |
| **Architecture** | a1g1: 1 attn + 1× GCN, d_h=128, graph_restricted, lr≈2.08e-4, ep=900 |
| **Submit** | `bash bash_interface/cluster/submit_peptides_func_hybrid_o5cdk766_a1g1_paper_repro.sh` |
| **Seeds** | 0–4 (task_id 1–5) |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_hybrid_o5cdk766_a1g1_paper_repro.sh
```
