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
| **SLURM job** | `28147236` |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_hybrid_o5cdk766_a1g1_paper_repro.sh
```

### Peptides-func 3g180qle a1g8 (GCN baseline → hybrid 1 attn + 8× GCN)

| Field | Value |
|-------|-------|
| **MP baseline** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3g180qle |
| **W&B group** | `paper_bestmodel_v2_peptides_func_3g180qle_a1g8_ep300` |
| **Config** | `configs/gated_hybrid/peptides-func-hybrid-3g180qle-a1g8-anchor.yaml` |
| **Architecture** | a1g8: 1 attn + 8× GCN, d_h=64, outer = gcn/peptides-func.yaml, ep=300 |
| **Submit** | `bash bash_interface/cluster/submit_peptides_func_hybrid_3g180qle_a1g8_paper_repro.sh` |
| **Seeds** | 0–4 (task_id 1–5) |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_hybrid_3g180qle_a1g8_paper_repro.sh
```

---

## UniGCN campaign (Jul 8–9 2026) — IDs cheat sheet

Entity/project: `weber-geoml-harvard-university/GNNPlus`  
Branch: `paper_repro_m04v86sm`

| Kind | SLURM / W&B ID | Name / group | Notes |
|------|----------------|--------------|-------|
| **W&B sweep** | `bq62chmz` | `GNNplus_hybrid_unigcn-peptides_func-val_ap` | [sweep](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz) |
| Sweep agents (1 GPU) | `29513197` | `gnnplus_sweep_peptides_func` | `%1`, 10 agents × 4 runs |
| Sweep agents (5 GPU) | `29854262` | same sweep | 20×4, `%5` |
| Sweep agents (5 GPU) | `29859293` | same sweep | 20×4, `%5` (2nd relaunch) |
| UniGCN few-runs (old) | `29436702` | `unigcn_hybrid` | pre–SiGMA-lock submit |
| UniGCN few-runs | `29466653` | `unigcn_hybrid_few_runs_*` | 36 tasks, `%5`, SiGMA lock |
| PS baseline vs hybrid | `29874107` | `peptides_struct_unigcn_baseline_vs_hybrid` | custom + a1g2 GINE+UNIGCN |
| PS paper UniGCN | `29874935` | `peptides_struct_unigcn_paper` | arXiv:2410.05499 params |
| PS hybrid v2 | `29874936` | `peptides_struct_unigcn_hybrid_v2` | a2g2 L8 ep300 |
| Anchor run (hybrid) | `y3ygn39y` | 63avcc5m a1g1 GINE | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/y3ygn39y) |
| PF UniGCN seed×LR | `29933958` | `peptides_func_2i5psq22_a5g3_lr_seeds_{b455,b45,b5}` | [2i5psq22](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2i5psq22) × 3 LR × 10 seeds, `%2` |
| PF UniGCN 10 seeds | *(fill SLURM after submit)* | `peptides_func_124caj93_a2g3_seeds` | [124caj93](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/124caj93) a2g3 × 10 seeds, `%2` |
| PF UniGCN high LR | `30132672` | `peptides_func_2i5psq22_a5g3_lr_high_{b5,b7,b9}` | [fpfl6ve9](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fpfl6ve9) × LR 5e-4/7e-4/9e-4 × 10 seeds, `%5` |

### Peptides-func UniGCN: 2i5psq22 a5g3 high-LR × 10 seeds

Tracked: Jul 11 2026 · branch `paper_repro_m04v86sm`

| Field | Value |
|-------|-------|
| **Anchor run** | [fpfl6ve9](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fpfl6ve9) (seed0 @ lr=5e-4 from job `29933958`) |
| **Sweep source** | [2i5psq22](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2i5psq22) / [bq62chmz](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz) |
| **SLURM array** | `30132672` |
| **Job name** | `pf_2i5_lrhigh` |
| **Config** | `configs/gated_hybrid/peptides-func-hybrid-2i5psq22-a5g3-unigcn-anchor.yaml` |
| **Run / submit** | `run_` / `submit_peptides_func_hybrid_2i5psq22_a5g3_lr_high_seeds.sh` |
| **Arch** | a5g3 = 5×attn + `GINE,GCNE,UNIGCN`, d_h=128, T=10, headwise, LN, full mask, ep=300 |
| **Tasks** | `1-30%5` = 3 LR × 10 seeds, max **5 GPUs** |
| **Logs** | `logs_gnnplus/pf_2i5_lrhigh_30132672_<TASK>.log` |

| Tasks | Seeds | `optim.base_lr` | W&B group | Status | `best_test_perf` (AP ↑) |
|-------|-------|-----------------|-----------|--------|-------------------------|
| 1–10 | 0–9 | `0.0005` | `peptides_func_2i5psq22_a5g3_lr_high_b5` | **10/10** finished | **0.7039 ± 0.0062** |
| 11–20 | 0–9 | `0.0007` | `peptides_func_2i5psq22_a5g3_lr_high_b7` | **10/10** finished | **0.7040 ± 0.0059** |
| 21–30 | 0–9 | `0.0009` | `peptides_func_2i5psq22_a5g3_lr_high_b9` | **10/10** finished | **0.7000 ± 0.0062** |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_hybrid_2i5psq22_a5g3_lr_high_seeds.sh
```

```bash
for g in b5 b7 b9; do
  python scripts/api_wanndb_query/aggregate_paper_repro.py \
    --group peptides_func_2i5psq22_a5g3_lr_high_${g} --metric best_test_perf
done
```

### Peptides-func UniGCN: 124caj93 a2g3 × 10 seeds

Tracked: Jul 10 2026 · branch `paper_repro_m04v86sm`

| Field | Value |
|-------|-------|
| **Source run** | [124caj93](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/124caj93) |
| **Sweep** | [bq62chmz](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz) |
| **SLURM array** | *(paste JOBID after submit)* |
| **Job name** | `pf_124caj93_seeds` |
| **Config** | `configs/gated_hybrid/peptides-func-hybrid-124caj93-a2g3-unigcn-anchor.yaml` |
| **Run / submit** | `run_` / `submit_peptides_func_hybrid_124caj93_a2g3_seeds.sh` |
| **Arch** | a2g3 = 2×attn + `GINE,GCNE,UNIGCN`, d_h=128, T=16, headwise, LN, full mask, Atom+RWSE, ep=300 |
| **LR** | `0.0005650212198206989` (sweep exact) |
| **Tasks** | `1-10%2` seeds 0–9, max **2 GPUs** |
| **W&B group** | `peptides_func_124caj93_a2g3_seeds` |
| **Logs** | `logs_gnnplus/pf_124caj93_seeds_<JOBID>_<TASK>.log` |

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_hybrid_124caj93_a2g3_seeds.sh
```

Aggregate:

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group peptides_func_124caj93_a2g3_seeds --metric best_test_perf
```

### Peptides-func UniGCN: 2i5psq22 a5g3 × 3 LR × 10 seeds

Tracked: Jul 10 2026 · branch `paper_repro_m04v86sm`

| Field | Value |
|-------|-------|
| **Source run** | [2i5psq22](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2i5psq22) |
| **Sweep** | [bq62chmz](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz) |
| **SLURM array** | `29933958` |
| **Job name** | `pf_2i5psq22_lr` |
| **Config** | `configs/gated_hybrid/peptides-func-hybrid-2i5psq22-a5g3-unigcn-anchor.yaml` |
| **Run script** | `bash_interface/cluster/run_peptides_func_hybrid_2i5psq22_a5g3_lr_seeds.sh` |
| **Submit script** | `bash_interface/cluster/submit_peptides_func_hybrid_2i5psq22_a5g3_lr_seeds.sh` |
| **Arch** | a5g3 = 5×attn + `GINE,GCNE,UNIGCN`, d_h=128, T=10, headwise, LN, full mask, Atom+RWSE, ep=300 |
| **Sweep exact LR** | `0.0004546350916048615` (grid uses rounded LRs below) |
| **Tasks** | `1-30%2` = 3 LR × 10 seeds (0–9), max **2 GPUs** |
| **Logs** | `logs_gnnplus/pf_2i5psq22_lr_29933958_<TASK>.log` |
| **Entity/project** | `weber-geoml-harvard-university/GNNPlus` |

| Tasks | Seeds | `optim.base_lr` | W&B group | Status | `best_test_perf` (AP ↑) |
|-------|-------|-----------------|-----------|--------|-------------------------|
| 1–10 | 0–9 | `0.000455` | `peptides_func_2i5psq22_a5g3_lr_seeds_b455` | **10/10** finished | **0.6943 ± 0.0063** |
| 11–20 | 0–9 | `0.00045` | `peptides_func_2i5psq22_a5g3_lr_seeds_b45` | **10/10** finished | **0.7002 ± 0.0056** |
| 21–30 | 0–9 | `0.0005` | `peptides_func_2i5psq22_a5g3_lr_seeds_b5` | **10/10** finished | **0.7033 ± 0.0064** |

Partial per-seed (`_b455`, metric=`best_test_perf`):

| seed | AP | run | state |
|------|-----|-----|-------|
| 0 | 0.6957 | `oajamg1b` | finished |
| 1 | 0.6992 | `gfjwknse` | finished |
| 2 | 0.6952 | `3kph2iff` | finished |
| 3 | 0.6941 | `5no3qi0b` | finished |
| 4 | 0.6935 | `6ensjgld` | finished |
| 5 | 0.6971 | `rl3oqz9x` | finished |
| 6 | 0.6949 | `zsqmnihz` | finished |
| 7 | 0.6815 | `lgpe1ewd` | finished |
| 8 | 0.6874 | `b4al0umv` | finished |
| 9 | 0.7049 | `leb2qw4o` | finished |

Re-aggregate when more finish:

```bash
for g in b455 b45 b5; do
  python scripts/api_wanndb_query/aggregate_paper_repro.py \
    --group peptides_func_2i5psq22_a5g3_lr_seeds_${g} --metric best_test_perf
done
```

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_hybrid_2i5psq22_a5g3_lr_seeds.sh
# Then paste ARRAY JOBID into the SLURM row above and the cheat-sheet table.
```
### Peptides-func UniGCN hybrid W&B sweep

| Field | Value |
|-------|-------|
| **Sweep ID** | `bq62chmz` |
| **Full path** | `weber-geoml-harvard-university/GNNPlus/bq62chmz` |
| **YAML** | `bash_interface/sweeps/peptides_func_hybrid_unigcn_sweep.yaml` |
| **W&B** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/bq62chmz |
| **Few-run arrays** | `29436702` (early), `29466653` (SiGMA-locked, `%5`) |
| **Agent arrays** | `29513197` (`%1`); `29854262`, `29859293` (`%5`, 20×4 each) |

Relaunch sweep agents (example 5 GPUs):

```bash
SWEEP_SLURM_TIME=240:00:00 SWEEP_ARRAY_TASKS=20 SWEEP_ARRAY_PARALLEL=5 RUNS_PER_AGENT=4 \
  bash bash_interface/sweeps/relaunch_sweep_agents.sh \
    peptides_func weber-geoml-harvard-university/GNNPlus/bq62chmz
```

### Peptides-struct UniGCN: custom_gnn vs hybrid (y3ygn39y + UNIGCN)

| Field | Value |
|-------|-------|
| **SLURM array** | `29874107` |
| **W&B group** | `peptides_struct_unigcn_baseline_vs_hybrid` |
| **Tasks** | 1–6 (2 variants × 3 seeds), parallel=5 |
| **(A) custom** | `configs/gcn/peptides-struct-unigcn.yaml` (`custom_gnn` + `unitarygcn`, GNNPlus outer hparams) |
| **(B) hybrid** | `configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a1g2-gine-unigcn.yaml` |
| **Source run** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/y3ygn39y |
| **Logs** | `logs_gnnplus/ps_unigcn_29874107_<TASK>.log` |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_unigcn_baseline_vs_hybrid.sh` |

### Peptides-struct UniGCN paper baseline (custom_gnn)

| Field | Value |
|-------|-------|
| **SLURM array** | `29874935` |
| **Config** | `configs/gcn/peptides-struct-unigcn-paper.yaml` |
| **Source** | arXiv:2410.05499 / `peptides-struct-UnitaryGCN-final.yaml` |
| **Key params** | Atom+LapPE, L8, H160, residual=True, drop=0.1, bs=200, ep=250, T=16 |
| **W&B group** | `peptides_struct_unigcn_paper` |
| **Logs** | `logs_gnnplus/ps_unigcn_paper_29874935_<TASK>.log` |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_unigcn_paper.sh` |

### Peptides-struct UniGCN hybrid v2 (best hybrid + UniGCN)

| Field | Value |
|-------|-------|
| **SLURM array** | `29874936` |
| **Config** | `configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a2g2-gine-unigcn-v2.yaml` |
| **Anchor** | [y3ygn39y](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/y3ygn39y) (63avcc5m a1g1 GINE) |
| **Upgrades** | a2g2 (2×attn + GINE + UNIGCN), L8, ep=300, lr=7e-4 |
| **W&B group** | `peptides_struct_unigcn_hybrid_v2` |
| **Logs** | `logs_gnnplus/ps_unigcn_v2_29874936_<TASK>.log` |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_unigcn_hybrid_v2.sh` |

---


## Phase-2 sweeps — tfeksgbl / rholn782 backbone ([tfeksgbl](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tfeksgbl))

Anchor: `configs/gated_hybrid/peptides-struct-hybrid-rholn782-anchor.yaml` (a2g2 GINE+GGNN, L12/H96, VN=4)

| Sweep | YAML | Swept |
|-------|------|-------|
| **A reg** | `peptides_struct_hybrid_tfeksgbl_sweep_a_reg.yaml` | attn_mask, norm, dropouts, gate, batch |
| **B scale** | `peptides_struct_hybrid_tfeksgbl_sweep_b_scale.yaml` | layers_mp, dim_inner, d_h, readout, ffn, residual |
| **C vn** | `peptides_struct_hybrid_tfeksgbl_sweep_c_vn.yaml` | add_virtual_nodes, num_vn, readout, ffn |

**12 GPUs max** (4 per sweep):

```bash
bash bash_interface/sweeps/launch_peptides_struct_tfeksgbl_phase2_sweeps.sh --create
# relaunch only:
bash bash_interface/sweeps/launch_peptides_struct_tfeksgbl_phase2_sweeps.sh
```
