# Peptides-func — MP-only on NEW best Homog_MP (a1g2 → a0g3)

Table 6 found a stronger Peptides-func model than paper a1g1 SiGMA:

| Model | Arch | AP ↑ | n | W&B |
|-------|------|------|---|-----|
| **Homog_MP (NEW best)** | a1g2 `GCN,GCN` gated | **0.7080±0.0063** | 5 | `paper_T6_peptides_func_Homog_MP` |
| Paper SiGMA | a1g1 `GCN` gated | 0.7052±0.0056 | 10 | `lr_ablation_…_b208_m0` |

This campaign is the Table-5-style **MP_only** control for that new best:
replace the 1 attention head by a GCN → **a0g3 `GCN,GCN,GCN`**, keep elementwise gating and all other HPs from o5cdk766 / Homog_MP.

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑🛑🛑  TO RUN  ·  not submitted yet  🛑🛑🛑                          ║
║  🧪  a0g3 GCN×3 gated × 5 seeds                                          ║
║  🚀  bash bash_interface/cluster/submit_peptides_func_homog_a1g2_mp_only.sh ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🛑 **TO RUN** |
| **SLURM array** | 🛑 *not submitted yet* |
| **Job name** | `sigma_func_a0g3` |
| **Tasks** | `1-5%5` |
| **Config** | `configs/gated_hybrid/peptides-func-hybrid-homog-a1g2-gcn-anchor.yaml` + CLI a0g3 |
| **W&B group** | `paper_T5_peptides_func_HomogMP_MPonly` |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_peptides_func_homog_a1g2_mp_only.sh
```

## Aggregate

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_T5_peptides_func_HomogMP_MPonly \
  --metric best_test_perf --state finished
```

### Fill-in

| Model | Arch | AP ↑ | n |
|-------|------|------|---|
| Homog_MP (SiGMA-style) | a1g2 GCN×2 | 0.7080±0.0063 | 5 |
| **MP_only** | a0g3 GCN×3 | | 5 |
