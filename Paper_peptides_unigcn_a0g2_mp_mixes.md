# Peptides UniGCN a0g2 MP mixes (no attention)

Take **best SiGMA HPs** so far, drop attention, run two gated MP mixes:

| Variant | Arch | `gnn_types` |
|---------|------|-------------|
| `UNIGCN_GINE` | a0g2 | `UNIGCN,GINE` |
| `UNIGCN_GATEDGCN` | a0g2 | `UNIGCN,GATEDGCN` |

### HP sources (heads only change)

| Dataset | Best SiGMA lineage | Anchor yaml | Notes |
|---------|-------------------|-------------|-------|
| Peptides-func | Homog_MP a1g2 `GCN,GCN` → AP **0.7080±0.0063** (n=5) | `peptides-func-hybrid-homog-a1g2-gcn-anchor.yaml` (o5cdk766 HPs) | Paper a1g1 was 0.7052 (n=10) |
| Peptides-struct | a1g1 `GINE` g3bsaq32 → MAE **0.2441±0.0017** (n=10) | `peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml` | Table 6 Homog_MP ≈ same MAE |

Gate / depth / LR / RWSE / epochs left as in those anchors; CLI forces `num_attn_heads=0`, `num_gnn_heads=2`.

Related chat context: UniGCN was explored earlier on peptides but **not** as paper-best SiGMA; this is a clean no-attn UniGCN+partner MP control on current best HPs. Also still 🛑: Homog→MP_only a0g3 GCN×3 (`Paper_peptides_func_homog_a1g2_mp_only.md`).

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑🛑🛑  TO RUN  ·  not submitted yet  🛑🛑🛑                          ║
║  🧪  2 ds × 2 mixes × 5 seeds = 20 jobs                                  ║
║  🚀  bash bash_interface/cluster/submit_peptides_unigcn_a0g2_mp_mixes.sh ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🛑 **TO RUN** |
| **SLURM array** | 🛑 *not submitted yet* |
| **Job name** | `pep_unigcn_a0g2` |
| **Tasks** | `1-20%8` |
| **W&B groups** | `paper_peptides_{peptides_func,peptides_struct}_a0g2_{UNIGCN_GINE,UNIGCN_GATEDGCN}` |
| **Tags** | `peptides_unigcn_a0g2`, `<Variant>`, `<ds>`, `seed<k>`, `a0g2`, `no_attn` |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_peptides_unigcn_a0g2_mp_mixes.sh
# 👉 paste JOBID here + CLUSTER_LAUNCHES.md
```

## Aggregate

```bash
for ds in peptides_func peptides_struct; do
  for v in UNIGCN_GINE UNIGCN_GATEDGCN; do
    echo "===== ${ds} / ${v} ====="
    python scripts/api_wanndb_query/aggregate_paper_repro.py \
      --group paper_peptides_${ds}_a0g2_${v} --metric best_test_perf --state finished
  done
done
```

### Fill-in

| Mix | Peptides-func ↑ | Peptides-struct ↓ | n |
|-----|-----------------|-------------------|---|
| UNIGCN+GINE a0g2 | | | 5 |
| UNIGCN+GATEDGCN a0g2 | | | 5 |
| SiGMA best (ref) | 0.7080±0.0063 (Homog a1g2) | 0.2441±0.0017 (a1g1) | 5 / 10 |
