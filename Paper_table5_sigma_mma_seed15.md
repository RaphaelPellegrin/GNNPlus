# Table 5 — SiGMA vs MMA, 15 new seeds (power fill)

Extend the Table 5 SiGMA / MMA (fully ungated) cohorts for Welch / paired tests.

| Field | Value |
| --- | --- |
| **Scripts** | `bash_interface/cluster/submit_paper_table5_sigma_mma_seed15.sh` · `run_paper_table5_sigma_mma_seed15.sh` |
| **Jobs** | 7 datasets × 2 variants × 15 seeds = **210** |
| **GPUs** | **6** max (`%6`) |
| **Dependency** | `afterany:53595181,53595192` (TU MMA + SiGMA-lite) |

## MMA = SiGMA − gates

Verified against W&B Reproduce args and `hybrid_gnn.py`:

- Same cfg / head counts / types / PE / LR / epochs as SiGMA.
- Only override: `gnn.hybrid.gate none` (empty `mp_gate` inherits → MP ungated too).

**Peptides-func fix:** Table 5 SiGMA is Homog_MP **a1g2** (`GCN,GCN`). Historical `paper_T5_peptides_func_SiGMA_ungated` seeds 0–4 were **a1g1** + `gate=none`. This fill runs **a1g2 + gate=none** for MMA so it matches SiGMA.

## Seed blocks (new)

| Dataset | Seeds | SiGMA W&B group | MMA W&B group |
| --- | --- | --- | --- |
| Peptides-func | 5–19 | `paper_T6_peptides_func_Homog_MP` | `paper_T5_peptides_func_SiGMA_ungated` |
| Peptides-struct | 5–19 | `paper_T5_peptides_struct_SiGMA` | `paper_T5_peptides_struct_SiGMA_ungated` |
| PascalVOC-SP | 5–19 | `paper_T5_voc_SiGMA` | `paper_T5_voc_SiGMA_ungated` |
| MNIST | 5–19 | `paper_T5_mnist_SiGMA` | `paper_T5_mnist_SiGMA_ungated` |
| CIFAR10 | 5–19 | `paper_T5_cifar10_SiGMA` | `paper_T5_cifar10_SiGMA_ungated` |
| PATTERN | 25–39 | `paper_T5_pattern_gritvn4_SiGMA` | `paper_T5_pattern_gritvn4_SiGMA_ungated` |
| CLUSTER | 20–34 | `paper_T5_cluster_SiGMA` | `paper_T5_cluster_SiGMA_ungated` |

## JOBIDs

| Date | JOBID | Notes |
| --- | --- | --- |
| | | paste after `sbatch` |
