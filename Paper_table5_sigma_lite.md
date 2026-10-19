# Table 5 — SiGMA-lite (gate after concatenation)

Ablation of the Table 5 **SiGMA** row: same head mix / depth / width / LR / seeds, but
`model.type: sigma_lite` (ungated heads + one elementwise sigmoid gate on the mixed
update after `out_proj`, à la Gated-GPS).

| Field | Value |
| --- | --- |
| **Scripts** | `bash_interface/cluster/submit_paper_table5_sigma_lite.sh` · `run_paper_table5_sigma_lite.sh` |
| **Jobs** | 7 datasets × 5 seeds = **35** |
| **GPUs** | **6** max (`%6`) |
| **Dependency** | `afterany:53598773` (SiGMA/MMA 15-seed fill) |

## Recipes (frozen from Table 5 SiGMA)

| Dataset | Seeds | Config / overrides | W&B group |
| --- | --- | --- | --- |
| Peptides-func | 0–4 | o5cdk766 + a1g2 `GCN,GCN` | `paper_T5_peptides_func_SiGMA_lite` |
| Peptides-struct | 0–4 | g3bsaq32-b7m0 | `paper_T5_peptides_struct_SiGMA_lite` |
| PascalVOC-SP | 0–4 | j7ukyzdm a2g2 | `paper_T5_voc_SiGMA_lite` |
| MNIST | 0–4 | lcvbyyss a2g2 | `paper_T5_mnist_SiGMA_lite` |
| CIFAR10 | 0–4 | ulij45a2 a8g4 | `paper_T5_cifar10_SiGMA_lite` |
| PATTERN | 5–9 | ta9qtxb9 + grit + VN4 | `paper_T5_pattern_gritvn4_SiGMA_lite` |
| CLUSTER | 0–4 | ht9bntg2 a1g1 | `paper_T5_cluster_SiGMA_lite` |

## JOBIDs

| Date | JOBID | Notes |
| --- | --- | --- |
| | | paste after `sbatch` |
