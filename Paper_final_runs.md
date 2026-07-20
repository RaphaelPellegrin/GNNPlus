# SiGMA paper final numbers (Tables III & IV)

Source draft: `SiGMA__LoG_2026_.pdf`  
Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)

**Table 5 ablations (SiGMA / SiGMA_ungated / Attn_only / MP_only):**
- LRGB: ✅ submitted (`32232124`) — [`Paper_ablations.md`](Paper_ablations.md)
- MNIST + CIFAR10: 🛑 **TO RUN** — [`Paper_ablations_mnist_cifar.md`](Paper_ablations_mnist_cifar.md) / [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)  
  `bash bash_interface/cluster/submit_paper_table5_mnist_cifar_ablations.sh`
- Peptides-func **Homog_MP → MP_only** (a0g3 GCN×3): 🛑 **TO RUN** — [`Paper_peptides_func_homog_a1g2_mp_only.md`](Paper_peptides_func_homog_a1g2_mp_only.md)  
  `bash bash_interface/cluster/submit_peptides_func_homog_a1g2_mp_only.sh`
- Peptides **UniGCN a0g2 mixes** (UNIGCN+GINE / UNIGCN+GATEDGCN, no attn): 🛑 **TO RUN** — [`Paper_peptides_unigcn_a0g2_mp_mixes.md`](Paper_peptides_unigcn_a0g2_mp_mixes.md)  
  `bash bash_interface/cluster/submit_peptides_unigcn_a0g2_mp_mixes.sh`

**SiGMA + GRIT attention** (PATTERN / CLUSTER, `attn_type=grit`): ✅ **`33458567`** — [`Paper_sigma_grit_attn.md`](Paper_sigma_grit_attn.md) / [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)  
(PATTERN finished ≈87.11%; CLUSTER still running as of last check.)

Recomputed with:

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group <W&B_GROUP> --metric best_test_perf --state finished
```

For CIFAR10 / MNIST / PATTERN / CLUSTER, multiply API fraction by 100 to match paper `%`.

---

## Table III — Dwivedi et al. benchmarks

| Dataset | Metric | Paper SiGMA | W&B recompute | Seeds (n) | W&B group / runs | Match? |
|---------|--------|-------------|---------------|-----------|------------------|--------|
| ZINC | MAE ↓ | **0.075** (no ±) | *(no multi-seed paper group found)* | **?** (likely 1) | closest hybrid: [fotdo14c](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fotdo14c) a1g1 MAE=0.07545; see notes | ⚠ incomplete |
| MNIST | Acc. ↑ (%) | **98.628 ± 0.105** | **98.628 ± 0.105** | **5** | [`paper_bestmodel_v1_mnist_lcvbyyss`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_mnist_lcvbyyss) | ✅ |
| CIFAR10 | Acc. ↑ (%) | **79.528 ± 0.180** | **79.528 ± 0.175** | **5** | [`paper_bestmodel_v1_cifar10_ulij45a2`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_cifar10_ulij45a2) · anchor [`cifar10-hybrid-ulij45a2-anchor.yaml`](configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml) · exemplar [`3tx560wq`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3tx560wq) | ✅ |
| PATTERN | Acc. ↑ (%) | **86.991 ± 0.039** | **86.991 ± 0.039** | **5** | [`paper_bestmodel_v1_pattern_ta9qtxb9`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_pattern_ta9qtxb9) | ✅ |
| CLUSTER | Acc. ↑ (%) | **78.956 ± 0.112** | **78.956 ± 0.112** | **5** | [`paper_bestmodel_v1_cluster_ht9bntg2`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_cluster_ht9bntg2) | ✅ |

Per-seed run IDs (Table III, except ZINC):

| Dataset | run ids (seeds 0…n−1) |
|---------|------------------------|
| MNIST | `uh7nxm4e`, `jorlmk2q`, `86jxiuvz`, `loebushu`, `z7y9ucx2` |
| CIFAR10 | `skdtqk7t`, `3tx560wq`, `hep56q27`, `7wm0gq2c`, `61g0yg8m` |
| PATTERN | `qisbs4ml`, `ra26vllb`, `ywdmqhoe`, `8sg20f03`, `4j6xwd33` |
| CLUSTER | `qdkezojh`, `8bx9c5m2`, `f6k8rjip`, `vkno6rq8`, `6ipfj6sf` |

---

## Table IV — LRGB + MalNet-Tiny

| Dataset | Metric | Paper SiGMA | W&B recompute | Seeds (n) | W&B group / runs | Match? |
|---------|--------|-------------|---------------|-----------|------------------|--------|
| Peptides-func | AP ↑ | **0.7052 ± 0.0056** | **0.7052 ± 0.0056** | **10** | [`lr_ablation_peptides_func_o5cdk766_a1g1_b208_m0`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/lr_ablation_peptides_func_o5cdk766_a1g1_b208_m0) | ✅ |
| Peptides-struct | MAE ↓ | **0.2441 ± 0.0017** | **0.2443 ± 0.0016** | **10** | [`paper_bestmodel_v2_peptides_struct_g3bsaq32_b7m0_ep250`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v2_peptides_struct_g3bsaq32_b7m0_ep250) | NO |
| PascalVOC-SP | F1 ↑ | **0.4687 ± 0.0070** | **0.4687 ± 0.0070** | **5** | [`paper_bestmodel_v1_voc_j7ukyzdm`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_voc_j7ukyzdm) | ✅ |
| COCO-SP | F1 ↑ | **0.4155 ± 0.0076** | **0.4155 ± 0.0076** | **5** | group [`paper_bestmodel_v2_coco_5b4z9l3u`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v2_coco_5b4z9l3u) — use **job `25558630` seeds 0–4 only** (see below) | ✅ |
| MalNet-Tiny | Acc. ↑ | **0.9338 ± 0.0027** | **0.9338 ± 0.0027** | **5** | [`paper_bestmodel_v3_malnet_vcb1cuql`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v3_malnet_vcb1cuql) | ✅ |

Per-seed run IDs (Table IV):

| Dataset | run ids |
|---------|---------|
| Peptides-func | `ndpvme51`, `l17ms5n7`, `iktkn0uj`, `vdcdxmwo`, `zplr0jg6`, `fkbgdgy0`, `odjx0t07`, `o1yvy09b`, `l31u4b3k`, `wvebnqdd` |
| Peptides-struct | `joskj6gk`, `9vpas7oc`, `uh3uel3v`, `78a665x8`, `zdss5uhy`, `3fmlxpcc`, `bqkect9l`, `4cs5bulv`, `16kk1pe4`, `ddiowu3o` |
| PascalVOC-SP | `zhfigzdi`, `axtosowj`, `umkqmd0q`, `vyt7hjj5`, `sdsawqf0` |
| COCO-SP (paper set) | [`4ee23w14`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/4ee23w14), [`91k1swaf`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/91k1swaf), [`nf18ov76`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/nf18ov76), [`xgjakrz0`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xgjakrz0), [`qrdyr0wv`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/qrdyr0wv) |
| MalNet-Tiny | `figmqani`, `jfmn0kxg`, `5booamy2`, `aqtomaqd`, `5zi4gj9p` |

---

## Re-aggregate all matched groups

```bash
# Table III (print fraction; ×100 for %)
for g in \
  paper_bestmodel_v1_mnist_lcvbyyss \
  paper_bestmodel_v1_cifar10_ulij45a2 \
  paper_bestmodel_v1_pattern_ta9qtxb9 \
  paper_bestmodel_v1_cluster_ht9bntg2
do
  echo "===== $g ====="
  python scripts/api_wanndb_query/aggregate_paper_repro.py --group "$g" --metric best_test_perf --state finished
done

# Table IV
for g in \
  lr_ablation_peptides_func_o5cdk766_a1g1_b208_m0 \
  paper_bestmodel_v2_peptides_struct_g3bsaq32_b7m0_ep250 \
  paper_bestmodel_v1_voc_j7ukyzdm \
  paper_bestmodel_v3_malnet_vcb1cuql
do
  echo "===== $g ====="
  python scripts/api_wanndb_query/aggregate_paper_repro.py --group "$g" --metric best_test_perf --state finished
done
```

COCO: do **not** aggregate the whole `paper_bestmodel_v2_coco_5b4z9l3u` group (has 10 finished = 2× seeds 0–4). Paper number uses the first job batch (`25558630`) only.

---

## Notes / open issues

1. **Seed count vs table captions.** Tables claim “5 runs”, but:
   - Peptides-func paper number matches **n=10** (`lr_ablation_…_b208_m0`)
   - Peptides-struct paper number matches **n=10** (`g3bsaq32_b7m0_ep250`)
2. **ZINC 0.075** has no `paper_bestmodel_*` multi-seed group. Closest finished hybrid is a1g1 run `fotdo14c` (0.07545). Paper also omits ± std.
3. **Abstract / §5.2 vs Table IV** still disagree for some numbers:

| Claim | Abstract / §5.2 | Table IV / W&B |
|-------|-----------------|----------------|
| Peptides-func AP | 0.717 | **0.7052 ± 0.0056** (n=10) |
| Peptides-struct MAE | 0.2491 ± 0.0012 | **0.2441 ± 0.0017** (W&B n=10 → 0.2443) |
| COCO-SP F1 | 0.4162 ± 0.0055 | **0.4155 ± 0.0076** (n=5, job 25558630) |

   Alt Peptides-struct that matches §5.2: [`paper_bestmodel_v2_peptides_struct_rholn782_lr6e-4`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v2_peptides_struct_rholn782_lr6e-4) → **0.2491 ± 0.0012** (n=5).
4. **Alt paper cohorts (not used in tables above):**  
   - Peptides-func 5-seed paper repro: `paper_bestmodel_v2_peptides_func_o5cdk766_a1g1_ep900` → 0.7020 ± 0.0041 (does **not** match Table IV)  
   - COCO full group (10 finished): ~0.4164 ± 0.0061  
   - MalNet v1: `paper_bestmodel_v1_malnet_9h3jqzkm` → 0.9340 ± 0.0072
