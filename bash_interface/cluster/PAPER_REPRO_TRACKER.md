# Paper repro tracker (`bestmodel_v1` × 5 seeds)

Track frozen **best hybrid** configs rerun with seeds **0–4** for paper tables (mean ± std on **val-best test**).

W&B project: https://wandb.ai/weber-geoml-harvard-university/GNNPlus

---

## Conventions

| Field | Pattern |
|-------|---------|
| **Cohort tag** | `bestmodel_v1` or `bestmodel_v2` (bump version when model choice changes) |
| **W&B group** | `paper_bestmodel_v<N>_<dataset>_<anchor_wandb_id>` |
| **W&B tags** | `paper_repro`, `bestmodel_v1`, `<dataset>`, `anchor_<id>`, … |
| **Run name** | `<dataset>_hybrid_<anchor>_<arch>_seed<N>_job<SLURM_ARRAY_JOB_ID>_<TASK>` |
| **Report metric** | `best_test_perf` in run **Summary** (= test at val-best epoch) |
| **Aggregate** | `python scripts/api_wanndb_query/aggregate_paper_repro.py --group <GROUP>` |

**W&B filters**

```text
group:paper_bestmodel_v1_cifar10_ulij45a2
tag:bestmodel_v1
tag:paper_repro
```

**Group URL template**

```text
https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/<GROUP>
```

---

## Summary table (2026-06-07 aggregate)

| Dataset | Version | Group | n | `best_test_perf` | Status |
|---------|---------|-------|---|------------------|--------|
| CIFAR10 | v1 | `paper_bestmodel_v1_cifar10_ulij45a2` | 5 | **0.7290 ± 0.0078** | done |
| MNIST | v1 | `paper_bestmodel_v1_mnist_lcvbyyss` | 5 | **0.9820 ± 0.0008** | done |
| MalNet | v1 | `paper_bestmodel_v1_malnet_9h3jqzkm` | 5 | **0.9340 ± 0.0072** | done |
| MalNet | v2 | `paper_bestmodel_v2_malnet_4j21kp8d` | 1/5 | 0.8990 (seed 0 only) | in progress |
| VOC-SP | v1 | `paper_bestmodel_v1_voc_j7ukyzdm` | 5 | **0.2814 ± 0.0702** (test/f1) | done — high variance |
| COCO-SP | v1 | `paper_bestmodel_v1_coco_o5hr3tma` | — | *(pending)* | not submitted |

---

## Active cohorts

### `bestmodel_v1` — CIFAR10 — [ulij45a2](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ulij45a2)

| Field | Value |
|-------|--------|
| **Dataset** | CIFAR10 |
| **Architecture** | 8×attn + 4×GATEDGCN MP, `d_h=256`, `layers_mp=10` |
| **Anchor W&B** | [ulij45a2](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ulij45a2) (seed 0 discovery run) |
| **Repro commit** | `ca47851df38e54a3ed052be53f9072bcb159464e` |
| **Config** | `configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_cifar10_hybrid_ulij45a2_paper_repro.sh` |
| **SLURM array** | `25310487` (tasks 1–5 → seeds 0–4) |
| **W&B group** | `paper_bestmodel_v1_cifar10_ulij45a2` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_cifar10_ulij45a2 |
| **Tags** | `paper_repro`, `bestmodel_v1`, `cifar10`, `anchor_ulij45a2`, `hybrid_a8g4` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/cifar10_paper_v1_25310487_<TASK>.log` |
| **Status** | finished |
| **Aggregate result** | **0.7290 ± 0.0078** (`best_test_perf`, n=5) |

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `cifar10_hybrid_ulij45a2_a8g4_seed0_job25310487_1` | skdtqk7t | 0.7326 |
| 1 | 2 | `cifar10_hybrid_ulij45a2_a8g4_seed1_job25310487_2` | 3tx560wq | 0.7327 |
| 2 | 3 | `cifar10_hybrid_ulij45a2_a8g4_seed2_job25310487_3` | hep56q27 | 0.7253 |
| 3 | 4 | `cifar10_hybrid_ulij45a2_a8g4_seed3_job25310487_4` | 7wm0gq2c | 0.7373 |
| 4 | 5 | `cifar10_hybrid_ulij45a2_a8g4_seed4_job25310487_5` | 61g0yg8m | 0.7173 |
| **mean ± std** | | | | **0.7290 ± 0.0078** |

```bash
# Fill run ids from logs
grep "View run at" logs_gnnplus/cifar10_paper_v1_25310487_*.log

# Paper table number
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_cifar10_ulij45a2
```

### `bestmodel_v1` — MNIST — [lcvbyyss](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/lcvbyyss)

| Field | Value |
|-------|--------|
| **Dataset** | MNIST |
| **Architecture** | 2×attn + 2×GATEDGCN MP (`GATEDGCN`×4 types), `d_h=64`, **elementwise** gate |
| **Anchor W&B** | [lcvbyyss](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/lcvbyyss) (seed 0 discovery run) |
| **Repro commit** | `a28a4f998373167d689d7fea4807c562a66d8c0a` |
| **Config** | `configs/gated_hybrid/mnist-hybrid-lcvbyyss-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_mnist_hybrid_lcvbyyss_paper_repro.sh` |
| **SLURM array** | `25313521` (tasks 1–5 → seeds 0–4) |
| **W&B group** | `paper_bestmodel_v1_mnist_lcvbyyss` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_mnist_lcvbyyss |
| **Tags** | `paper_repro`, `bestmodel_v1`, `mnist`, `anchor_lcvbyyss`, `hybrid_a2g2`, `gate_elementwise` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/mnist_paper_v1_25313521_<TASK>.log` |
| **Status** | finished |
| **Aggregate result** | **0.9820 ± 0.0008** (`best_test_perf`, n=5) |

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `mnist_hybrid_lcvbyyss_a2g2_seed0_job25313521_1` | uh7nxm4e | 0.9821 |
| 1 | 2 | `mnist_hybrid_lcvbyyss_a2g2_seed1_job25313521_2` | jorlmk2q | 0.9810 |
| 2 | 3 | `mnist_hybrid_lcvbyyss_a2g2_seed2_job25313521_3` | 86jxiuvz | 0.9832 |
| 3 | 4 | `mnist_hybrid_lcvbyyss_a2g2_seed3_job25313521_4` | loebushu | 0.9817 |
| 4 | 5 | `mnist_hybrid_lcvbyyss_a2g2_seed4_job25313521_5` | z7y9ucx2 | 0.9818 |
| **mean ± std** | | | | **0.9820 ± 0.0008** |

```bash
grep "View run at" logs_gnnplus/mnist_paper_v1_25313521_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_mnist_lcvbyyss
```

### `bestmodel_v1` — MalNet-Tiny — [9h3jqzkm](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/9h3jqzkm) *(v1, a0g2)*

| Field | Value |
|-------|--------|
| **Dataset** | MalNet-Tiny (`LocalDegreeProfile`) |
| **Architecture** | 0×attn + 2×GCNE MP (a0g2), `d_h=110`, **elementwise** gate, RMSNorm |
| **Anchor W&B** | [9h3jqzkm](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/9h3jqzkm) (seed 0 discovery run) |
| **Repro commit** | `ed02d752b932f932b924a1def36c244c8f9c1694` |
| **Config** | `configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_paper_repro.sh` |
| **SLURM array** | `25313522` (tasks 1–5 → seeds 0–4) |
| **W&B group** | `paper_bestmodel_v1_malnet_9h3jqzkm` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_malnet_9h3jqzkm |
| **Tags** | `paper_repro`, `bestmodel_v1`, `malnet`, `anchor_9h3jqzkm`, `hybrid_a0g2`, `gate_elementwise` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/malnet_paper_v1_25313522_<TASK>.log` |
| **Status** | finished (submitted 2026-06-07) |
| **Aggregate result** | **0.9340 ± 0.0072** (`best_test_perf`, n=5) |

**Run names** (W&B run ids filled):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `malnet_hybrid_9h3jqzkm_a0g2_seed0_job25313522_1` | fbnaz7tg | 0.924 |
| 1 | 2 | `malnet_hybrid_9h3jqzkm_a0g2_seed1_job25313522_2` | 2usx7o39 | 0.936 |
| 2 | 3 | `malnet_hybrid_9h3jqzkm_a0g2_seed2_job25313522_3` | apiw6l3u | 0.944 |
| 3 | 4 | `malnet_hybrid_9h3jqzkm_a0g2_seed3_job25313522_4` | 1t0594xm | 0.932 |
| 4 | 5 | `malnet_hybrid_9h3jqzkm_a0g2_seed4_job25313522_5` | i9owtpa9 | 0.934 |
| **mean ± std** | | | | **0.9340 ± 0.0072** |

```bash
grep "View run at" logs_gnnplus/malnet_paper_v1_25313522_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_malnet_9h3jqzkm
```

### `bestmodel_v2` — MalNet-Tiny — [4j21kp8d](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/4j21kp8d) *(v2, a1g2 — lower variance check)*

| Field | Value |
|-------|--------|
| **Dataset** | MalNet-Tiny (`LocalDegreeProfile`) |
| **Architecture** | 1×attn + 2×GCNE/GINE MP (a1g2), `d_h=110`, **elementwise** gate, RMSNorm |
| **Anchor W&B** | [4j21kp8d](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/4j21kp8d) (seed 0 discovery run) |
| **Repro commit** | `5f30e22107510047fdad90910bf343fdf7b44c96` |
| **Config** | `configs/gated_hybrid/malnet-hybrid-4j21kp8d-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_malnet_hybrid_4j21kp8d_paper_repro.sh` |
| **SLURM array** | `25324182` (tasks 1–5 → seeds 0–4) |
| **W&B group** | `paper_bestmodel_v2_malnet_4j21kp8d` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v2_malnet_4j21kp8d |
| **Tags** | `paper_repro`, `bestmodel_v2`, `malnet`, `anchor_4j21kp8d`, `hybrid_a1g2`, `gate_elementwise` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/malnet_paper_v2_25324182_<TASK>.log` |
| **Status** | in progress (1/5 seeds with metric) |
| **Aggregate result** | seed 0: **0.899** (n=1 so far; v1 mean was 0.934) |

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `malnet_hybrid_v2_4j21kp8d_a1g2_seed0_job25324182_1` | dy4ce9nr | 0.899 |
| 1 | 2 | `malnet_hybrid_v2_4j21kp8d_a1g2_seed1_job25324182_2` | | |
| 2 | 3 | `malnet_hybrid_v2_4j21kp8d_a1g2_seed2_job25324182_3` | | |
| 3 | 4 | `malnet_hybrid_v2_4j21kp8d_a1g2_seed3_job25324182_4` | | |
| 4 | 5 | `malnet_hybrid_v2_4j21kp8d_a1g2_seed4_job25324182_5` | | |
| **mean ± std** | | | | *(pending n=5)* |

```bash
grep "View run at" logs_gnnplus/malnet_paper_v2_25324182_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v2_malnet_4j21kp8d
```

### `bestmodel_v1` — VOC-SP — [j7ukyzdm](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/j7ukyzdm)

| Field | Value |
|-------|--------|
| **Dataset** | VOC superpixels (`edge_wt_region_boundary` + RWSE) |
| **Architecture** | 2×attn + 2×GATEDGCN MP (a2g2+RWSE), `d_h=64`, **headwise** gate, FFN, `layers_mp=16` |
| **Anchor W&B** | [j7ukyzdm](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/j7ukyzdm) (seed 0 discovery run) |
| **Repro commit** | `5f30e22107510047fdad90910bf343fdf7b44c96` |
| **Config** | `configs/gated_hybrid/voc-hybrid-j7ukyzdm-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_voc_hybrid_j7ukyzdm_paper_repro.sh` |
| **SLURM array** | `25322496` (tasks 1–5 → seeds 0–4) |
| **W&B group** | `paper_bestmodel_v1_voc_j7ukyzdm` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_voc_j7ukyzdm |
| **Tags** | `paper_repro`, `bestmodel_v1`, `voc`, `anchor_j7ukyzdm`, `hybrid_a2g2`, `gate_headwise`, `rwse` |
| **Metric** | `best_test_perf` (`metric_best: f1`) |
| **Logs** | `logs_gnnplus/voc_paper_v1_25322496_<TASK>.log` |
| **Status** | finished |
| **Aggregate result** | **0.2814 ± 0.0702** (`best_test_perf` = test/f1, n=5) — seed 4 outlier (0.163) |

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `voc_hybrid_j7ukyzdm_a2g2_seed0_job25322496_1` | zhfigzdi | 0.3404 |
| 1 | 2 | `voc_hybrid_j7ukyzdm_a2g2_seed1_job25322496_2` | axtosowj | 0.2820 |
| 2 | 3 | `voc_hybrid_j7ukyzdm_a2g2_seed2_job25322496_3` | umkqmd0q | 0.3277 |
| 3 | 4 | `voc_hybrid_j7ukyzdm_a2g2_seed3_job25322496_4` | vyt7hjj5 | 0.2936 |
| 4 | 5 | `voc_hybrid_j7ukyzdm_a2g2_seed4_job25322496_5` | sdsawqf0 | 0.1634 |
| **mean ± std** | | | | **0.2814 ± 0.0702** |

```bash
grep "View run at" logs_gnnplus/voc_paper_v1_25322496_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_voc_j7ukyzdm
```

### `bestmodel_v1` — COCO-SP — [o5hr3tma](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o5hr3tma)

| Field | Value |
|-------|--------|
| **Dataset** | COCO superpixels (`edge_wt_region_boundary`) |
| **Architecture** | 2×attn + 8×GATEDGRAPH MP (a2g8), `d_h=32`, **elementwise** gate, RMSNorm |
| **Anchor W&B** | [o5hr3tma](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o5hr3tma) (seed 0 discovery run) |
| **Repro commit** | `ca47851df38e54a3ed052be53f9072bcb159464e` |
| **Config** | `configs/gated_hybrid/coco-hybrid-o5hr3tma-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_coco_hybrid_o5hr3tma_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v1_coco_o5hr3tma` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_coco_o5hr3tma |
| **Tags** | `paper_repro`, `bestmodel_v1`, `coco`, `anchor_o5hr3tma`, `hybrid_a2g8`, `gate_elementwise` |
| **Metric** | `best_test_perf` (`metric_best: f1`) |
| **Logs** | `logs_gnnplus/coco_paper_v1_<JOBID>_<TASK>.log` |
| **Status** | not submitted |
| **Aggregate result** | *(pending)* |

**Expected run names** (fill W&B run id after submit):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `coco_hybrid_o5hr3tma_a2g8_seed0_job<JOBID>_1` | | |
| 1 | 2 | `coco_hybrid_o5hr3tma_a2g8_seed1_job<JOBID>_2` | | |
| 2 | 3 | `coco_hybrid_o5hr3tma_a2g8_seed2_job<JOBID>_3` | | |
| 3 | 4 | `coco_hybrid_o5hr3tma_a2g8_seed3_job<JOBID>_4` | | |
| 4 | 5 | `coco_hybrid_o5hr3tma_a2g8_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_coco_hybrid_o5hr3tma_paper_repro.sh

grep "View run at" logs_gnnplus/coco_paper_v1_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_coco_o5hr3tma
```

---

## Planned / template (copy for next dataset)

```markdown
### `bestmodel_v1` — <DATASET> — <anchor_wandb_id>

| Field | Value |
|-------|--------|
| **Anchor W&B** | <url> |
| **Config** | `configs/gated_hybrid/<...>-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_<...>_paper_repro.sh` |
| **SLURM array** | `<JOBID>` |
| **W&B group** | `paper_bestmodel_v1_<dataset>_<anchor_id>` |
| **Tags** | `paper_repro`, `bestmodel_v1`, ... |
| **Metric** | `best_test_perf` (`metric_best: ...`) |

| Seed | Task | Run name | W&B run id | `best_test_perf` |
|------|------|----------|------------|------------------|
| 0 | 1 | | | |
...
```

---

## Changelog

| Date | Notes |
|------|-------|
| 2026-06-27 | Created tracker; CIFAR10 ulij45a2 cohort `25310487` |
| 2026-06-27 | MNIST lcvbyyss anchor config + 5-seed submit scripts |
| 2026-06-27 | MalNet-Tiny 9h3jqzkm anchor config + 5-seed submit scripts |
| 2026-06-07 | Submitted MNIST lcvbyyss cohort `25313521`, MalNet 9h3jqzkm cohort `25313522` |
| 2026-06-07 | VOC-SP j7ukyzdm anchor config + 5-seed submit scripts |
| 2026-06-07 | MalNet v1 `25313522` finished: **0.9340 ± 0.0072**; v2 `25324182`, VOC `25322496` submitted |
| 2026-06-07 | Full aggregate: CIFAR **0.729±0.008**, MNIST **0.982±0.001**, VOC **0.281±0.070** (f1); MalNet v2 1/5 |
| 2026-06-07 | COCO-SP o5hr3tma anchor config + 5-seed submit scripts |
