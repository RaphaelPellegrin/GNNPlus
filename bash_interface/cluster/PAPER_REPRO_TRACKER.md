# Paper repro tracker (`bestmodel_v1` × 5 seeds)

Track frozen **best hybrid** configs rerun with seeds **0–4** for paper tables (mean ± std on **val-best test**).

W&B project: https://wandb.ai/weber-geoml-harvard-university/GNNPlus

---

## Conventions

| Field | Pattern |
|-------|---------|
| **Cohort tag** | `bestmodel_v1`, `bestmodel_v2`, `bestmodel_v3`, … (bump version when model choice changes) |
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
| MNIST | v2 | `paper_bestmodel_v2_mnist_429u8olp` | — | *(pending)* | not submitted |
| MalNet | v1 | `paper_bestmodel_v1_malnet_9h3jqzkm` | 5 | **0.9340 ± 0.0072** | done |
| MalNet | v2 | `paper_bestmodel_v2_malnet_4j21kp8d` | 1/5 | 0.8990 (seed 0 only) | in progress |
| MalNet | v3 | `paper_bestmodel_v3_malnet_vcb1cuql` | — | *(pending)* | not submitted |
| MalNet | v4 | `paper_bestmodel_v4_malnet_apiw6l3u` | — | *(pending)* | not submitted |
| VOC-SP | v1 | `paper_bestmodel_v1_voc_j7ukyzdm` | 5 | **0.2814 ± 0.0702** (test/f1) | done — high variance |
| COCO-SP | v1 | `paper_bestmodel_v1_coco_o5hr3tma` | — | *(pending)* | not submitted |
| CLUSTER | v1 | `paper_bestmodel_v1_cluster_ht9bntg2` | — | *(pending)* | not submitted |
| PATTERN | v1 | `paper_bestmodel_v1_pattern_ta9qtxb9` | — | *(pending)* | not submitted |
| peptides-struct | v1 | `paper_bestmodel_v1_peptides_struct_rholn782` | — | *(pending)* | not submitted |

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

### `bestmodel_v1` — MNIST — [lcvbyyss](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/lcvbyyss) *(v1, a2g2)*

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

### `bestmodel_v2` — MNIST — [429u8olp](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/429u8olp) *(v2, a8g2 — lower variance check)*

| Field | Value |
|-------|--------|
| **Dataset** | MNIST |
| **Architecture** | 8×attn + 2×GATEDGCN/GAT MP (a8g2), `d_h=32`, **headwise** gate |
| **Anchor W&B** | [429u8olp](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/429u8olp) (seed 0 discovery run) |
| **Repro commit** | `0ce0d126bd765a345fe953a7a31bd692675ef35e` |
| **Config** | `configs/gated_hybrid/mnist-hybrid-429u8olp-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_mnist_hybrid_429u8olp_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v2_mnist_429u8olp` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v2_mnist_429u8olp |
| **Tags** | `paper_repro`, `bestmodel_v2`, `mnist`, `anchor_429u8olp`, `hybrid_a8g2`, `gate_headwise` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/mnist_paper_v2_<JOBID>_<TASK>.log` |
| **Status** | not submitted |
| **Aggregate result** | *(pending)* |

**Expected run names** (fill W&B run id after submit):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `mnist_hybrid_v2_429u8olp_a8g2_seed0_job<JOBID>_1` | | |
| 1 | 2 | `mnist_hybrid_v2_429u8olp_a8g2_seed1_job<JOBID>_2` | | |
| 2 | 3 | `mnist_hybrid_v2_429u8olp_a8g2_seed2_job<JOBID>_3` | | |
| 3 | 4 | `mnist_hybrid_v2_429u8olp_a8g2_seed3_job<JOBID>_4` | | |
| 4 | 5 | `mnist_hybrid_v2_429u8olp_a8g2_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_mnist_hybrid_429u8olp_paper_repro.sh

grep "View run at" logs_gnnplus/mnist_paper_v2_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v2_mnist_429u8olp
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

### `bestmodel_v3` — MalNet-Tiny — [vcb1cuql](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vcb1cuql) *(v3, a1g1)*

| Field | Value |
|-------|--------|
| **Dataset** | MalNet-Tiny (`LocalDegreeProfile`) |
| **Architecture** | 1×attn + 1×GCNE MP (a1g1), `d_h=64`, **elementwise** gate, LayerNorm |
| **Anchor W&B** | [vcb1cuql](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vcb1cuql) (seed 0 discovery run) |
| **Repro commit** | `ec04a7e2799a90e311807b21ab4ab6910622a33f` |
| **Config** | `configs/gated_hybrid/malnet-hybrid-vcb1cuql-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_malnet_hybrid_vcb1cuql_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v3_malnet_vcb1cuql` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v3_malnet_vcb1cuql |
| **Tags** | `paper_repro`, `bestmodel_v3`, `malnet`, `anchor_vcb1cuql`, `hybrid_a1g1`, `gate_elementwise` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/malnet_paper_v3_<JOBID>_<TASK>.log` |
| **Status** | not submitted |
| **Aggregate result** | *(pending)* |

**Expected run names** (fill W&B run id after submit):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `malnet_hybrid_v3_vcb1cuql_a1g1_seed0_job<JOBID>_1` | | |
| 1 | 2 | `malnet_hybrid_v3_vcb1cuql_a1g1_seed1_job<JOBID>_2` | | |
| 2 | 3 | `malnet_hybrid_v3_vcb1cuql_a1g1_seed2_job<JOBID>_3` | | |
| 3 | 4 | `malnet_hybrid_v3_vcb1cuql_a1g1_seed3_job<JOBID>_4` | | |
| 4 | 5 | `malnet_hybrid_v3_vcb1cuql_a1g1_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_malnet_hybrid_vcb1cuql_paper_repro.sh

grep "View run at" logs_gnnplus/malnet_paper_v3_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v3_malnet_vcb1cuql
```

### `bestmodel_v4` — MalNet-Tiny — [apiw6l3u](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u) *(a0g3)*

| Field | Value |
|-------|--------|
| **Dataset** | MalNet-Tiny (LDP) |
| **Architecture** | **0×attn + 3×GCNE MP (a0g3)**, `d_h=110`, graph_restricted, elementwise+rmsnorm, L8 |
| **Parent run** | [apiw6l3u](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/apiw6l3u) (v1 repro seed 2, **a0g2**, `best_test_perf` **0.944**) |
| **Change vs parent** | `num_gnn_heads`: 2 → **3** |
| **Config** | `configs/gated_hybrid/malnet-hybrid-apiw6l3u-a0g3-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_malnet_hybrid_apiw6l3u_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v4_malnet_apiw6l3u` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v4_malnet_apiw6l3u |
| **Tags** | `paper_repro`, `bestmodel_v4`, `malnet`, `anchor_apiw6l3u`, `hybrid_a0g3` |
| **Metric** | `best_test_perf` (`metric_best: accuracy`) |
| **Logs** | `logs_gnnplus/malnet_paper_v4_<JOBID>_<TASK>.log` |
| **Status** | not submitted |

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `malnet_hybrid_apiw6l3u_a0g3_seed0_job<JOBID>_1` | | |
| 1 | 2 | `malnet_hybrid_apiw6l3u_a0g3_seed1_job<JOBID>_2` | | |
| 2 | 3 | `malnet_hybrid_apiw6l3u_a0g3_seed2_job<JOBID>_3` | | |
| 3 | 4 | `malnet_hybrid_apiw6l3u_a0g3_seed3_job<JOBID>_4` | | |
| 4 | 5 | `malnet_hybrid_apiw6l3u_a0g3_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_malnet_hybrid_apiw6l3u_paper_repro.sh

grep "View run at" logs_gnnplus/malnet_paper_v4_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v4_malnet_apiw6l3u
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

### `bestmodel_v1` — CLUSTER — [ht9bntg2](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ht9bntg2) *(v1, a1g1+RWSE)*

| Field | Value |
|-------|--------|
| **Dataset** | CLUSTER (`PyG-GNNBenchmarkDataset`) |
| **Architecture** | 1×attn + 1×GATEDGCN MP (a1g1), `d_h=64`, **headwise** gate, LayerNorm, RWSE |
| **Anchor W&B** | [ht9bntg2](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ht9bntg2) (seed 1 discovery run, ≈0.793 SBM) |
| **Repro commit** | `8138ddee6f3fa4d58052ebd86d17939e832eaea3` |
| **Config** | `configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_cluster_hybrid_ht9bntg2_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v1_cluster_ht9bntg2` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_cluster_ht9bntg2 |
| **Tags** | `paper_repro`, `bestmodel_v1`, `cluster`, `anchor_ht9bntg2`, `hybrid_a1g1`, `gate_headwise`, `rwse` |
| **Metric** | `best_test_perf` (`metric_best: accuracy-SBM`) |
| **Logs** | `logs_gnnplus/cluster_paper_v1_<JOBID>_<TASK>.log` |
| **Status** | not submitted |
| **Aggregate result** | *(pending)* |

**Expected run names** (fill W&B run id after submit):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `cluster_hybrid_ht9bntg2_a1g1_seed0_job<JOBID>_1` | | |
| 1 | 2 | `cluster_hybrid_ht9bntg2_a1g1_seed1_job<JOBID>_2` | | |
| 2 | 3 | `cluster_hybrid_ht9bntg2_a1g1_seed2_job<JOBID>_3` | | |
| 3 | 4 | `cluster_hybrid_ht9bntg2_a1g1_seed3_job<JOBID>_4` | | |
| 4 | 5 | `cluster_hybrid_ht9bntg2_a1g1_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_cluster_hybrid_ht9bntg2_paper_repro.sh

grep "View run at" logs_gnnplus/cluster_paper_v1_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_cluster_ht9bntg2
```

### `bestmodel_v1` — PATTERN — [ta9qtxb9](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) *(v1, a2g2+RWSE)*

| Field | Value |
|-------|--------|
| **Dataset** | PATTERN (`PyG-GNNBenchmarkDataset`) |
| **Architecture** | 2×attn + 2×GCNE MP (a2g2), `d_h=90`, **elementwise** gate, RMSNorm, RWSE |
| **Anchor W&B** | [ta9qtxb9](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) (seed 0 discovery run, ≈0.871 SBM) |
| **Repro commit** | `c62ea95f392c56d5b330edcfeda0c460b416683b` |
| **Config** | `configs/gated_hybrid/pattern-hybrid-ta9qtxb9-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_pattern_hybrid_ta9qtxb9_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v1_pattern_ta9qtxb9` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_pattern_ta9qtxb9 |
| **Tags** | `paper_repro`, `bestmodel_v1`, `pattern`, `anchor_ta9qtxb9`, `hybrid_a2g2`, `gate_elementwise`, `rwse` |
| **Metric** | `best_test_perf` (`metric_best: accuracy-SBM`) |
| **Logs** | `logs_gnnplus/pattern_paper_v1_<JOBID>_<TASK>.log` |
| **Status** | not submitted |
| **Aggregate result** | *(pending)* |

**Expected run names** (fill W&B run id after submit):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `pattern_hybrid_ta9qtxb9_a2g2_seed0_job<JOBID>_1` | | |
| 1 | 2 | `pattern_hybrid_ta9qtxb9_a2g2_seed1_job<JOBID>_2` | | |
| 2 | 3 | `pattern_hybrid_ta9qtxb9_a2g2_seed2_job<JOBID>_3` | | |
| 3 | 4 | `pattern_hybrid_ta9qtxb9_a2g2_seed3_job<JOBID>_4` | | |
| 4 | 5 | `pattern_hybrid_ta9qtxb9_a2g2_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_pattern_hybrid_ta9qtxb9_paper_repro.sh

grep "View run at" logs_gnnplus/pattern_paper_v1_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_pattern_ta9qtxb9
```

---

### `bestmodel_v1` — peptides-struct — [rholn782](https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/rholn782) (MOE_6 anchor)

| Field | Value |
|-------|--------|
| **Dataset** | peptides-structural (OGB graph regression, 11 targets) |
| **Architecture** | 2×attn + 2×MP (GINE+GGNN), `d_h=16`, `layers_mp=12`, `dim_inner=96`, vn=4, pyramid readout, RWSE |
| **MOE anchor** | [rholn782](https://wandb.ai/weber-geoml-harvard-university/MOE_6/runs/rholn782) (`test_mean` ≈ **0.229**) |
| **GNNPlus baseline** | [xfb9wdir](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xfb9wdir) (standard GINE L5/H200, no VN, `test/mae` ≈ **0.242**) |
| **Config** | `configs/gated_hybrid/peptides-struct-hybrid-rholn782-anchor.yaml` |
| **Submit** | `bash bash_interface/cluster/submit_peptides_struct_hybrid_rholn782_paper_repro.sh` |
| **SLURM array** | *(pending)* |
| **W&B group** | `paper_bestmodel_v1_peptides_struct_rholn782` |
| **Group URL** | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_bestmodel_v1_peptides_struct_rholn782 |
| **Tags** | `paper_repro`, `bestmodel_v1`, `peptides_struct`, `anchor_rholn782`, `hybrid_a2g2`, `vn4`, `pyramid_readout` |
| **Metric** | `best_test_perf` (`metric_best: mae`) |
| **Resources** | 128GB, 192h, parallel=3 (VN graphs are memory-heavy) |
| **Logs** | `logs_gnnplus/peptides_struct_paper_v1_<JOBID>_<TASK>.log` |
| **Status** | not submitted |
| **Aggregate result** | *(pending)* |

**VN reliability note:** GNNPlus best-hybrid sweep with `add_virtual_nodes=true` had only **2 finished** runs ([jy7xsmrb](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/jy7xsmrb), [sy0apx7e](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/sy0apx7e)) vs 2 crashed (likely SLURM time/OOM on 120h sweep agents). Paper repro uses **192h / 128GB**; set `PEPTIDES_PAPER_BATCH_SIZE=32` if needed.

**Expected run names** (fill W&B run id after submit):

| Seed | SLURM task | Run name | W&B run id | `best_test_perf` |
|------|------------|----------|------------|------------------|
| 0 | 1 | `peptides_struct_hybrid_rholn782_a2g2_seed0_job<JOBID>_1` | | |
| 1 | 2 | `peptides_struct_hybrid_rholn782_a2g2_seed1_job<JOBID>_2` | | |
| 2 | 3 | `peptides_struct_hybrid_rholn782_a2g2_seed2_job<JOBID>_3` | | |
| 3 | 4 | `peptides_struct_hybrid_rholn782_a2g2_seed3_job<JOBID>_4` | | |
| 4 | 5 | `peptides_struct_hybrid_rholn782_a2g2_seed4_job<JOBID>_5` | | |
| **mean ± std** | | | | |

```bash
bash bash_interface/cluster/submit_peptides_struct_hybrid_rholn782_paper_repro.sh

grep "View run at" logs_gnnplus/peptides_struct_paper_v1_<JOBID>_*.log

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_peptides_struct_rholn782
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
| 2026-06-07 | MNIST 429u8olp (v2, a8g2) anchor config + 5-seed submit scripts |
| 2026-06-07 | MalNet v3 vcb1cuql (a1g1, d_h=64) anchor config + 5-seed submit scripts |
| 2026-06-07 | CLUSTER ht9bntg2 (a1g1+RWSE) paper repro config + 5-seed submit scripts |
| 2026-06-07 | PATTERN ta9qtxb9 (a2g2+RWSE) paper repro config + 5-seed submit scripts |
| 2026-06-07 | peptides-struct rholn782 (MOE hybrid+VN) paper repro config + 5-seed submit scripts |
