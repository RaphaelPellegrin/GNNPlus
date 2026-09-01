# Rebuttal experiment log — SiGMA d_h-matched (Tab. 3/4 budget analog)

Living tracker for the **≤500k / ≤1M** SiGMA shrink campaign (TU Tab. 17/18 recipe
applied to Dwivedi + LRGB benchmarks). Complements
[`Paper_sigma_dh_matched.md`](Paper_sigma_dh_matched.md) and
[`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md).

**W&B:** [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
**Groups:** `paper_sigma_dh_matched_<family>_{lr001,lr01}` · tag `sigma_dh_matched`  
**Cluster repo:** `/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus`  
**Out dir:** `$GNNPLUS_OUT_DIR/sigma_dh_matched/<fam>_<lr>_seed<s>/`  
(default: `/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results`)

---

## Campaign summary

| Metric | Value |
|--------|------:|
| Total jobs (3 tiers) | **150** |
| Families × 2 LRs × 5 seeds | 15 × 2 × 5 |
| **Fast tier Slurm COMPLETED** | **96/100** (`41709078` 50 + `42412053` 46) |
| **Slow tier Slurm COMPLETED** | **5/40** (`41709082`) |
| **Coco tier** | 2 running · 2 TIMEOUT (tasks 1–2) · rest pending |
| **W&B finished (full 5-seed lr001 groups)** | **7** families (Dwivedi×5 + Pep-func `dh23`) |
| **Need resubmit** | MalNet tasks **91, 94, 96, 99**; COCO tasks **1–2** |
| Skipped dataset | ZINC (main already ≤500k) |

**Last updated:** 2026-08-28 (cluster `sacct -X` + W&B API on holylogin06).

### Performance vs paper SiGMA (finished runs only)

Compared to [`Paper_sigma_params.md`](Paper_sigma_params.md) Table III baselines.
Best LR per family (`lr001` won everywhere so far). Acc = `best_test_perf` × 100.

| Dataset | Budget | Params (small / paper) | Small SiGMA | Paper SiGMA | Δ (pp) | Verdict |
|---------|--------|------------------------|------------:|------------:|-------:|---------|
| PATTERN | ~1M `dh16` | 844k / 1.99M | **87.23±0.18%** | 86.99±0.04% | **+0.23** | ≈ same / slightly better |
| PATTERN | ~500k `dh4` | 519k / 1.99M | **87.03±0.07%** | 86.99±0.04% | **+0.04** | ≈ same |
| CLUSTER | ~500k `dh36` | 437k / 1.03M | 78.82±0.08% | 78.96±0.11% | −0.13 | ≈ same |
| CLUSTER | ~500k `dh24` | 254k / 1.03M | 78.72±0.16% | 78.96±0.11% | −0.24 | ≈ same |
| MNIST | ~500k `dh37` | 488k / 965k | **98.62±0.07%** | 98.63±0.11% | −0.01 | ≈ identical |
| Pep-func | ~500k `dh23` | 491k / 1.54M | **0.7002±0.0084 AP** | 0.7080±0.0063 | **−0.8 pp** | ≈ same |
| CIFAR10 | ~500k `dh20` | 477k / **27.8M** | 75.15±0.57% | 79.53±0.18% | **−4.37** | **clear drop** |

**Takeaway:** Shrinking `d_h` is **benign on PATTERN / CLUSTER / MNIST / Pep-func** (≤0.8 pp at
best LR) despite 25–50% of paper params. **CIFAR10 is the outlier** — ~4.4 pp loss at
~500k params (1.7% of paper param count); `dh34` (~1M) not finished yet.  
`lr=0.01` hurts MNIST (−2.1 pp) and CLUSTER (−0.7–0.8 pp) vs `lr=0.001`.

### Best-LR run registry (finished small SiGMA)

W&B entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Metric: `best_test_perf` (Acc × 100 for Dwivedi). **Best LR = `lr001` (0.001)** for all
finished families below. SLURM parent: fast `41709078` (tasks 1–50) + slow `41709082`
(tasks 1–5 for CIFAR dh20).

| Dataset | Budget | `d_h` | Params | Best LR | Result | n | W&B group | Run IDs (seeds 0–4) |
|---------|--------|------|-------:|---------|--------|--:|-----------|---------------------|
| PATTERN | ~1M | 16 | 843,771 | **lr001** | **87.23±0.18%** | 5 | [`paper_sigma_dh_matched_pattern_dh16_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_pattern_dh16_lr001) | `2r2c15tk`, `xv5pv0qw`, `hjgldww7`, `lmc9sfzg`, `dd8i28nj` |
| PATTERN | ~500k | 4 | 518,907 | **lr001** | **87.03±0.07%** | 5 | [`paper_sigma_dh_matched_pattern_dh4_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_pattern_dh4_lr001) | `jyirb9xb`, `m7ez4v2x`, `j6qxravr`, `6geowka0`, `inshid5c` |
| CLUSTER | ~500k | 36 | 437,078 | **lr001** | 78.82±0.08% | 5 | [`paper_sigma_dh_matched_cluster_dh36_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_cluster_dh36_lr001) | `t223xw7w`, `fvhdrwji`, `jsu5z7so`, `p0af48de`, `ewf4p16q` |
| CLUSTER | ~500k | 24 | 254,102 | **lr001** | 78.72±0.16% | 5 | [`paper_sigma_dh_matched_cluster_dh24_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_cluster_dh24_lr001) | `tk4esb27`, `l1up71vy`, `04t862ln`, `vc6v1noc`, `h58gryxl` |
| MNIST | ~500k | 37 | 487,954 | **lr001** | **98.62±0.07%** | 5 | [`paper_sigma_dh_matched_mnist_dh37_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_mnist_dh37_lr001) | `lharj7it`, `4j7wgdph`, `gfakkhqt`, `osnhyfmn`, `1erfq9gv` |
| Pep-func | ~500k | 23 | 491,280 | **lr001** | **0.7002±0.0084 AP** | 5 | [`paper_sigma_dh_matched_pepfunc_dh23_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_pepfunc_dh23_lr001) | `jymcah8k`, `p3jl40xy`, `xoojqs74`, `tlpzwy96`, `8kqvfsme` |
| CIFAR10 | ~500k | 20 | 477,070 | **lr001** | 75.15±0.57% | 5 | [`paper_sigma_dh_matched_cifar_dh20_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_cifar_dh20_lr001) | `9cv9fzmd`, `ktl4zf45`, `opkpdg89`, `902s5coe`, `oftwykv6` |

#### Per-seed detail (best LR only)

**PATTERN `dh16` lr001** — group [`paper_sigma_dh_matched_pattern_dh16_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_pattern_dh16_lr001)

| Seed | Run ID | Acc (%) | Link |
|-----:|--------|--------:|------|
| 0 | `2r2c15tk` | 87.10 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/2r2c15tk) |
| 1 | `xv5pv0qw` | 87.51 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xv5pv0qw) |
| 2 | `hjgldww7` | 87.07 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/hjgldww7) |
| 3 | `lmc9sfzg` | 87.28 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/lmc9sfzg) |
| 4 | `dd8i28nj` | 87.16 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/dd8i28nj) |

**PATTERN `dh4` lr001** — group [`paper_sigma_dh_matched_pattern_dh4_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_pattern_dh4_lr001)

| Seed | Run ID | Acc (%) | Link |
|-----:|--------|--------:|------|
| 0 | `jyirb9xb` | 86.99 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/jyirb9xb) |
| 1 | `m7ez4v2x` | 87.04 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/m7ez4v2x) |
| 2 | `j6qxravr` | 86.94 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/j6qxravr) |
| 3 | `6geowka0` | 87.06 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/6geowka0) |
| 4 | `inshid5c` | 87.13 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/inshid5c) |

**CLUSTER `dh36` lr001** — group [`paper_sigma_dh_matched_cluster_dh36_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_cluster_dh36_lr001)

| Seed | Run ID | Acc (%) | Link |
|-----:|--------|--------:|------|
| 0 | `t223xw7w` | 78.93 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/t223xw7w) |
| 1 | `fvhdrwji` | 78.79 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/fvhdrwji) |
| 2 | `jsu5z7so` | 78.89 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/jsu5z7so) |
| 3 | `p0af48de` | 78.76 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/p0af48de) |
| 4 | `ewf4p16q` | 78.75 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ewf4p16q) |

**CLUSTER `dh24` lr001** — group [`paper_sigma_dh_matched_cluster_dh24_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_cluster_dh24_lr001)

| Seed | Run ID | Acc (%) | Link |
|-----:|--------|--------:|------|
| 0 | `tk4esb27` | 78.96 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tk4esb27) |
| 1 | `l1up71vy` | 78.62 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/l1up71vy) |
| 2 | `04t862ln` | 78.53 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/04t862ln) |
| 3 | `vc6v1noc` | 78.72 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vc6v1noc) |
| 4 | `h58gryxl` | 78.77 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/h58gryxl) |

**MNIST `dh37` lr001** — group [`paper_sigma_dh_matched_mnist_dh37_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_mnist_dh37_lr001)

| Seed | Run ID | Acc (%) | Link |
|-----:|--------|--------:|------|
| 0 | `lharj7it` | 98.67 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/lharj7it) |
| 1 | `4j7wgdph` | 98.51 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/4j7wgdph) |
| 2 | `gfakkhqt` | 98.68 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/gfakkhqt) |
| 3 | `osnhyfmn` | 98.60 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/osnhyfmn) |
| 4 | `1erfq9gv` | 98.63 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/1erfq9gv) |

**CIFAR10 `dh20` lr001** — group [`paper_sigma_dh_matched_cifar_dh20_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_cifar_dh20_lr001) · SLURM `41709082` tasks 1–5

| Seed | Run ID | Acc (%) | Link |
|-----:|--------|--------:|------|
| 0 | `9cv9fzmd` | 75.91 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/9cv9fzmd) |
| 1 | `ktl4zf45` | 75.22 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ktl4zf45) |
| 2 | `opkpdg89` | 75.35 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/opkpdg89) |
| 3 | `902s5coe` | 74.34 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/902s5coe) |
| 4 | `oftwykv6` | 74.95 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/oftwykv6) |

**Pep-func `dh23` lr001** — group [`paper_sigma_dh_matched_pepfunc_dh23_lr001`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/groups/paper_sigma_dh_matched_pepfunc_dh23_lr001) · SLURM rerun `42412053` tasks 51–55

| Seed | Run ID | AP | Link |
|-----:|--------|---:|------|
| 0 | `jymcah8k` | 0.7080 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/jymcah8k) |
| 1 | `p3jl40xy` | 0.7044 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/p3jl40xy) |
| 2 | `xoojqs74` | 0.6862 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xoojqs74) |
| 3 | `tlpzwy96` | 0.7031 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tlpzwy96) |
| 4 | `8kqvfsme` | 0.6995 | [run](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/8kqvfsme) |

**Pending best-LR registry** (aggregate W&B when finished): Pep-func `dh75`, Pep-struct
`dh43`/`dh92`, MalNet `dh57` (4 seeds failed), CIFAR `dh34`, VOC `dh15`/`h64_dh12`, COCO `dh34`.

```bash
# Re-aggregate a best-LR group
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_dh_matched_pattern_dh16_lr001 --metric best_test_perf --state finished
```

---

## Initial submission (2026-08-24)

| Tier | SLURM JOBID | Array | Parallel | Time limit | Partition |
|------|------------:|-------|---------:|------------|-----------|
| fast | **41709078** | `1-100%10` | 10 | 48h | `mweber_gpu` |
| slow | **41709082** | `1-40%5` | 5 | 120h | `mweber_gpu` |
| coco | **41709085** | `1-10%2` | 2 | 14d → **48h** (maintenance) | `mweber_gpu` |

Launch commands (reference only — do not re-run whole arrays):

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results

SIGMA_DH_MATCHED_PARALLEL=10 SIGMA_DH_MATCHED_PARTITION=mweber_gpu \
  bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh    # → 41709078

SIGMA_DH_MATCHED_PARALLEL=5 SIGMA_DH_MATCHED_PARTITION=mweber_gpu \
  bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh    # → 41709082

SIGMA_DH_MATCHED_PARALLEL=2 SIGMA_DH_MATCHED_PARTITION=mweber_gpu \
  SIGMA_DH_MATCHED_TIME=14-00:00:00 \
  bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh    # → 41709085
```

---

## Status by tier (initial run)

### Fast — `41709078` (100 tasks)

| Tasks | Family | Slurm | W&B | Notes |
|------:|--------|:-----:|:---:|-------|
| 1–10 | PATTERN `dh16` | ✅ COMPLETED | ✅ 10/10 finished | ~3.5h/run |
| 11–20 | PATTERN `dh4` | ✅ COMPLETED | ✅ 10/10 finished | ~3h/run |
| 21–30 | CLUSTER `dh36` | ✅ COMPLETED | ✅ 10/10 finished | ~1h/run |
| 31–40 | CLUSTER `dh24` | ✅ COMPLETED | ✅ 10/10 finished | ~1h/run |
| 41–50 | MNIST `dh37` | ✅ COMPLETED | ✅ 10/10 finished | ~6.5h/run |
| 51–60 | Pep-func `dh23` | ❌ FAILED `0:53` (~2s) | 0/10 | Instant fail — disk quota |
| 61–70 | Pep-func `dh75` | ❌ FAILED `0:53` (~2s) | 0/10 | Instant fail — disk quota |
| 71–80 | Pep-struct `dh43` | ❌ FAILED `1:0` (~32m) | 0/10 (2 `failed`) | Trained; died writing `results/` |
| 81–90 | Pep-struct `dh92` | ❌ FAILED `1:0` (~32m) | 0/10 (8 `failed`) | Same — `OSError: [Errno 122] Disk quota exceeded` |
| 91–100 | MalNet `dh57` | ❌ FAILED `0:53` (~2–8s) | 0/10 | Instant fail — disk quota |

**Fast tier score:** 50 ✅ · 50 ❌

#### W&B quick metrics (finished families)

| Group | `best_test_perf` (mean ± std, n=5) |
|-------|-------------------------------------|
| `pattern_dh16_lr001` | 0.8723 ± 0.0018 |
| `pattern_dh16_lr01` | 0.8696 ± 0.0004 |
| `pattern_dh4_lr001` | 0.8703 ± 0.0007 |
| `pattern_dh4_lr01` | 0.8690 ± 0.0003 |
| `cluster_dh36_lr001` | 0.7882 ± 0.0008 |
| `cluster_dh36_lr01` | 0.7812 ± 0.0020 |
| `cluster_dh24_lr001` | 0.7872 ± 0.0016 |
| `cluster_dh24_lr01` | 0.7822 ± 0.0014 |
| `mnist_dh37_lr001` | 0.9862 ± 0.0007 |
| `mnist_dh37_lr01` | 0.9652 ± 0.0045 |

### Fast rerun — `42412053` (tasks 51–100)

| Tasks | Family | Slurm | W&B | Notes |
|------:|--------|:-----:|:---:|-------|
| 51–60 | Pep-func `dh23` | ✅ COMPLETED | ✅ 5/5 lr001 finished | AP **0.7002±0.0084** vs paper 0.7080 |
| 61–70 | Pep-func `dh75` | ✅ COMPLETED | (check groups) | rerun after holylabs cleanup |
| 71–80 | Pep-struct `dh43` | ✅ COMPLETED | (check groups) | |
| 81–90 | Pep-struct `dh92` | ✅ COMPLETED | (check groups) | |
| 91–100 | MalNet `dh57` | **4 FAILED** (91, 94, 96, 99) | partial | instant-fail, no Slurm logs — resubmit |

**Rerun score:** 46/50 ✅ · 4 ❌ (MalNet only)

### Slow — `41709082` (40 tasks)

| Tasks | Family | Slurm | W&B | Notes |
|------:|--------|:-----:|:---:|-------|
| 1–5 | CIFAR `dh20` lr001 | ✅ COMPLETED | ✅ 5/5 finished | ~39h/run |
| 6–10 | CIFAR `dh20` lr01 | 🔄 RUNNING | (check) | ~1d7h at 2026-08-28 check |
| 11–20 | CIFAR `dh34` | ⏳ PENDING | 0/10 | `JobArrayTaskLimit` (max 5 parallel) |
| 21–30 | VOC `dh15` | ⏳ PENDING | 0/10 | queued |
| 31–40 | VOC `h64_dh12` | ⏳ PENDING | 0/10 | queued |

**Slow tier score (2026-08-28):** 5 ✅ · 5 🔄 · 30 ⏳

| Group | `best_test_perf` (mean ± std, n=5) |
|-------|-------------------------------------|
| `cifar_dh20_lr001` | 0.7515 ± 0.0057 |

### Coco — `41709085` (10 tasks)

| Tasks | Family | Slurm | W&B | Notes |
|------:|--------|:-----:|:---:|-------|
| 1–2 | COCO `dh34` lr001 s0–1 | ⏱ TIMEOUT | 2 `crashed` | Hit **48h** walltime — **resubmit** with 2d+ |
| 3–4 | COCO `dh34` lr001 s2–3 | 🔄 RUNNING | (check) | ~1d7h at 2026-08-28; 2d walltime |
| 5 | COCO `dh34` lr001 s4 | ⏳ PENDING | — | |
| 6–10 | COCO `dh34` lr01 | ⏳ PENDING | — | |

**Coco tier score (2026-08-28):** 0 ✅ · 2 🔄 · 2 ⏱ timeout · 6 ⏳

---

## Root cause: disk quota on lab filesystem (holylabs), not netscratch

**Symptom:** `OSError: [Errno 122] Disk quota exceeded` at end of Pep-struct runs
when appending to `results/<dataset>_result.txt` in the **repo** (holylabs path).

**Code path:** `GNNPlus/train/custom_train.py` (legacy append to `results/`, not
`$GNNPLUS_OUT_DIR`). Checkpoints and W&B logging use netscratch correctly.

### Storage diagnostics (2026-08-27, holylogin06)

**Holylabs lab home** (`/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/`) — ~**100G** total:

| Path | Size | Notes |
|------|-----:|-------|
| `graph_moes/` | **78G** | Largest consumer — main quota pressure |
| `conda/` | **13G** | Local conda envs |
| `GNNPlus/` | **3.2G** | Repo + `logs_gnnplus/` + `results/` |
| `computed_encodings/` | **2.3G** | |
| `PhenomNN/` | 1.5G | |
| `geometric-algebra-transformer/` | 1.3G | |
| `Hypergraph_Encodings/` | 865M | |
| `Heterogeneity_Profile/` | (check) | Listed in `du` output |
| **netscratch** `gnnplus_results/` | **101G** | Separate filesystem — **OK** |
| netscratch `sigma_dh_matched/` | 547M | This campaign checkpoints |

```text
df -h /n/holylabs/.../rpellegrin  → 2.0P total, 32% used (filesystem not full)
lfs quota -u $USER /n/holylabs    → not available on login node
```

**Interpretation:** Per-user holylabs quota (~100G) is essentially full (`graph_moes`
+ `conda` + repos). Netscratch has plenty of room. GNNPlus is only 3.2G — not the
main offender, but every job still tries to write `results/` and `logs_gnnplus/` here.

### Log forensics (tasks 51 / 81 / 91)

```bash
tail -40 logs_gnnplus/sigma_dh_fast_41709078_51.log   # → No such file
tail -40 logs_gnnplus/sigma_dh_fast_41709078_81.log   # → Pep-struct training epoch 243+
tail -40 logs_gnnplus/sigma_dh_fast_41709078_91.log   # → No such file
grep -i quota logs_gnnplus/sigma_dh_fast_41709078_{51,81,91}.log  # only 81 exists
```

| Task | Log exists? | Meaning |
|------|-------------|---------|
| **51** (Pep-func) | **No** | Failed before Slurm could create stdout log (~2s) — quota at job open |
| **81** (Pep-struct) | **Yes** | Trained through epoch 243+; quota error at end (see W&B `8jph21wh`) |
| **91** (MalNet) | **No** | Same as 51 — instant fail, no log file |

List what logs do exist for this job:

```bash
ls -la logs_gnnplus/sigma_dh_fast_41709078_5*.log logs_gnnplus/sigma_dh_fast_41709078_9*.log 2>/dev/null | head
```

### Mitigation (recommended — symlink `results/` → netscratch)

**Applied for rerun `42412053`** (2026-08-27 holylabs cleanup + `results/` symlink). Re-apply if quota errors return:

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
mkdir -p /n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/results
ln -sfn /n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/results results
touch results/_quota_test && rm results/_quota_test
```

Optional code fix (future): write `*_result.txt` under `cfg.run_dir` or skip when
quota fails (metrics already in W&B).

---

## Reruns

| Date | Tier | SLURM JOBID | Tasks | Command | Outcome |
|------|------|------------:|-------|---------|---------|
| 2026-08-27 | fast | **42403449** | 51–100 | resubmit after `graph_moes` archive + `results` symlink | **FAILED** all tasks `0:53` ~2–7s, no logs — holylabs still ~110G (`Heterogeneity_Profile` not moved yet) |
| 2026-08-27 | fast | **42412053** | 51–100 | resubmit after full holylabs cleanup (~32G) | **DONE** 46/50 — Pep-func/struct OK; MalNet **91,94,96,99** failed |
| — | fast | — | 91,94,96,99 | `SIGMA_DH_MATCHED_ARRAY=91,94,96,99 bash .../submit_sigma_dh_matched_fast.sh` | **Not submitted** — do tomorrow if holylabs OK |
| — | coco | — | 1–2 | `SIGMA_DH_MATCHED_ARRAY=1,2 bash .../submit_sigma_dh_matched_coco.sh` | **Not submitted** — resubmit after 3–4 finish or in parallel |

**Rerun `42412053` details** (final):

- **46/50** array tasks COMPLETED (`sacct -X`)
- Failed: **91, 94, 96, 99** (MalNet `dh57`) — no Slurm stdout logs (instant fail)
- Pep-func `dh23` lr001: W&B **0.7002±0.0084 AP** (5/5 seeds)
- Prior failed rerun: `42403449` (submitted before `Heterogeneity_Profile` data move)

**Rerun `42403449` details** (superseded):

- Logs: `logs_gnnplus/sigma_dh_fast_42403449_<TASK>.log`
- Parallel: 20 GPUs · 48h · `mweber_gpu`
- Preflight fixes: moved `graph_moes` → netscratch archive; `GNNPlus/results` → symlink to netscratch

**Still live (2026-08-31)** — do not duplicate:

- `41709082` / `41709085` — SiGMA d_h-matched slow + COCO tiers
- `42745538` — MalNet d_h fast resubmit (if submitted)
- **Errica TU `grid_select` (Phase 1)** — see `Paper_tu_errica_fair_comparison.md`:
  - **`42750648`** GIN — ✅ **4480/4480 COMPLETED**
  - **`43116245`** GraphSAGE — 🔄 **4768/5040** at last check
  - **`43434937`** GCN — 🔄 **2240** tasks (submitted 2026-08-31)
  - **`43434950`** GAT — 🔄 **2240** tasks (submitted 2026-08-31)
- `42746310` — canonical SiGMA OOM rerun (exploratory; not final table)

**Errica canonical `42673425`:** done (570/630; 60 SiGMA OOM on DD/REDDIT-B @ bs128).

---

## Why fast tier tasks 51–100 failed (`41709078`)

The second half is **not a bug in the array** — it is the designed task map:
tasks 51–100 = Pep-func, Pep-struct, MalNet (after PATTERN / CLUSTER / MNIST in 1–50).

Three **distinct** failure modes, all tied to the **holylabs lab quota** (not netscratch):

| Tasks | Family | Elapsed | Slurm exit | W&B | What happened |
|------:|--------|---------|------------|-----|---------------|
| 51–70 | Pep-func `dh23`/`dh75` | ~2–3 s | `0:53` | none | **Instant crash** before/during startup (no W&B log) |
| 71–88 | Pep-struct `dh43`/`dh92` | ~32 min | `1:0` | `failed` | **Training completed** (epoch 249); died on final `results/*.txt` write |
| 89–100 | MalNet `dh57` | ~2–8 s | `0:53` | none | **Instant crash** (same pattern as Pep-func) |

### Confirmed: Pep-struct (tasks 71–88)

W&B `output.log` for e.g. `8jph21wh` (pepstruct dh92 lr001 seed0):

```text
> Epoch 249: ... | Best so far: epoch 186 ...
OSError: [Errno 122] Disk quota exceeded
  with open(f'results/{cfg.dataset.name}_result.txt','a') as f:
```

- Checkpoints + metrics on **netscratch** were fine.
- Legacy end-of-run append in `GNNPlus/train/custom_train.py` writes to **repo-local**
  `results/` on holylabs → quota full → Slurm marks job **FAILED** even though training finished.

### Likely: Pep-func + MalNet instant fails (tasks 51–70, 89–100)

No W&B runs ⇒ failed within seconds, before meaningful training.

Most probable cause: **same holylabs quota**, hitting earlier in the job lifecycle, e.g.:

- `common_env.sh`: `mkdir -p ... logs logs_gnnplus` in the repo
- Slurm stdout log: `logs_gnnplus/sigma_dh_fast_41709078_<N>.log`
- Dataset cache under repo `datasets/` if `GNNPLUS_DATASET_DIR` was unset at submit time

Tasks 1–50 (Dwivedi) had already written ~50 result lines + logs into the 3.2G repo tree,
filling the per-user holylabs cap before the LRGB/MalNet wave.

**Confirm on cluster:**

```bash
tail -40 logs_gnnplus/sigma_dh_fast_41709078_51.log   # Pep-func instant fail
tail -40 logs_gnnplus/sigma_dh_fast_41709078_81.log   # Pep-struct (quota at end)
grep -i quota logs_gnnplus/sigma_dh_fast_41709078_{51,81,91}.log
```

### Fix before resubmitting 51–100

1. Symlink `results/` → netscratch (see [Mitigation](#mitigation-recommended--symlink-results--netscratch) above).
2. Ensure `export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets` at submit.
3. Resubmit: `SIGMA_DH_MATCHED_ARRAY=51-100 bash .../submit_sigma_dh_matched_fast.sh`

---

## Failure log (detail)

| Job | Task | Family | Exit | Elapsed | Error / diagnosis |
|-----|-----:|--------|------|---------|-------------------|
| 41709078 | 51–70 | Pep-func | `0:53` | ~2s | Holylabs quota at startup (no W&B) |
| 41709078 | 71–88 | Pep-struct | `1:0` | ~32m | Quota on `results/peptides-*_result.txt`; training completed (epoch 249) |
| 41709078 | 89–100 | MalNet | `0:53` | ~2–8s | Holylabs quota at startup (no W&B) |
| 41709085 | 1–2 | COCO lr001 | TIMEOUT | 48:00:00 | Walltime too short vs ~56h/seed anchor |
| 42412053 | 91,94,96,99 | MalNet `dh57` | `1:0` | ~instant | No Slurm log — holylabs quota at job open (likely) |

**Example W&B failure (pepstruct dh92 lr001 seed0):**  
https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/8jph21wh

---

## Diagnostics to run on cluster

```bash
# Quota / space
du -sh /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin
du -sh /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/* | sort -hr | head -20
ls -la /n/netscratch/mweber_lab/Lab/rpellegrin/

# Live jobs
squeue -u $USER -o "%.18i %.30j %.2t %.10M %R"

# History
sacct -j 41709078,41709082,41709085 --format=JobID,State,ExitCode,Elapsed -X

# Instant-fail log
tail -40 logs_gnnplus/sigma_dh_fast_41709078_51.log
```

---

## W&B aggregate (when complete)

```bash
conda activate gnnplus
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_dh_matched_pattern_dh16_lr001
```

Filter tag: `sigma_dh_matched`.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-08-24 | Initial submit: fast `41709078`, slow `41709082`, coco `41709085` on `mweber_gpu` |
| 2026-08-27 | Status review: 55/150 done; fast tasks 51–100 failed (quota); coco 1–2 timeout at 48h |
| 2026-08-27 | Created `rebuttal.md`; reruns 51–100 and coco 1–2 **planned**, not yet launched |
| 2026-08-27 | Storage check: holylabs repo 3.2G; netscratch `gnnplus_results` **101G** (σ_dh 547M); quota is holylabs per-user, not netscratch |
| 2026-08-27 | Perf vs paper: PATTERN/CLUSTER/MNIST ≈ match; **CIFAR dh20 −4.4 pp** at 500k params |
| 2026-08-27 | Added **best-LR run registry** (6 families × 5 seeds, all `lr001`) with W&B run IDs |
| 2026-08-27 | Holylabs ~100G full (`graph_moes` 78G); tasks 51/91 have **no Slurm log** (quota at open) |
| 2026-08-27 | Quota fix: `graph_moes` → netscratch `_archive_graph_moes_2026`; `results/` symlink; **fast rerun `42403449`** tasks 51–100 |
| 2026-08-27 | `Heterogeneity_Profile`: moved `graph_datasets` (82G) + `graph_datasets_with_g_encodings` (71G) → netscratch `heterogeneity_profile_data/` + symlinks; deleted `wandb` + `logs_*` (partial) |
| 2026-08-27 | Rerun `42403449` failed (still over quota); **`42412053`** tasks 51–100 **RUNNING** (Pep-func loading OK) |
| 2026-08-28 | **`42412053` DONE** 46/50; Pep-func `dh23` AP 0.7002±0.0084; MalNet 4 tasks failed |
| 2026-08-28 | Slow `41709082` still 5/40; COCO 3–4 running ~1d7h; fast tier **96/100** combined |
| 2026-08-29 | Errica hybrid Phase 1a: smoke **`42750459`** ✅; GIN `grid_select` **`42750648`** launched (4480 tasks) |
