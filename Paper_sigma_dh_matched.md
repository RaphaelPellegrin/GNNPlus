# SiGMA Tab. 3/4 with TU-style d_h shrink (≤500k / ≤1M)

Apply the **Tab. 17 → Tab. 18** recipe from the TU appendix to Dwivedi + LRGB
SiGMA (paper Tables 3–4): keep the best paper heads / depth / train recipe,
shrink **per-head width `d_h`** so params land under **~500k** and/or **~1M**.

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)
Master tracker: [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)

## What Tab. 17 / 18 did (TU)

Same SiGMA (a2g4, L=12, H=64, same heads). Only **per-head width `d_h`** changes:

| Table | `d_h` | Params vs GCN L12/H64 |
|-------|------:|------------------------|
| 17 | 16 | ~1.7× |
| 18 | 4 | ~1.0× |

Heads are **not** dropped. That is different from the budget campaign in
[`Paper_sigma_budget.md`](Paper_sigma_budget.md) (a2g2→a1g1, then shrink H).

## Skip (main already ≤500k)

| Dataset | Main params | Why skip |
|---------|------------:|----------|
| **ZINC** | **450,281** | Already ≤500k (`fotdo14c`) |

MNIST / COCO / MalNet mains already ≤1M → only a ≤500k shrink is launched.
PATTERN / CLUSTER keep the ratio analogs already authored (both under 1M; CLUSTER both under 500k).

## Local param counts (dummy encoder; absolute budgets)

Generator: `scripts/generate_sigma_dh_matched_configs.py`  
Configs: `configs/gated_hybrid/dh_matched/`

| Family | Anchor | Change | Local params | Budget |
|--------|--------|--------|-------------:|--------|
| PATTERN `dh16` | a2g2 GCNE×2 GRIT VN=4 H90 | `d_h` 90→16 | ~843,771 | ≤1M (Tab17 ratio) |
| PATTERN `dh4` | same | `d_h`→4 | ~518,907 | ≤500k (Tab18 ratio) |
| CLUSTER `dh36` | a1g1 GATEDGCN H56 | `d_h` 64→36 | ~437,078 | ≤500k (Tab17 ratio) |
| CLUSTER `dh24` | same | `d_h`→24 | ~254,102 | ≤500k (Tab18 ratio) |
| MNIST `dh37` | a2g2 GATEDGCN×2 H60 | `d_h` 64→37 | ~487,954 | ≤500k |
| CIFAR `dh20` | a8g4 GATEDGCN×4 H35 | `d_h` 256→20 | ~477,070 | ≤500k |
| CIFAR `dh34` | same | `d_h`→34 | ~978,270 | ≤1M |
| Pep-func `dh23` | a1g2 GCN×2 H275 | `d_h` 128→23 | ~491,280 | ≤500k |
| Pep-func `dh75` | same | `d_h`→75 | ~995,316 | ≤1M |
| Pep-struct `dh43` | a1g1 GINE H200 | `d_h` 200→43 | ~497,642 | ≤500k |
| Pep-struct `dh92` | same | `d_h`→92 | ~997,687 | ≤1M |
| VOC `dh15` | a2g2 GATEDGCN×2 H95 | `d_h` 64→15 | ~994,838 | ≤1M |
| VOC `h64_dh12` | same | **H** 95→64 + `d_h`→12 | ~499,213 | ≤500k |
| COCO `dh34` | a1g1 GATEDGCN H52 | `d_h` 52→34 | ~480,349 | ≤500k |
| MalNet `dh57` | a1g1 GCNE H110 (`vcb1cuql`) | `d_h` 64→57 | ~497,617 | ≤500k |

**VOC note:** at H=95, even `d_h=1` is still ~623k (encoder + FFN + depth). ≤500k
requires shrinking **H→64** (as suggested) in addition to `d_h`.

Recount:

```bash
conda activate gnnplus
python scripts/count_tu_model_params.py \
  --cfg configs/gated_hybrid/dh_matched/cifar10-a8g4-dh20.yaml \
  --dim-in 5 --dim-out 10
```

Regen configs (does not overwrite PATTERN/CLUSTER hand configs):

```bash
python scripts/generate_sigma_dh_matched_configs.py
```

## Cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_sigma_dh_matched.sh
```

Smoke (seed 0 of each family):

```bash
SIGMA_DH_MATCHED_ARRAY=1,6,11,16,21,26,31,36,41,46,51,56,61,66,71 \
SIGMA_DH_MATCHED_PARALLEL=15 \
  bash bash_interface/cluster/submit_sigma_dh_matched.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *not submitted yet* |
| **Tasks** | `1-75%20` · 15 families × 5 seeds |
| **Mem / time** | 128GB / 120h |
| **Partition** | `mweber_gpu` |
| **Scripts** | `submit_sigma_dh_matched.sh` / `run_sigma_dh_matched.sh` |
| **Configs** | `configs/gated_hybrid/dh_matched/` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_dh_matched/<fam>_seed<s>/` |
| **Logs** | `logs_gnnplus/sigma_dh_matched_<JOB>_<TASK>.log` |

### Task map

| Tasks | Family | W&B group |
|------:|--------|-----------|
| 1–5 | PATTERN `d_h=16` | `paper_sigma_dh_matched_pattern_dh16` |
| 6–10 | PATTERN `d_h=4` | `paper_sigma_dh_matched_pattern_dh4` |
| 11–15 | CLUSTER `d_h=36` | `paper_sigma_dh_matched_cluster_dh36` |
| 16–20 | CLUSTER `d_h=24` | `paper_sigma_dh_matched_cluster_dh24` |
| 21–25 | MNIST `d_h=37` | `paper_sigma_dh_matched_mnist_dh37` |
| 26–30 | CIFAR `d_h=20` | `paper_sigma_dh_matched_cifar_dh20` |
| 31–35 | CIFAR `d_h=34` | `paper_sigma_dh_matched_cifar_dh34` |
| 36–40 | Pep-func `d_h=23` | `paper_sigma_dh_matched_pepfunc_dh23` |
| 41–45 | Pep-func `d_h=75` | `paper_sigma_dh_matched_pepfunc_dh75` |
| 46–50 | Pep-struct `d_h=43` | `paper_sigma_dh_matched_pepstruct_dh43` |
| 51–55 | Pep-struct `d_h=92` | `paper_sigma_dh_matched_pepstruct_dh92` |
| 56–60 | VOC `d_h=15` | `paper_sigma_dh_matched_voc_dh15` |
| 61–65 | VOC H64 `d_h=12` | `paper_sigma_dh_matched_voc_h64_dh12` |
| 66–70 | COCO `d_h=34` | `paper_sigma_dh_matched_coco_dh34` |
| 71–75 | MalNet `d_h=57` | `paper_sigma_dh_matched_malnet_dh57` |

## Aggregate

```bash
for g in \
  paper_sigma_dh_matched_pattern_dh16 \
  paper_sigma_dh_matched_pattern_dh4 \
  paper_sigma_dh_matched_cluster_dh36 \
  paper_sigma_dh_matched_cluster_dh24 \
  paper_sigma_dh_matched_mnist_dh37 \
  paper_sigma_dh_matched_cifar_dh20 \
  paper_sigma_dh_matched_cifar_dh34 \
  paper_sigma_dh_matched_pepfunc_dh23 \
  paper_sigma_dh_matched_pepfunc_dh75 \
  paper_sigma_dh_matched_pepstruct_dh43 \
  paper_sigma_dh_matched_pepstruct_dh92 \
  paper_sigma_dh_matched_voc_dh15 \
  paper_sigma_dh_matched_voc_h64_dh12 \
  paper_sigma_dh_matched_coco_dh34 \
  paper_sigma_dh_matched_malnet_dh57
do
  python scripts/api_wanndb_query/aggregate_paper_repro.py \
    --group "$g" --metric best_test_perf --state finished
done
```

PATTERN / CLUSTER / MNIST / CIFAR / MalNet Acc is often `best_test_perf` × 100 for %-tables.
Pep-func uses AP; Pep-struct / ZINC MAE; VOC / COCO F1.

## Contrast vs head-drop budget campaign

| Campaign | Heads | How capacity shrinks | Doc |
|----------|-------|----------------------|-----|
| This (`dh_matched`) | **Kept** (a8g4 stays a8g4) | `d_h` (+ VOC `H`) | this file |
| `Paper_sigma_budget` | Often → a1g1 | fewer heads, then H/`d_h`/L | [`Paper_sigma_budget.md`](Paper_sigma_budget.md) |
