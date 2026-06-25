# GNNPlus on Harvard FASRC (mweber_gpu)

Cluster layout mirrors `Heterogeneity_Profile/bash_interface/cluster/` (same conda paths, W&B entity, `mweber_gpu` partition).

**AWS alternative:** see [`bash_interface/aws/RUNNING.md`](../aws/RUNNING.md) (Docker + EC2).

## 1. Clone / sync repo on cluster

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin
git clone <your-GNNPlus-remote> GNNPlus   # first time only
cd GNNPlus
git pull
```

Or rsync from laptop:

```bash
rsync -avz --exclude '.git' --exclude 'results' \
  ~/Desktop/Academic_Research/Repos_GNN/GNNPlus/ \
  rpellegrinext@boslogin08:/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus/
```

## 2. Create conda env (once, interactive)

```bash
salloc --partition test --nodes=1 --cpus-per-task=4 --mem=16GB --time=0-04:00:00
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull origin harvard_cluster
bash bash_interface/cluster/clean_gnnplus_env.sh   # if reinstalling
bash bash_interface/cluster/create_gnnplus_env.sh
```

Env path: `/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs/gnnplus`

Do **not** submit `sbatch` from inside an activated conda env (FASRC quirk). Open a fresh shell on the login node.

## 3. Submit jobs (login node)

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Optional: persistent dataset cache (avoids re-downloading in $HOME)
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
mkdir -p "$GNNPLUS_DATASET_DIR"

# Pre-download GNNBenchmark datasets once (avoids corrupt MNIST/CIFAR zips in arrays):
#   bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10

# Quick sanity check (~5 epochs)
sbatch bash_interface/cluster/smoke_test_cifar10_gatedgcn.sh
sbatch bash_interface/cluster/smoke_test_hybrid_mnist.sh

# Paper CIFAR10 baselines: gcn + gine + gatedgcn × 2 seeds (array 1–6)
sbatch bash_interface/cluster/cifar10_paper_baselines.sh

# Or submit the full paper suite (README repeat counts, auto-downloads datasets):
#   bash bash_interface/cluster/submit_paper_suite.sh --dry-run   # preview
#   bash bash_interface/cluster/submit_paper_suite.sh             # all below
#   bash bash_interface/cluster/submit_paper_suite.sh mnist coco voc peptides-func
#
# | Dataset        | Config        | Seeds | Epochs (yaml) | Download source        |
# |----------------|---------------|-------|---------------|------------------------|
# | CIFAR10        | cifar10       | 2     | 200           | data.pyg.org           |
# | MNIST          | mnist         | 2     | 200           | data.pyg.org           |
# | Peptides-func  | peptides-func | 4     | 300           | Dropbox (OGB loader)   |
# | Peptides-struct| peptides-struct | 4   | 300           | Dropbox (OGB loader)   |
# | COCO-SP        | coco          | 2     | 300           | PyG COCOSuperpixels    |
# | Pascal VOC-SP  | voc           | 2     | 200           | PyG VOCSuperpixels     |
```

Check queue: `squeue -u $USER`

## 4. Weights & Biases

| Variable | Default |
|----------|---------|
| `WANDB_ENTITY` | `weber-geoml-harvard-university` |
| `WANDB_PROJECT` | `GNNPlus` (override e.g. `MOE_6` if you prefer) |
| `WANDB_API_KEY` | **Required** — export before `sbatch` (see below); not stored in git |

Runs appear under the Harvard team entity. Config overrides on the CLI enable W&B without editing YAML:

```bash
wandb.use True wandb.entity weber-geoml-harvard-university wandb.project GNNPlus
```

On the cluster, set your key once (not in the repo):

```bash
cat > ~/.gnnplus_env <<'EOF'
export WANDB_API_KEY="paste-from-https://wandb.ai/authorize"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="GNNPlus"
EOF
chmod 600 ~/.gnnplus_env
source ~/.gnnplus_env   # or add to ~/.bashrc
```

`sbatch` inherits env vars from your login shell if you `export` before submitting, or use `#SBATCH --export=ALL` (already set in job scripts).

## 5. Logs

- Smoke: `logs_gnnplus/smoke_cifar10_gatedgcn_<jobid>.log`
- CIFAR10 array: `logs_gnnplus/cifar10_<array_jobid>_<task>.log`

## 6. Local vs cluster

| Local | Cluster |
|-------|---------|
| `sh run.sh 0 cifar10 2` | `sbatch bash_interface/cluster/cifar10_paper_baselines.sh` |
| `conda activate GNNPlus` | `ENV_NAME=gnnplus` via `common_env.sh` |

No model code changes required — jobs call `python main.py --cfg configs/...` with W&B CLI overrides.

## 7. Troubleshooting

- **`[Errno 28] No space left on device` during env create**: check `df -h $HOME`. If home is **100% full**, conda/pip fail even when the env path is on holylabs. Fix:
  ```bash
  du -sh ~/* ~/.cache/* 2>/dev/null | sort -h | tail -20   # find large dirs
  rm -rf ~/.cache/pip ~/.cache/wandb   # often safe if reproducible
  export TMPDIR=/n/netscratch/mweber_lab/Lab/rpellegrinext/cache/tmp
  export PIP_CACHE_DIR=/n/netscratch/mweber_lab/Lab/rpellegrinext/cache/pip
  rm -rf /n/holylabs/.../conda/envs/gnnplus   # remove broken partial env
  bash bash_interface/cluster/create_gnnplus_env.sh
  ```
  `create_gnnplus_env.sh` redirects caches off `$HOME` and aborts if pip uses `~/.local`.
- **`Defaulting to user installation` / torch in `~/.local`**: run `clean_gnnplus_env.sh` then recreate. Verify with `PYTHONNOUSERSITE=1 python -c "import torch; print(torch.__file__)"` — path must be under `.../conda/envs/gnnplus/`.
- **`GNNPlus import check failed`**: run `clean_gnnplus_env.sh` + `create_gnnplus_env.sh`.
- **CUDA / PyG mismatch**: recreate env with `create_gnnplus_env.sh` (cu121 wheels for cuda/12.9).
- **Dataset download slow**: set `GNNPLUS_DATASET_DIR` to lab scratch (see above).
- **`BadZipFile` on MNIST/CIFAR10**: parallel array jobs raced on download. Pre-download once:
  `bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10`
- **COCO OOM at 64GB**: resubmit with `--mem=128GB`.
- **GLIBC warnings** for `pyg-lib` / `torch-sparse`: safe to ignore; training works without them.
- **Missing raw `.zip` after prep**: normal — PyG keeps `processed/*.pt` only.

## 8. Session notes — resubmit 2026-06-20

Pick up here after dataset fix + paper-suite resubmit.

### Paths & env (login node)

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull origin harvard_cluster
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export ENV_NAME=gnnplus
conda deactivate 2>/dev/null || true   # do NOT sbatch from activated conda
mkdir -p logs_gnnplus
```

Conda env (used inside jobs via `common_env.sh`):
`/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs/gnnplus`

W&B: https://wandb.ai/weber-geoml-harvard-university/GNNPlus

### Dataset cache (verified OK)

| Dataset | Path | Status |
|---------|------|--------|
| MNIST | `.../GNNBenchmarkDataset/MNIST/processed/` | OK (Jun 20 prep) |
| CIFAR10 | `.../GNNBenchmarkDataset/CIFAR10/processed/` | OK (Jun 19; ~1.2G `.pt` files) |
| Duplicate | `.../CIFAR10/CIFAR10/` | **removed** (~2.5G saved) |

Loader expects `GNNPLUS_DATASET_DIR/GNNBenchmarkDataset/<NAME>/`, not `GNNPLUS_DATASET_DIR/<NAME>/`.

### Original paper suite (2026-06-19)

Submitted: `bash bash_interface/cluster/submit_paper_suite.sh mnist peptides-func coco voc` (no CIFAR10).

| Job ID | Dataset | Tasks |
|--------|---------|-------|
| 23395149 | mnist | 6 |
| 23395161 | peptides-func | 12 |
| 23395162 | coco | 6 (all failed — OOM at 64GB) |
| 23395163 | voc | 6 |

**Finished** (from `grep Finished logs_gnnplus/gnnplus_*_233951*.log`):

| Task | Run |
|------|-----|
| mnist 2 | gcn seed 1 |
| peptides 1–3,5,7,9–12 | gcn s0–s2, gine s0/s2, gatedgcn s0–s3 |
| voc 1,5,6 | gcn s0, gatedgcn s0/s1 |

**Still missing after original batch:**

| Dataset | Missing tasks | Models / seeds |
|---------|---------------|----------------|
| CIFAR10 | all 1–6 | never submitted |
| MNIST | 1,3,4,5,6 | gcn s0; gine s0/s1; gatedgcn s0/s1 |
| peptides-func | 4,6,8 | gcn s3; gine s1; gine s3 |
| coco | 1–6 | all (use `--mem=128GB`) |
| voc | 2,3,4 | gcn s1; gine s0/s1 |

### Resubmit batch A (2026-06-20, first paste)

| Job ID | Dataset | Array |
|--------|---------|-------|
| 23643326 | cifar10 | 1–6 %6 |
| 23643329 | mnist | 1,3,4,5,6 %4 |
| 23643331 | peptides-func | 4,6,8 %3 |
| 23643332 | coco | 1–6 %4, mem=128GB |
| 23643338 | voc | 2,3,4 %3 |

### Resubmit batch B (2026-06-20, second paste — **duplicate of batch A**)

| Job ID | Dataset | Array |
|--------|---------|-------|
| 23643461 | cifar10 | 1–6 %6 |
| 23643462 | mnist | 1,3,4,5,6 %6 |
| 23643464 | peptides-func | 4,6,8 %6 |
| 23643465 | coco | 1–6 %4, mem=128GB |
| 23643466 | voc | 2,3,4 %4 |

At submit time, **23643326 tasks 1–4 were already running**; batch B was queued behind them (all `PD`).

**Action if still pending:** cancel one duplicate set to avoid double W&B runs and wasted GPU:

```bash
# Keep batch A (already running); cancel batch B if still pending:
scancel 23643461 23643462 23643464 23643465 23643466

# Or keep whichever batch is running and cancel the other.
squeue -u $USER
```

### Parallelism

- Multiple `sbatch` lines queue immediately — SLURM runs jobs in parallel up to partition capacity.
- `--array=1-N%M` limits **M concurrent array tasks per job** (not sequential training).
- `submit_paper_suite.sh` uses `%6` (mnist/cifar/peptides) and `%4` (coco/voc).

### Monitor & audit

```bash
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %.10l"
grep -h "Finished" logs_gnnplus/*.log | sort -u
tail -f logs_gnnplus/cifar10_23643326_1.log
```

Log patterns: `logs_gnnplus/cifar10_<jobid>_<task>.log`, `logs_gnnplus/gnnplus_<dataset>_<jobid>_<task>.log`

### Task index (2-seed datasets: mnist, cifar10, coco, voc)

| Task | Model | Seed |
|------|-------|------|
| 1 | gcn | 0 |
| 2 | gcn | 1 |
| 3 | gine | 0 |
| 4 | gine | 1 |
| 5 | gatedgcn | 0 |
| 6 | gatedgcn | 1 |

Peptides-func (4 seeds): tasks 1–4 gcn, 5–8 gine, 9–12 gatedgcn.

### Copy-paste resubmit (missing tasks only)

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export ENV_NAME=gnnplus
conda deactivate 2>/dev/null || true

sbatch --job-name=gnnplus_cifar10 --array=1-6%6 --time=48:00:00 --mem=64GB \
  --export=ALL,ENV_NAME=gnnplus bash_interface/cluster/cifar10_paper_baselines.sh

sbatch --job-name=gnnplus_mnist --array=1,3,4,5,6%6 --time=48:00:00 --mem=64GB \
  --export=ALL,DATASET=mnist,NUM_SEEDS=2,ENV_NAME=gnnplus \
  bash_interface/cluster/run_paper_array.sh

sbatch --job-name=gnnplus_peptides_func --array=4,6,8%6 --time=96:00:00 --mem=64GB \
  --export=ALL,DATASET=peptides-func,NUM_SEEDS=4,ENV_NAME=gnnplus \
  bash_interface/cluster/run_paper_array.sh

sbatch --job-name=gnnplus_coco --array=1-6%4 --time=96:00:00 --mem=128GB \
  --export=ALL,DATASET=coco,NUM_SEEDS=2,ENV_NAME=gnnplus \
  bash_interface/cluster/run_paper_array.sh

sbatch --job-name=gnnplus_voc --array=2,3,4%4 --time=48:00:00 --mem=64GB \
  --export=ALL,DATASET=voc,NUM_SEEDS=2,ENV_NAME=gnnplus \
  bash_interface/cluster/run_paper_array.sh
```

### PATTERN / CLUSTER / MalNet — best-hybrid Bayes sweeps

After `git pull` (needs `submit_best_hybrid_sweep_suite.sh` on cluster):

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export ENV_NAME=gnnplus
conda deactivate 2>/dev/null || true

# Creates W&B sweeps + launches 8 agents × 4 runs each (pattern/cluster 128GB, mal 64GB)
bash bash_interface/cluster/submit_best_hybrid_sweep_suite.sh

# Or one dataset:
bash bash_interface/cluster/submit_best_hybrid_sweep_suite.sh pattern
```

Sweeps: `pattern_best_hybrid_sweep.yaml`, `cluster_best_hybrid_sweep.yaml`, `mal_best_hybrid_sweep.yaml` → W&B projects `GNNplus_best_hybrid-pattern|cluster|mal`.
