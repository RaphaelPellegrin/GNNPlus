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

# Quick sanity check (~5 epochs)
sbatch bash_interface/cluster/smoke_test_cifar10_gatedgcn.sh

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
