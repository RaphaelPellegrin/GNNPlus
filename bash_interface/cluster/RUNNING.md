# GNNPlus on Harvard FASRC (mweber_gpu)

Cluster layout mirrors `Heterogeneity_Profile/bash_interface/cluster/` (same conda paths, W&B entity, `mweber_gpu` partition).

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
```

Check queue: `squeue -u $USER`

## 4. Weights & Biases

| Variable | Default |
|----------|---------|
| `WANDB_ENTITY` | `weber-geoml-harvard-university` |
| `WANDB_PROJECT` | `GNNPlus` (override e.g. `MOE_6` if you prefer) |
| `WANDB_API_KEY` | same default as Heterogeneity_Profile scripts |

Runs appear under the Harvard team entity. Config overrides on the CLI enable W&B without editing YAML:

```bash
wandb.use True wandb.entity weber-geoml-harvard-university wandb.project GNNPlus
```

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

- **`GNNPlus import check failed`**: run `create_gnnplus_env.sh` or set `ENV_NAME` to an existing env.
- **CUDA / PyG mismatch**: recreate env with `create_gnnplus_env.sh` (cu121 wheels for cuda/12.9).
- **Dataset download slow**: set `GNNPLUS_DATASET_DIR` to lab scratch (see above).
