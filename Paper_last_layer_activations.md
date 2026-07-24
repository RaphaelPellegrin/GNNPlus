# All-layer activations per graph (TU datasets)

Show that different graphs induce different activations **at every layer**.

| | |
|--|--|
| **Y-axis** | Mean over nodes of \(\|h_v\|_2\) after each GNN/hybrid layer (before classifier) |
| **X-axis** | Graph index |
| **Snapshots** | `mid` (epoch ≈ max/2), `last` (final epoch), `best` (val-best) |
| **Plots** | all-layer overlay, layer×graph heatmap, last-layer alone |
| **Also** | CSV per snapshot; `summary.json` with **test Acc**; `mid.pt` / `last.pt` / `best.pt` |

No reusable TU checkpoints → **retrain**, then dump activations.

## Models

| Slot | Dataset | Config |
|------|---------|--------|
| 1 | MUTAG | `configs/heterogeneity/mutag-sigma.yaml` |
| 2 | ENZYMES | `configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml` |
| 3 | PROTEINS | `configs/heterogeneity/proteins-sigma.yaml` |

## Local smoke

```bash
cd /Users/pellegrinraphael/Desktop/Academic_Research/Repos_GNN/GNNPlus

python -m pytest unittests/test_last_layer_activations.py -q

# MUTAG: short train → mid/last/best all-layer plots
python scripts/heterogeneity/run_last_layer_activations.py \
  --cfg configs/heterogeneity/mutag-sigma.yaml \
  --max_epoch 6 --seed 0 --no-wandb \
  --output_dir results/activations/mutag_smoke

# optional ENZYMES (heavy — keep epochs tiny)
python scripts/heterogeneity/run_last_layer_activations.py \
  --cfg configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml \
  --max_epoch 2 --seed 0 --no-wandb \
  --output_dir results/activations/enzymes_smoke
```

Look under `results/activations/mutag_smoke/{mid,last,best}/` for:
- `*_all_layers_by_index.png` (main figure: one curve per layer)
- `*_all_layers_heatmap.png`
- `*_last_layer_by_index.png`

## Cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_last_layer_activations_tu.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *paste JOBID* |
| **Outs** | `$GNNPLUS_OUT_DIR/activations/<ds>_<tag>_seed<S>/{mid,last,best}/` |
| **W&B** | `layer_act_{mutag,enzymes,proteins}` |

## Appendix Acc

| Dataset | Model | Acc (mean±std) |
|---------|-------|----------------|
| MUTAG | SiGMA | 🛑 |
| ENZYMES | SiGMA ogpkubk9 | GNNPlus plateau ≈ 0.467±0.049 |
| PROTEINS | SiGMA | 🛑 |
