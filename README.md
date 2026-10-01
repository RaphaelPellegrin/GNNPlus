# SiGMA: Learning under Graph-Level Heterogeneity with Gated Message-Passing and Attention (LoG 2026)

**Lukas Fesser\*, Raphael Pellegrin\*, Melanie Weber** (\*equal contribution)
— *Proceedings of the Fifth Learning on Graphs Conference (LoG 2026)*

This is the official code for **SiGMA** (**Si**gmoid **G**ated **M**essage-Passing and **A**ttention).
It is built on top of the [GNN+](https://github.com/LUOyk1999/GNNPlus) codebase (ICML 2025), which is
itself based on [GraphGPS](https://github.com/rampasek/GraphGPS); see [Acknowledgements](#acknowledgements).

## What is SiGMA?

Graphs within the same dataset can differ widely in how hard they are to predict and in how much local
versus non-local computation they need (*graph-level heterogeneity*). SiGMA is a gated hybrid GNN: each
layer runs one or more **message-passing (MP) heads** and **attention heads in parallel**, multiplies every
head by a **learned node-wise sigmoid gate**, then concatenates, projects, and adds a residual update. The
gates let the model switch individual heads, or whole layers, on and off per node and per input graph.

We write **`aNgM`** for a layer with `N` attention heads and `M` MP heads (e.g. `a1g2`).

### SiGMA vs. MMA

**MMA** (Mixture of Message-Passing and Attention) is the same architecture **without gates**. Both are
the `hybrid_gnn` model; only one config field differs:

| Model | `gnn.hybrid.gate` | Description |
| - | - | - |
| **SiGMA** | `headwise` or `elementwise` | One sigmoid gate per head (`headwise`) or per channel (`elementwise`) |
| **MMA** | `none` | Heads are concatenated without gating |

`gnn.hybrid.mp_gate` optionally overrides the gate for the MP heads only (e.g. gated attention with
ungated message passing: `gate: headwise`, `mp_gate: none`).

## Installation

Tested with Python 3.10, PyTorch 2.2.0 and PyTorch Geometric 2.3.1.

```bash
conda create -n sigma python=3.10
conda activate sigma

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install -r requirements-cluster.txt
pip install -e .
```

Use the CUDA tag (`cu118`, `cu121`, ...) that matches your driver.

## Quick start: SiGMA and MMA on ZINC

Two example scripts train on ZINC with the hyperparameters from Tables 11 and 14 of the paper
(`a1g1`: one vanilla attention head + one UniConv MP head, 12 layers, hidden width 64, `d_h = 32`):

```bash
bash bash_interface/run_sigma_zinc.sh   # SiGMA (headwise gates)  -> configs/sigma/zinc-sigma.yaml
bash bash_interface/run_mma_zinc.sh     # MMA (no gates)          -> configs/sigma/zinc-mma.yaml
```

Both scripts accept `SEED`, `DATASET_DIR` (default `./datasets`; ZINC downloads automatically) and
`WANDB=True`, and forward any extra arguments as YACS overrides:

```bash
SEED=1 WANDB=True bash bash_interface/run_sigma_zinc.sh optim.max_epoch 100
```

They also work as SLURM jobs (`sbatch bash_interface/run_sigma_zinc.sh`, submitted from the repo root).
Any config can be run directly with `python main.py --cfg <config.yaml> [key value ...]`.

## Configuring a SiGMA model

Set `model.type: hybrid_gnn` and configure the hybrid block under `gnn.hybrid`:

| Field | Values | Meaning |
| - | - | - |
| `num_attn_heads`, `num_gnn_heads` | int | `N` and `M` in `aNgM` |
| `d_h` | int | Per-head width |
| `gnn_types` | comma list, one per MP head | `GCN`, `GCNE`, `GIN`, `GINE`, `GAT`, `SAGE`, `GATEDGCN`, `GGNN`, `UNIGCN`, ... |
| `attn_type` | `vanilla`, `grit`, `physics` | Dense attention, GRIT attention, or Transolver++ physics attention |
| `attn_mask` | `full`, `graph_restricted` | Attend to all nodes of the same graph, or only along graph edges |
| `gate` / `mp_gate` | `headwise`, `elementwise`, `none` | Gating mode (see above) |
| `norm` | `layernorm`, `rmsnorm`, `none` | Pre-head normalisation |

`GATEDGCN` and `GCNE` wrap the full edge-aware GNN+ layers (including their FFN); `RESGATEDGCN` and
`GCNE_CONV` are the raw-convolution variants. `UNIGCN` is the unitary (complex-valued Taylor) graph
convolution; see `gnn.use_hermitian`, `gnn.unitary_taylor_order` and `gnn.unitary_return_real`.

## Additional features

- **Datasets.** Synthetic routing tasks (`dataset.format: PyG-GcnGinRouting`, `PyG-GinDepthRouting`).
- **Fair TU evaluation.** The fixed 10-fold splits of Errica et al. (ICLR 2020) are vendored under
  `splits/errica/`; enable with `dataset.split_mode: errica-cv-10` and `dataset.split_index: <fold>`.
- **Gate diagnostics.** `gnn.hybrid.log_gate_stats` logs per-layer gate statistics to W&B, including
  per-difficulty gates on the synthetic routing tasks. `HybridGNN.forward(batch, gate_override=...)`
  replaces the learned gates at evaluation time (`ones` or per-graph `mean`).
- **Activation dumps.** `GNNPlus/experiments/last_layer_activations.py` writes per-graph, per-layer
  activation norms as CSVs and plots.
- **Training.** Early stopping (`train.early_stop_patience`, `train.early_stop_use_loss`) and extra W&B
  tags via the `WANDB_EXTRA_TAGS` environment variable (comma-separated).

## Tests

```bash
python -m pytest unittests
```

## Citation

If you use this code, please cite SiGMA:

```bibtex
@inproceedings{fesser2026sigma,
  title     = {{SiGMA}: Learning under Graph-Level Heterogeneity with Gated Message-Passing and Attention},
  author    = {Lukas Fesser and Raphael Pellegrin and Melanie Weber},
  booktitle = {Proceedings of the Fifth Learning on Graphs Conference (LoG 2026)},
  year      = {2026}
}
```

## Acknowledgements

This repository is a fork of **GNN+** by Yuankai Luo, Lei Shi and Xiao-Ming Wu; the GNN+ baselines
(`configs/gcn`, `configs/gine`, `configs/gatedgcn`, `run.sh`) are unchanged and still runnable, e.g.
`python main.py --cfg configs/gcn/peptides-func.yaml --repeat 2 seed 0`. Please also cite their work:

```bibtex
@inproceedings{luo2025can,
  title     = {Can Classic {GNN}s Be Strong Baselines for Graph-level Tasks? Simple Architectures Meet Excellence},
  author    = {Yuankai Luo and Lei Shi and Xiao-Ming Wu},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=ZH7YgIZ3DF}
}
```

GNN+ is in turn based on the [GraphGPS](https://github.com/rampasek/GraphGPS) codebase.
