# SiGMA: Learning under Graph-Level Heterogeneity with Gated Message-Passing and Attention

**Lukas Fesser, Raphael Pellegrin, Melanie Weber** (equal contribution)

*Proceedings of the Fifth Learning on Graphs Conference (LoG 2026)*

This is the official implementation of **SiGMA** (**Si**gmoid **G**ated **M**essage-Passing and
**A**ttention), a gated hybrid graph neural network for graph-level learning.

## Overview

Graph neural networks usually apply one fixed computational template to every input graph: the same
depth, the same local message-passing operator and the same global attention mechanism. Yet graphs
within a single dataset can differ widely in how hard they are to predict and in which computation
suits them best (*graph-level heterogeneity*): different graphs prefer different message-passing
operators and different depths.

SiGMA addresses this by placing **heterogeneous message-passing (MP) heads and attention heads in
the same layer** and modulating **each head with a learned node-wise sigmoid gate**. The gates let the
model regulate, per node, per layer and per input graph, which local or global mechanisms are active.
Because gated heads enter through a residual update, suppressed layers barely change the
representation, so the model also learns an input-dependent effective depth.

## Model architecture

At layer $`\ell`$, SiGMA normalises the node representations, $`\hat h_i^\ell = \mathrm{Norm}(h_i^\ell)`$,
and runs a set of attention heads $`\mathcal{H}^\ell_{\mathrm{att}}`$ and a set of MP heads
$`\mathcal{H}^\ell_{\mathrm{mp}}`$ in parallel (either set may be empty). Each head outputs a
$`d_h`$-dimensional vector and is gated before the heads are combined.

**Gated attention heads.** The gate is computed jointly with the query (head-wise gating shown):

```math
W^a_{qg}\,\hat h_i^\ell = \big[\, q_i^a \,\Vert\, g_i^{\mathrm{att},a} \,\big] \in \mathbb{R}^{d_h+1},
\qquad k_j^a = W^a_k \hat h_j^\ell, \qquad v_j^a = W^a_v \hat h_j^\ell,
```

```math
\tilde a_i^{\mathrm{att},a} = \sigma\big(g_i^{\mathrm{att},a}\big) \sum_{j} \alpha^a_{ij}\, v_j^a,
\qquad
\alpha^a_{ij} = \mathrm{softmax}_j\Big( (q_i^a)^\top k_j^a / \sqrt{d_h} + B_{ij} \Big),
```

where $`B_{ij}`$ masks disallowed pairs (e.g. nodes of different graphs in a batch).

**Gated message-passing heads.** Each MP head $`p`$ computes a head-specific representation and gate,
then applies its graph operator $`\Phi_p`$ (GCN, GIN, GINE, GatedGCN, GraphSAGE, GAT, UniConv, ...):

```math
W^p_{hg}\,\hat h_i^\ell = \big[\, u_i^p \,\Vert\, g_i^{\mathrm{mp},p} \,\big],
\qquad
\tilde a_i^{\mathrm{mp},p} = \sigma\big(g_i^{\mathrm{mp},p}\big)\, \Phi_p\big(\{u_j^p\}_j, \mathcal{E}\big)_i .
```

**Fusion.** The gated heads are concatenated, projected and added through a residual update,
optionally followed by a feed-forward block:

```math
h_i^{\ell+1} = h_i^\ell + W^\ell_{\mathrm{out}}\,
\mathrm{Concat}\Big( \{\tilde a_i^{\mathrm{att},a}\}_{a \in \mathcal{H}^\ell_{\mathrm{att}}},\;
\{\tilde a_i^{\mathrm{mp},p}\}_{p \in \mathcal{H}^\ell_{\mathrm{mp}}} \Big).
```

We write `aNgM` for a layer with `N` attention heads and `M` MP heads (e.g. `a1g2`). The design
space ranges from purely local MPNN-like models (`a0gM`) to pure graph transformers (`aNg0`).

### Gating variants and MMA


| Variant                                            | Config                         | Gate                                           |
| -------------------------------------------------- | ------------------------------ | ---------------------------------------------- |
| SiGMA, head-wise                                   | `gnn.hybrid.gate: headwise`    | One scalar gate per node and head              |
| SiGMA, element-wise                                | `gnn.hybrid.gate: elementwise` | One gate per node and channel ($`d_h`$ per head) |
| **MMA** (Mixture of Message-Passing and Attention) | `gnn.hybrid.gate: none`        | No gates: heads are concatenated at full scale |


`gnn.hybrid.mp_gate` overrides the gate for the MP heads only, e.g. gated attention with ungated  
message passing (`gate: headwise`, `mp_gate: none`).

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



## How to run



### Quick start: SiGMA and MMA on ZINC

Two example scripts train on ZINC with the paper's settings (Tables 11 and 14): `a1g1` with one
vanilla attention head and one UniConv MP head, 12 layers, hidden width 64, $`d_h = 32`$.

```bash
bash bash_interface/run_sigma_zinc.sh   # SiGMA (head-wise gates) -> configs/sigma/zinc-sigma.yaml
bash bash_interface/run_mma_zinc.sh     # MMA (no gates)          -> configs/sigma/zinc-mma.yaml
```

ZINC is downloaded automatically. Both scripts accept `SEED`, `DATASET_DIR` (default `./datasets`)
and `WANDB=True`, and forward any extra arguments as config overrides:

```bash
SEED=1 WANDB=True bash bash_interface/run_sigma_zinc.sh optim.max_epoch 100
```

On a SLURM cluster, submit them from the repo root: `sbatch bash_interface/run_sigma_zinc.sh`.

### Any config

```bash
python main.py --cfg <config.yaml> [key value ...]
```

Every field of the YAML can be overridden on the command line, e.g.
`gnn.hybrid.gate none` turns any SiGMA config into MMA.

### Configuring a SiGMA model

Set `model.type: hybrid_gnn` and configure the hybrid block under `gnn.hybrid`:


| Field                             | Values                            | Meaning                                                                        |
| --------------------------------- | --------------------------------- | ------------------------------------------------------------------------------ |
| `num_attn_heads`, `num_gnn_heads` | int                               | `N` and `M` in `aNgM`                                                          |
| `d_h`                             | int                               | Per-head width                                                                 |
| `gnn_types`                       | comma list, one entry per MP head | `GCN`, `GCNE`, `GIN`, `GINE`, `GAT`, `SAGE`, `GATEDGCN`, `GGNN`, `UNIGCN`, ... |
| `attn_type`                       | `vanilla`, `grit`                 | Dense attention or GRIT attention                                              |
| `attn_mask`                       | `full`, `graph_restricted`        | Attend to all nodes of the same graph, or only along graph edges               |
| `gate` / `mp_gate`                | `headwise`, `elementwise`, `none` | Gating mode (see above)                                                        |
| `norm`                            | `layernorm`, `rmsnorm`, `none`    | Pre-head normalisation                                                         |


`GATEDGCN` and `GCNE` wrap the full edge-aware GatedGCN+ / GCN+ layers (including their FFN);
`RESGATEDGCN` and `GCNE_CONV` are the raw-convolution variants. `UNIGCN` is the unitary
(complex-valued Taylor) graph convolution, configured by `gnn.use_hermitian`,
`gnn.unitary_taylor_order` and `gnn.unitary_return_real`.

### Additional features

- **Synthetic routing tasks.** `dataset.format: PyG-GcnGinRouting` and `PyG-GinDepthRouting`.



## Code structure


| Path                                    | Contents                                             |
| --------------------------------------- | ---------------------------------------------------- |
| `GNNPlus/layer/gated_hybrid_layer.py`   | The SiGMA layer (gated attention + MP heads, fusion) |
| `GNNPlus/network/hybrid_gnn.py`         | The full SiGMA model (`model.type: hybrid_gnn`)      |
| `GNNPlus/config/gated_hybrid_config.py` | All `gnn.hybrid.*` options and their defaults        |
| `configs/sigma/`                        | Example SiGMA and MMA configs                        |
| `bash_interface/`                       | Example run scripts                                  |
| `unittests/`                            | Unit tests (`python -m pytest unittests`)            |




## Citation

If you use this code, please cite:

```bibtex
@inproceedings{fesser2026sigma,
  title     = {{SiGMA}: Learning under Graph-Level Heterogeneity with Gated Message-Passing and Attention},
  author    = {Lukas Fesser and Raphael Pellegrin and Melanie Weber},
  booktitle = {Proceedings of the Fifth Learning on Graphs Conference (LoG 2026)},
  year      = {2026}
}
```



## Codebase

This code is built on the **GNN+** codebase,
[github.com/LUOyk1999/GNNPlus](https://github.com/LUOyk1999/GNNPlus) (Luo, Shi and Wu, *Can Classic
GNNs Be Strong Baselines for Graph-level Tasks?*, ICML 2025), which is itself based on
[GraphGPS](https://github.com/rampasek/GraphGPS). The GNN+ baselines (`configs/gcn`, `configs/gine`,
`configs/gatedgcn`, `run.sh`) are unchanged and still runnable. If you use them, please also cite:

```bibtex
@inproceedings{luo2025can,
  title     = {Can Classic {GNN}s Be Strong Baselines for Graph-level Tasks? Simple Architectures Meet Excellence},
  author    = {Yuankai Luo and Lei Shi and Xiao-Ming Wu},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=ZH7YgIZ3DF}
}
```

