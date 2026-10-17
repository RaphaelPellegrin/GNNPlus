# TU Tables 17 / 18 / 19 — MMA and SiGMA-lite (cluster)

Paper tables (LoG 2026 App. G):

| Table | \(d_h\) | Models |
| --- | --- | --- |
| 17 | 16 | SiGMA homo/hetero, **MMA** |
| 18 | 4 (~1× GCN width) | SiGMA homo/hetero, **MMA**, baselines |
| 19 | 4 | **SiGMA-lite** only (same heads/width as Tab. 18) |

Recipe (shared with SiGMA hetero ablations): a2g4 = 2 vanilla attn + GCN,GIN,SAGE,GAT; \(L=12\), \(d=64\); random 50/25/25; AdamW; LR ∈ \{1e-3, 1e-2\}; 5 seeds; batch 64 (COLLAB 32, REDDIT-B 16). No other HPs tuned for MMA / SiGMA-lite.

Default SLURM array concurrency is **6 GPUs** (`%6`). Override with `TU_MMA_PARALLEL` / `SIGMA_LITE_PARALLEL`.

## Scripts

| Model | Submit | Run | Config |
| --- | --- | --- | --- |
| MMA | `bash_interface/cluster/submit_tu_mma.sh` | `run_tu_mma.sh` | `configs/tu_sigma_homo_hetero/sigma-hetero-a2g4{-matched}-anchor.yaml` + `gate=none` |
| SiGMA-lite | `bash_interface/cluster/submit_sigma_lite_tu.sh` | `run_sigma_lite_tu.sh` | `configs/sigma_lite/tu-sigma-lite-a2g4.yaml` |
| Both | `bash_interface/cluster/submit_tu_mma_and_sigma_lite.sh` | — | — |

## W&B groups (`weber-geoml-harvard-university/GNNPlus`)

- MMA Tab.17: `tu_hh_<ds>_MMA_{lr001,lr01}`
- MMA Tab.18: `tu_1x_<ds>_MMA_{lr001,lr01}`
- SiGMA-lite Tab.17: `tu_hh_<ds>_SiGMA_lite_{lr001,lr01}`
- SiGMA-lite Tab.18/19: `tu_1x_<ds>_SiGMA_lite_{lr001,lr01}`

Historical MMA-equivalent runs used `SiGMA_ungated` instead of `MMA`.

## Task map (each array = 120)

10 tasks per dataset block (2 LRs × 5 seeds). Order: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY.

- Tab.17: tasks 1–60
- Tab.18: tasks 61–120

## JOBIDs

| Date | Job | Notes |
| --- | --- | --- |
| | | paste after `sbatch` |
