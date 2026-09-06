# TU Tab.17/18 — SiGMA gated vs ungated × depth × d_h × H

Re-run **hetero a2g4** gated vs ungated under the Tab.17/18 protocol, sweeping
capacity knobs that those tables held fixed.

| Knob | Paper Tab.17 | Paper Tab.18 | **This grid** |
|------|-------------:|-------------:|---------------|
| `L` (`layers_mp`) | 12 | 12 | **{1, 2, 4, 8, 16}** (L=12 omitted) |
| `d_h` | 16 | 4 | **{1, 2, 4, 16}** |
| `H` (`dim_inner`) | 64 | 64 | **{64, 8}** (+ fill **{4, 2}**) |
| Heads | a2g4 hetero | a2g4 hetero | same |
| Gate | headwise only | headwise only | **headwise vs none** |

**Job count:** 6 ds × 5 L × 4 d_h × 2 H × 2 gate × 2 LR × 5 seeds = **4800**.

Hypothesis: map where Δ(gated − ungated) becomes reliably positive (expect small
`d_h` / small `H` / deep `L`).

Related: [`Paper_tu_sigma_homo_hetero.md`](Paper_tu_sigma_homo_hetero.md) ·
[`Paper_tu_sigma_dh_extreme.md`](Paper_tu_sigma_dh_extreme.md)

---

## Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# smoke
TU_LDHH_ARRAY=1,11 TU_LDHH_PARALLEL=2 \
  bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh

# MUTAG only (recommended first)
TU_LDHH_ARRAY=1-800 bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh

# later datasets (800 each):
# TU_LDHH_ARRAY=801-1600   # ENZYMES
# TU_LDHH_ARRAY=1601-2400  # PROTEINS
# TU_LDHH_ARRAY=2401-3200  # COLLAB
# TU_LDHH_ARRAY=3201-4000  # IMDB-BINARY
# TU_LDHH_ARRAY=4001-4800  # REDDIT-BINARY
```

| Field | Value |
|-------|-------|
| **Smoke** | ✅ **`44876758`** · tasks `1,11` · `%2` |
| **MUTAG** | ✅ **`44876760`** · tasks `1-800` · `%20` · **800/800 COMPLETED** |
| **ENZYMES** | ✅ **`44897907`** · tasks `801-1600` · `%20` |
| **Other datasets** | 🛑 PROTEINS…REDDIT not yet (`1601-4800`) |
| **Submit** | `bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh` |
| **Worker** | `bash_interface/cluster/run_tu_sigma_depth_dh_h.sh` |
| **Base cfg** | `configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml` |
| **W&B** | `tu_L<k>_dh<m>_H<h>_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/` |
| **Logs** | `logs_gnnplus/tu_LdhH_<JOBID>_<TASK>.log` |

### Task map

| Tasks | Dataset |
|------:|---------|
| 1–800 | MUTAG |
| 801–1600 | ENZYMES |
| 1601–2400 | PROTEINS |
| 2401–3200 | COLLAB |
| 3201–4000 | IMDB-BINARY |
| 4001–4800 | REDDIT-BINARY |

Within each dataset, nest order: `L → d_h → H → gate → lr → seed`.  
Per `(L,d_h,H)` block of 20: gated lr001 · gated lr01 · ungated lr001 · ungated lr01.

Smoke: **1** = MUTAG L1 dh1 H64 gated lr001 seed0 · **11** = ungated.

---

## H ∈ {4, 2} fill (narrow trunk)

Same L / d_h / gate / LR / seed map as above; only `TU_LDHH_HS="4 2"` changes
which `H` values occupy the H slots (task IDs 1–4800 again, but **different**
W&B groups / out dirs — safe alongside the H=64/8 campaign).

```bash
# smoke: MUTAG L1 dh1 H4 gated+ungated
TU_LDHH_HS="4 2" TU_LDHH_ARRAY=1,11 TU_LDHH_PARALLEL=2 \
  bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh

# MUTAG H=4/2 (800), then ENZYMES…
TU_LDHH_HS="4 2" TU_LDHH_ARRAY=1-800 bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh
# TU_LDHH_HS="4 2" TU_LDHH_ARRAY=801-1600 bash ...
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *paste JOBID* |
| **H list** | `4 2` via `TU_LDHH_HS` |
| **Tasks** | same blocks as H=64/8 (800/ds) |
| **W&B** | `tu_L*_dh*_H{4,2}_*` |

Plots (after MUTAG): `fig_mutag_delta_heatmap_H4.png`, `…_H2.png` via
`python scripts/api_wanndb_query/aggregate_tu_depth_dh_h_mutag.py --H 4 2`.

---

## Analysis (after runs)

Protocol: better LR mean±std; Δ = gated − ungated; paired *t* by seed.

Interesting slices:

1. Fix `H=64`, plot Δ vs `d_h` for each `L` (recover Tab.17/18-style + extremes).
2. Fix `d_h∈{4,16}`, compare `H=64` vs `H=8`.
3. Heatmap: rows=`L`, cols=`d_h`, color=Δ (one panel per dataset / H).

```python
# sketch: query W&B groups tu_L*_dh*_H*_<ds>_SiGMA_{hetero,ungated}_*
```

---

## Notes

- Paper **L=12** is not in this grid; add later if needed for exact Tab.17/18 redo.
- `H=8` with `d_h=16` is allowed (concat 96 → project to 8) but unusual.
- Overlaps prior `tu_dh{1,2}_*` / Tab.17/18 cells at specific `(L,d_h,H)` only if
  those match (prior extreme used **L=12**; this grid does not).
