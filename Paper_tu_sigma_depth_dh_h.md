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
| **Smoke H4** | ✅ **`44899287`** · tasks `1,11` · `%2` |
| **MUTAG H∈{4,2}** | ✅ **`44899291`** · tasks `1-800` · `%20` |
| **H list** | `4 2` via `TU_LDHH_HS` |
| **Tasks** | same blocks as H=64/8 (800/ds) |
| **W&B** | `tu_L*_dh*_H{4,2}_*` |

Plots (after MUTAG): `fig_mutag_delta_heatmap_H4.png`, `…_H2.png` via
`python scripts/api_wanndb_query/aggregate_tu_depth_dh_h_mutag.py --H 4 2`.

---

## Analysis (as datasets finish)

Protocol: better LR mean±std; Δ = gated − ungated; paired *t* by seed.
One heatmap per `(dataset, H)` → `results/tu_sigma_depth_dh_h/analysis/`.

```bash
# after each dataset block completes:
python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py \
  --datasets mutag --H 64 8          # already done
python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py \
  --datasets mutag --H 4 2           # after 44899291
python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py \
  --datasets enzymes --H 64 8        # after 44897907
python scripts/api_wanndb_query/aggregate_tu_depth_dh_h.py \
  --datasets proteins collab imdb_binary reddit_binary --H 64 8
```

Outputs: `fig_<ds>_delta_heatmap_H{H}.png` + `<ds>_best_lr.csv`
(see [`results/tu_sigma_depth_dh_h/analysis/README.md`](results/tu_sigma_depth_dh_h/analysis/README.md)).

### MUTAG H∈{64,8} snapshot (2026-09-06)

Paired gated−ungated on best-LR-per-family seeds:

| Cell | Δ (pp) | p |
|------|-------:|--:|
| **L=1, d_h=1, H=8** | **+4.26** | **0.003** |
| L=1, d_h=16, H=8 | +4.26 | 0.047 |
| L=1, d_h=2, H=8 | +4.68 | 0.051 |
| almost all H=64 cells | ≤0 / n.s. | — |

Mean Δ by trunk: **H=8 ≈ +0.8 pp**, **H=64 ≈ −1.0 pp**. So the
“gating helps” signal is mainly **shallow + narrow `H`**, not low `d_h`
alone — useful for App. H capacity narrative; still not preference→γ routing.

Interesting slices:

1. Fix `H`, heatmap Δ vs `(L, d_h)` (main figure).
2. Compare `H=64` vs `H=8` vs `H=4` vs `H=2` side-by-side.
3. Cross-dataset: which `(L,d_h,H)` cells are consistently red / significant.

### Gate dump (MUTAG L=1 H=8 best gated LRs) → preference join

Do **not** dump on the login node (wrong Python / hung torch). Use GPU array:

```bash
# smoke
TU_LDHH_GDUMP_ARRAY=1 TU_LDHH_GDUMP_PARALLEL=1 \
  bash bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh

# full: 4 cells × 5 seeds = 20
bash bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh
```

| Field | Value |
|-------|-------|
| **JOBID** | 🛑 pending |
| **Submit** | `bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh` |
| **Worker** | `bash_interface/cluster/run_dump_tu_depth_dh_h_gates.sh` |
| **Cells** | dh1/lr01 · dh2/lr01 · dh4/lr001 · dh16/lr01 |
| **Out** | `$GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/mutag_L1_dh*_H8_gated_*/gate_values_per_graph.pt` |
| **Logs** | `logs_gnnplus/tu_LdhH_gdmp_<JOBID>_<TASK>.log` |

Then join locally with `scripts/heterogeneity/join_tu_gate_operator_preference.py`.

---

## Notes

- Paper **L=12** is not in this grid; add later if needed for exact Tab.17/18 redo.
- `H=8` with `d_h=16` is allowed (concat 96 → project to 8) but unusual.
- Overlaps prior `tu_dh{1,2}_*` / Tab.17/18 cells at specific `(L,d_h,H)` only if
  those match (prior extreme used **L=12**; this grid does not).
