# Main-text progressive d_h — SiGMA gated vs ungated

**Goal:** On Tab. 3/4 paper recipes, shrink **`d_h`** toward the Appendix H
regime and show that **gating helps more as capacity gets tighter**.

Complements:
- TU extreme `d_h∈{1,2}` — [`Paper_tu_sigma_dh_extreme.md`](Paper_tu_sigma_dh_extreme.md)
- Prior gated-only shrink — [`Paper_sigma_dh_matched.md`](Paper_sigma_dh_matched.md) / [`rebuttal.md`](rebuttal.md)
- Full-width Table 5 ungated (often n.s.)

**Hypothesis:** Δ(gated − ungated) **increases** as `d_h` decreases
(PATTERN / CLUSTER / MNIST / Pep-func).

---

## Design

| Family | Paper recipe (kept) | `d_h` ladder |
|--------|---------------------|--------------|
| PATTERN | a2g2 GCNE×2 GRIT VN4 H90 | **16, 8, 4, 2, 1** |
| CLUSTER | a1g1 GATEDGCN H56 L16 | **24, 12, 8, 4, 1** |
| MNIST | a2g2 GATEDGCN×2 H60 | **37, 16, 8, 4, 1** |
| Pep-func | a1g2 GCN×2 H275 | **23, 12, 8, 4, 1** |

Per cell: **gated** (yaml `gate`) vs **ungated** (`gnn.hybrid.gate none`),
LR ∈ `{1e-3, 1e-2}` × 5 seeds; report better LR.

Base configs (override `gnn.hybrid.d_h` at launch):

- `configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml`
- `configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml`
- `configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml`
- `configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml`

**Jobs:** 20 (fam×dh) × 2 gate × 2 LR × 5 seeds = **400**

---

## Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# smoke: PATTERN d_h=1 gated + ungated @ lr001 seed0
SIGMA_DH_PROG_ARRAY=81,91 SIGMA_DH_PROG_PARALLEL=2 \
  bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh

# recommended: one family at a time
SIGMA_DH_PROG_ARRAY=1-100   bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh  # PATTERN
# SIGMA_DH_PROG_ARRAY=101-200 bash ...  # CLUSTER
# SIGMA_DH_PROG_ARRAY=201-300 bash ...  # MNIST
# SIGMA_DH_PROG_ARRAY=301-400 bash ...  # Pep-func

# or full 400
# bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *paste JOBID* |
| **Submit** | `bash_interface/cluster/submit_sigma_dh_prog_ungated.sh` |
| **Worker** | `bash_interface/cluster/run_sigma_dh_prog_ungated.sh` |
| **Tasks** | `1-400%20` |
| **W&B** | `paper_sigma_dh_prog_<fam>_dh<k>_{gated,ungated}_{lr001,lr01}` |
| **Out** | `$GNNPLUS_OUT_DIR/sigma_dh_prog/` |
| **Logs** | `logs_gnnplus/sigma_dh_prog_<JOBID>_<TASK>.log` |

### Task map

| Tasks | Family | `d_h` order |
|------:|--------|-------------|
| 1–100 | PATTERN | 16 → 8 → 4 → 2 → 1 |
| 101–200 | CLUSTER | 24 → 12 → 8 → 4 → 1 |
| 201–300 | MNIST | 37 → 16 → 8 → 4 → 1 |
| 301–400 | Pep-func | 23 → 12 → 8 → 4 → 1 |

Per (fam, `d_h`) block of 20: gated lr001 · gated lr01 · ungated lr001 · ungated lr01 (5 seeds each).

Smoke: task **81** = PATTERN dh1 gated lr001 seed0 · **91** = PATTERN dh1 ungated lr001 seed0.

---

## Overlap with prior gated `dh_matched`

These gated cells already exist under `paper_sigma_dh_matched_*` (job `41709078` etc.):

| Family | Existing gated `d_h` |
|--------|---------------------|
| PATTERN | 16, 4 |
| CLUSTER | 36, 24 (36 not in this ladder) |
| MNIST | 37 |
| Pep-func | 23, 75 |

This campaign uses a **new W&B prefix** (`paper_sigma_dh_prog_*`) so groups stay clean.
If GPUs are scarce, skip re-running those gated tasks and compare prog-ungated to old gated means.

---

## Results (fill when finished)

Best-LR mean±std; Δ = gated − ungated (acc/AP pp; positive ⇒ gated better).

### PATTERN (Acc %)

| `d_h` | Gated | Ungated | Δ |
|------:|------:|--------:|--:|
| 16 | | | |
| 8 | | | |
| 4 | | | |
| 2 | | | |
| 1 | | | |

### CLUSTER (Acc %)

| `d_h` | Gated | Ungated | Δ |
|------:|------:|--------:|--:|
| 24 | | | |
| 12 | | | |
| 8 | | | |
| 4 | | | |
| 1 | | | |

### MNIST (Acc %)

| `d_h` | Gated | Ungated | Δ |
|------:|------:|--------:|--:|
| 37 | | | |
| 16 | | | |
| 8 | | | |
| 4 | | | |
| 1 | | | |

### Pep-func (AP)

| `d_h` | Gated | Ungated | Δ |
|------:|------:|--------:|--:|
| 23 | | | |
| 12 | | | |
| 8 | | | |
| 4 | | | |
| 1 | | | |

**Success criterion for rebuttal:** Δ grows (or becomes significant) at small `d_h`,
matching Appendix H / TU extreme-`d_h` narrative.

---

## Aggregate sketch

```bash
python - <<'PY'
import math, wandb
api = wandb.Api()
E,P = 'weber-geoml-harvard-university','GNNPlus'
ladders = {
  'pattern': [16,8,4,2,1],
  'cluster': [24,12,8,4,1],
  'mnist': [37,16,8,4,1],
  'pepfunc': [23,12,8,4,1],
}
for fam, dhs in ladders.items():
  print(f'\n=== {fam} ===')
  for dh in dhs:
    best = {}
    for gate in ('gated','ungated'):
      cands=[]
      for lr in ('lr001','lr01'):
        g=f'paper_sigma_dh_prog_{fam}_dh{dh}_{gate}_{lr}'
        xs=[]
        for r in api.runs(f'{E}/{P}', filters={'group': g}):
          if r.state!='finished': continue
          p=r.summary.get('best_test_perf')
          if p is None: continue
          p=float(p)
          # Pep-func AP stays in [0,1]; others often logged as fraction too
          xs.append(p*100 if p<=1.5 and fam!='pepfunc' else (p if fam=='pepfunc' else (p*100 if p<=1.5 else p)))
        if len(xs)==5:
          m=sum(xs)/5; s=math.sqrt(sum((x-m)**2 for x in xs)/4)
          cands.append((m,s,lr))
      if cands: best[gate]=max(cands, key=lambda t: t[0])
    if len(best)==2:
      gm,gs,glr=best['gated']; um,us,ulr=best['ungated']
      print(f'dh={dh:2d}  gated={gm:.4f}±{gs:.4f}({glr})  ung={um:.4f}±{us:.4f}({ulr})  Δ={gm-um:+.4f}')
PY
```
