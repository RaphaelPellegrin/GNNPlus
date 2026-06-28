# Cluster sweep-agent job map

Parsed from `logs_gnnplus/sweep_agent_*.log` on holylogin05 (2026-06-07).

Use this file to map **SLURM array job IDs** → **W&B sweep** → **dataset**.  
W&B entity/project: `weber-geoml-harvard-university/GNNPlus`.

## Refresh on cluster

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

for f in $(ls logs_gnnplus/sweep_agent_*.log 2>/dev/null); do
  jid=$(basename "$f" | sed -E 's/sweep_agent_([0-9]+)_([0-9]+)\.log/\1_\2/')
  sid=$(grep -m1 'SWEEP_ID=' "$f" | sed 's/.*\///;s/ runs_per_agent.*//')
  ds=$(grep -m1 "Loaded dataset '" "$f" | sed "s/.*Loaded dataset '//;s/'.*//")
  echo "$jid  sweep=$sid  dataset=${ds:-pending}"
done | sort -u
```

Copy output here or run the grouping snippet at the bottom of this file.

## Dataset name aliases

| Log name | Likely dataset |
|----------|----------------|
| `edge_wt_region_boundary` | Pascal VOC-SP or COCO-SP (PyG superpixel node labels) |
| `subset` | MalNet-Tiny (subset loader) |
| `LocalDegreeProfile` | MalNet-Tiny (LDP encoding) |
| `pending` | Agent not yet past dataset load (or log truncated) |

## Sweep index (W&B links)

| Sweep ID | Dataset(s) | W&B |
|----------|------------|-----|
| `1xc54ggv` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/1xc54ggv |
| `nc9ijn4m` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/nc9ijn4m |
| `mhc71f9c` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/mhc71f9c |
| `nkwgduxb` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/nkwgduxb |
| `xxgqtksk` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/xxgqtksk |
| `4tzj5ty2` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/4tzj5ty2 |
| `j6ivhwyj` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/j6ivhwyj |
| `6pzm9hv3` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/6pzm9hv3 |
| `r4vnk32t` | MNIST | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/r4vnk32t |
| `0yksmizq` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/0yksmizq |
| `yt923k6q` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/yt923k6q |
| `zmn3pwl0` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/zmn3pwl0 |
| `42ln2l5u` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/42ln2l5u |
| `x0h9ao6z` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/x0h9ao6z |
| `q5l4oy7f` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/q5l4oy7f |
| `5q8upl19` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/5q8upl19 |
| `o7tsb3k1` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/o7tsb3k1 |
| `19501lc2` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/19501lc2 |
| `f2pecp63` | CIFAR10 | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/f2pecp63 |
| `h5hnvspp` | ENZYMES | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/h5hnvspp |
| `cj7qt89k` | MUTAG | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/cj7qt89k |
| `74r0j7lh` | ogbg-molhiv | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/74r0j7lh |
| `yvt7ag8d` | ogbg-molpcba | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/yvt7ag8d |
| `ivcmi91x` | CLUSTER | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/ivcmi91x |
| `jrnsnn39` | CLUSTER | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/jrnsnn39 |
| `i2g7uadv` | CLUSTER | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/i2g7uadv |
| `tfurpu5k` | PATTERN | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/tfurpu5k |
| `ve1sldtl` | PATTERN | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/ve1sldtl |
| `c3ft7rym` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/c3ft7rym |
| `wzob50ux` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/wzob50ux |
| `hrfmtir9` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/hrfmtir9 |
| `m202dgkm` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/m202dgkm |
| `3y4mke63` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/3y4mke63 |
| `hymdtc7m` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/hymdtc7m |
| `2as0vt6i` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/2as0vt6i |
| `nztrcvfk` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/nztrcvfk |
| `8abis8pw` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/8abis8pw |
| `xgzha5uu` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/xgzha5uu |
| `u3a23jsp` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/u3a23jsp |
| `mclluug7` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/mclluug7 |
| `j4ul76cp` | peptides-functional | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/j4ul76cp |
| `s4f4os9u` | peptides-structural | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/s4f4os9u |
| `qvk5l4uk` | peptides-structural | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/qvk5l4uk |
| `y86v2hoj` | VOC/COCO (`edge_wt_region_boundary`) | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/y86v2hoj |
| `plo2ianx` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/plo2ianx |
| `cshiosue` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/cshiosue |
| `jjtcafgz` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/jjtcafgz |
| `xggzoblj` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/xggzoblj |
| `i030zruf` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/i030zruf |
| `hch2kkia` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/hch2kkia |
| `kb2ye07d` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/kb2ye07d |
| `gskccsb2` | VOC/COCO | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/gskccsb2 |
| `cg69pzxb` | MalNet (`subset`) | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/cg69pzxb |
| `kpdtinm5` | MalNet (`subset`) | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/kpdtinm5 |
| `gveuibxi` | MalNet (`subset`) | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/gveuibxi |
| `d78ypkgd` | MalNet (`subset`) | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/d78ypkgd |
| `84rlksar` | MalNet (`LocalDegreeProfile`) | https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/84rlksar |

## Array jobs (one row per SLURM job ID)

| Job ID | Sweep | Dataset | Tasks | Notes |
|--------|-------|---------|-------|-------|
| 23838789 | `1xc54ggv` | MNIST | 1–16 | |
| 23838800 | `nc9ijn4m` | CIFAR10 | 1–16 | |
| 23838846 | `y86v2hoj` | VOC/COCO | 1–16 | |
| 23838857 | `plo2ianx` | VOC/COCO | 1–16 | |
| 23839008 | `h5hnvspp` | ENZYMES | 1–16 | |
| 23839019 | `74r0j7lh` | ogbg-molhiv | 1–16 | |
| 23839031 | `cg69pzxb` | MalNet | 1–16 | tasks 1–7 pending at snapshot |
| 23839036 | `cj7qt89k` | MUTAG | 1–16 | |
| 23839038 | `mihl3nfg` | ? | 1–16 | all pending |
| 23839041 | `mxfva26n` | ? | 1–16 | all pending |
| 23839054 | `yvt7ag8d` | ogbg-molpcba | 1–16 | tasks 10–16 pending |
| 23839059 | `ivcmi91x` | CLUSTER | 1–16 | |
| 23839063 | `tfurpu5k` | PATTERN | 1–16 | tasks 14–16 pending |
| 23878867 | — | ? | 1–16 | no sweep in log |
| 23878871 | — | ? | 1–16 | no sweep in log |
| 23879809 | `mhc71f9c` | MNIST | 1–16 | |
| 23879839 | `0yksmizq` | CIFAR10 | 1–16 | |
| 23881229 | `cshiosue` | VOC/COCO | 1–16 | |
| 23881231 | `jjtcafgz` | VOC/COCO | 1–16 | |
| 23881240 | `c3ft7rym` | peptides-functional | 1–16 | |
| 23881249 | `s4f4os9u` | peptides-structural | 1–16 | |
| 23884506 | `xggzoblj` | VOC/COCO | 1–16 | |
| 23884517 | `i030zruf` | VOC/COCO | 1–16 | |
| 23884588 | `kpdtinm5` | MalNet | 1–16 | |
| 23885467 | `xxgqtksk` | MNIST | 1–16 | tasks 7–16 pending |
| 23885469 | `xbrzyjf9` | ? | 1–16 | all pending |
| 24028000 | `mhc71f9c` | MNIST | 1–8 | |
| 24028710 | `0yksmizq` | CIFAR10 | 1–7 | |
| 24028740 | `zmn3pwl0` | CIFAR10 | 1–8 | tasks 5–8 pending |
| 24028758 | `wzob50ux` | peptides-functional | 1–8 | tasks 5–8 pending |
| 24028888 | `4tzj5ty2` | MNIST | 1–8 | tasks 5–8 pending |
| 24028897 | `zmn3pwl0` | CIFAR10 | 1–8 | all pending |
| 24028909 | `wzob50ux` | peptides-functional | 1–8 | all pending |
| 24028998 | `nkwgduxb` | MNIST | 1–16 | |
| 24028999 | `yt923k6q` | CIFAR10 | 1–16 | |
| 24029000 | `hrfmtir9` | peptides-functional | 1–16 | |
| 24029044 | `nkwgduxb` | MNIST | 1–8 | |
| 24030551 | `hrfmtir9` | peptides-functional | 1–8 | |
| 24030686 | `yt923k6q` | CIFAR10 | 1–13 | |
| 24030825 | `nkwgduxb` | MNIST | 1–24 | |
| 24035920 | `m202dgkm` | peptides-functional | 1–16 | |
| 24035935 | `42ln2l5u` | CIFAR10 | 1–10 | |
| 24049784 | `nkwgduxb` | MNIST | 1–24 | |
| 24066076 | `x0h9ao6z` | CIFAR10 | 1–16 | tasks 12–15 pending |
| 24067932 | `3y4mke63` | peptides-functional | 1–16 | task 13 pending |
| 24073577 | `x0h9ao6z` | CIFAR10 | 1–16 | tasks 9,12–14 pending |
| 24073631 | `3y4mke63` | peptides-functional | 1–16 | tasks 11,16 pending |
| 24080312 | `nkwgduxb` | MNIST | 1–24 | |
| 24082752 | `5q8upl19` | CIFAR10 | 1 | |
| 24082763 | `hymdtc7m` | peptides-functional | 1 | |
| 24151608 | `3y4mke63` | peptides-functional | 1–16 | tasks 1–3,5,7 pending |
| 24151678 | `2as0vt6i` | peptides-functional | 1 | |
| 24153748 | `o7tsb3k1` | CIFAR10 | 1 | |
| 24218080 | `j6ivhwyj` | MNIST | 1 | |
| 24218140 | `q5l4oy7f` | CIFAR10 | 1–16 | tasks 9,12,15–16 pending |
| 24218141 | `q5l4oy7f` | CIFAR10 | 1–16 | several pending |
| 24221445 | `6pzm9hv3` | MNIST | 1 | |
| 24224077 | `nztrcvfk` | peptides-functional | 1 | |
| 24224125 | `8abis8pw` | peptides-functional | 1 | |
| 24224136 | `xgzha5uu` | peptides-functional | 1–8 | |
| 24238774 | `19501lc2` | CIFAR10 | 1–8 | tasks 7–8 pending |
| 24238819 | `r4vnk32t` | MNIST | 1 | |
| 24238846 | `u3a23jsp` | peptides-functional | 1–8 | |
| 24340200 | `q5l4oy7f` | CIFAR10 | 1–8 | tasks 1,6–8 pending |
| 24489931 | `kb2ye07d` | VOC/COCO | 1–4 | |
| 24489970 | `gskccsb2` | VOC/COCO | 1–8 | |
| 24489974 | `d78ypkgd` | MalNet | 1 | |
| 24491747 | `q5l4oy7f` | CIFAR10 | 1–16 | several pending |
| 24493767 | `mclluug7` | peptides-functional | 1 | |
| 24493976 | `j4ul76cp` | peptides-functional | 1 | |
| 24493990 | `f2pecp63` | CIFAR10 | 1–9 | tasks 10–16: sweep empty in log |
| 24494015 | `mclluug7` | peptides-functional | 1 | |
| 24495423 | `gveuibxi` | MalNet | 1–16 | several pending |
| 24670840 | — | ? | 1–8 | no sweep |
| 24670854 | — | ? | 1–8 | no sweep |
| 24670880 | — | ? | 1–8 | no sweep |
| 24670894 | — | ? | 1–8 | no sweep |
| 24679066 | — | ? | 1–16 | no sweep |
| 24679677 | — | ? | 1–16 | no sweep |
| 24680579 | `gveuibxi` | MalNet | 1–16 | tasks 1–4,6–7 pending |
| 24680581 | `d78ypkgd` | MalNet | 1–4 | all pending |
| 24680583 | `ve1sldtl` | PATTERN | 1–8 | task 8 pending |
| 24680584 | `jrnsnn39` | CLUSTER | 1–8 | |
| 24680587 | `84rlksar` | MalNet | 1–8 | all pending at snapshot |
| 24680588 | `qvk5l4uk` | peptides-structural | 1–16 | several pending |
| 24834215 | `jrnsnn39` | CLUSTER | 1–16 | tasks 9,13,16 pending |
| 24835334 | `f2pecp63` | CIFAR10 | 1–4 | |
| 24840878 | `ve1sldtl` | PATTERN | 1–16 | several pending |
| 24841184 | `gveuibxi` | MalNet | 1–16 | tasks 1–9,16 pending |
| 24841185 | `qvk5l4uk` | peptides-structural | 1–16 | mostly pending |
| 24848931 | `hch2kkia` | VOC/COCO | 1–14 | |
| 24863281 | `i2g7uadv` | CLUSTER | 1–16 | |
| 24934955 | `i2g7uadv` | CLUSTER | 1–8 | |
| 24942145 | `hch2kkia` | VOC/COCO | 1–3 | |
| 25025513 | `ve1sldtl` | PATTERN | 1–8 | tasks 1–6 pending |
| 25032685 | `84rlksar` | MalNet | 1–8 | |
| 25032963 | `kb2ye07d` | VOC/COCO | 1–2 | |
| 25086299 | `84rlksar` | MalNet | 1–16 | |

## By dataset (active / recent sweeps)

### MNIST
`23838789`, `23879809`, `23885467`, `24028000`, `24028888`, `24028998`, `24029044`, `24030825`, `24049784`, `24080312`, `24218080`, `24221445`, `24238819`

### CIFAR10
`23838800`, `23879839`, `24028710`, `24028740`, `24028897`, `24028999`, `24030686`, `24035935`, `24066076`, `24073577`, `24082752`, `24153748`, `24218140`, `24218141`, `24238774`, `24340200`, `24491747`, `24493990`, `24835334`

### PATTERN
`23839063`, `24680583`, `24840878`, `25025513`

### CLUSTER
`23839059`, `24680584`, `24834215`, `24863281`, `24934955`

### MalNet
`23839031`, `23884588`, `24489974`, `24495423`, `24680579`, `24680581`, `24680587`, `24841184`, `25032685`, `25086299`

### VOC / COCO
`23838846`, `23838857`, `23881229`, `23881231`, `23884506`, `23884517`, `24489931`, `24489970`, `24848931`, `24942145`, `25032963`

### peptides-functional
`23881240`, `24028758`, `24028909`, `24029000`, `24030551`, `24035920`, `24067932`, `24073631`, `24082763`, `24151608`, `24151678`, `24224077`, `24224125`, `24224136`, `24238846`, `24493767`, `24493976`, `24494015`

### peptides-structural
`23881249`, `24680588`, `24841185`

### Other graph benchmarks
| Dataset | Job IDs |
|---------|---------|
| ENZYMES | `23839008` |
| MUTAG | `23839036` |
| ogbg-molhiv | `23839019` |
| ogbg-molpcba | `23839054` |

### Unknown / no sweep in log
`23839038`, `23839041`, `23878867`, `23878871`, `23885469`, `24670840`, `24670854`, `24670880`, `24670894`, `24679066`, `24679677`

## GRIT runs (direct `sbatch`, not W&B sweeps)

Submitted via `bash bash_interface/cluster/submit_grit.sh`.  
W&B project: https://wandb.ai/weber-geoml-harvard-university/GNNPlus

| Job ID | Variant | Dataset | Config | W&B run name | Log | Metric | Notes |
|--------|---------|---------|--------|--------------|-----|--------|-------|
| `25287393` | hybrid | PATTERN | `configs/gated_hybrid/pattern-grit-repro-a1g1.yaml` | `pattern_grit_hybrid_seed0_job25287393` | `logs_gnnplus/pattern_grit_hybrid_25287393.log` | `test/accuracy-SBM` | [laneuetq](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/laneuetq) |
| `25309402` | **standalone** | PATTERN | `configs/grit/pattern-grit-rrwp.yaml` | `pattern_grit_rrwp_seed0_job25309402` | `logs_gnnplus/pattern_grit_rrwp_25309402.log` | `test/accuracy-SBM` | Relaunch after RRWP PE-stats fix (`614213b`) |
| `25302952` | standalone | PATTERN | `configs/grit/pattern-grit-rrwp.yaml` | — | `logs_gnnplus/pattern_grit_rrwp_25302952.log` | `test/accuracy-SBM` | **FAILED** `Unexpected PE stats selection RRWP` |
| `25296336` | standalone | PATTERN | `configs/grit/pattern-grit-rrwp.yaml` | — | `logs_gnnplus/pattern_grit_rrwp_25296336.log` | `test/accuracy-SBM` | **FAILED** `KeyError: gnn.dim_edge` |
| `25296338` | standalone | CLUSTER | `configs/grit/cluster-grit-rrwp.yaml` | — | `logs_gnnplus/cluster_grit_rrwp_25296338.log` | `test/accuracy-SBM` | **FAILED** `KeyError: gnn.dim_edge` |
| `25296339` | standalone | ZINC | `configs/grit/zinc-grit-rrwp.yaml` | — | `logs_gnnplus/zinc_grit_rrwp_25296339.log` | `test/mae` | **FAILED** `KeyError: gnn.dim_edge` |

**Variant cheat sheet**

| Variant | `model.type` | Pos. encoding | Submit |
|---------|--------------|---------------|--------|
| standalone (paper GRIT) | `GritTransformer` | RRWP | `submit_grit.sh <ds> standalone` |
| hybrid (1 attn + 1 GRIT MP) | `hybrid_gnn` | RWSE | `submit_grit.sh <ds> hybrid` |

**Monitor GRIT jobs**

```bash
squeue -u $USER | grep grit
tail -f logs_gnnplus/pattern_grit_rrwp_25309402.log
grep -E "Precomputing RRWP|RRWP done|wandb|epoch|accuracy-SBM|Traceback" logs_gnnplus/pattern_grit_rrwp_25309402.log | tail -20
```

**W&B lookup:** tag `job_<JOBID>` or search run name `*_grit_*_job<JOBID>`.

### Troubleshooting standalone GRIT (no W&B run)

| Symptom | Cause |
|---------|--------|
| Only [laneuetq](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/laneuetq) visible | That run is **hybrid** GRIT (`model.type=hybrid_gnn`), not standalone |
| Standalone jobs `25296336`/`38`/`39` missing on W&B | Failed at **config load** (`KeyError: gnn.dim_edge`) before training; fixed in `grit_config.py` |
| `wandb.init` after loader | RRWP crash = no run (fixed: early init in `main.py`) |
| Log shows `Unexpected PE stats selection RRWP` | RRWP wrongly routed to `compute_posenc_stats`; fixed in `master_loader.py` |
| Log stops at `Precomputing RRWP` | `torch_sparse` missing or OOM during dense RRWP (`adj.to_dense()`) |
| Log never reaches `RRWP done` | PATTERN/CLUSTER precompute is slow (many graphs); ZINC can take hours |

**Diagnose on cluster:**

```bash
for j in 25309402 25287393 25302952 25296336 25296338 25296339; do
  echo "=== JOB $j ==="
  sacct -j $j -X --format=JobID,State,ExitCode,Elapsed,MaxRSS 2>/dev/null
  f=$(ls logs_gnnplus/*_${j}.log 2>/dev/null | head -1)
  [ -n "$f" ] && grep -E "RRWP|Traceback|Error|wandb|GritTransformer|hybrid_gnn|epoch" "$f" | tail -8
done
```

**W&B filters:** standalone → `config.model.type = GritTransformer`; hybrid → `config.model.type = hybrid_gnn` + tag `grit`.

**Relaunch standalone** (after `git pull` ≥ `614213b`):

```bash
bash bash_interface/cluster/submit_grit.sh pattern standalone   # → 25309402
bash bash_interface/cluster/submit_grit.sh cluster standalone
bash bash_interface/cluster/submit_grit.sh zinc standalone
```

## PATTERN hybrid anchored on qcz7umtl (GCNE baseline)

Best MP-only baseline: [qcz7umtl](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/qcz7umtl)  
(`configs/gcn/pattern.yaml`, seed 2, `layer_type: gcne`, SBM ≈ 0.866)

| Variant | Config | Architecture |
|---------|--------|----------------|
| baseline (done) | `configs/gcn/pattern.yaml` | 12×GCNE MP-only |
| hybrid a1g1 | `configs/gated_hybrid/pattern-hybrid-qcz7umtl-a1g1.yaml` | 1×attn + 1×GCNE MP |
| hybrid a2g1 | `configs/gated_hybrid/pattern-hybrid-qcz7umtl-a2g1.yaml` | 2×attn + 1×GCNE MP |

Submit: `bash bash_interface/cluster/submit_pattern_hybrid_qcz7umtl.sh a1g1`  
Fair grid (LR sweep): `PATTERN_FAIR_TASKS=2-5 bash submit_pattern_gcne_fair_comparison.sh`

| Job ID | Task | Variant | Config | W&B run name | Log |
|--------|------|---------|--------|--------------|-----|
| `25297819` | — | a1g1 anchor | `pattern-hybrid-qcz7umtl-a1g1.yaml` | `pattern_hybrid_qcz7umtl_a1g1_seed2_job25297819` | `logs_gnnplus/pattern_hybrid_qcz7umtl_a1g1_25297819.log` |
| `25297831` | 2 | a1g1 lr=0.001 | `pattern-gcne-repro-a1.yaml` | `pattern_gcne_seed2_repro_hybrid_a1g1_lr0p001` | `logs_gnnplus/pattern_fair_25297831_2.log` |
| `25297831` | 3 | a2g1 lr=0.001 | `pattern-gcne-repro-a2.yaml` | `pattern_gcne_seed2_repro_hybrid_a2g1_lr0p001` | `logs_gnnplus/pattern_fair_25297831_3.log` |
| `25297831` | 4 | a1g1 lr=0.002 | `pattern-gcne-repro-a1.yaml` | `pattern_gcne_seed2_repro_hybrid_a1g1_lr0p002` | `logs_gnnplus/pattern_fair_25297831_4.log` |
| `25297831` | 5 | a2g1 lr=0.002 | `pattern-gcne-repro-a2.yaml` | `pattern_gcne_seed2_repro_hybrid_a2g1_lr0p002` | `logs_gnnplus/pattern_fair_25297831_5.log` |

```bash
squeue -u $USER | grep -E 'pattern_hybrid|pattern_fair'
tail -f logs_gnnplus/pattern_hybrid_qcz7umtl_a1g1_25297819.log
tail -f logs_gnnplus/pattern_fair_25297831_2.log
```

## Paper repro (5 seeds)

See **`bash_interface/cluster/PAPER_REPRO_TRACKER.md`** for `bestmodel_v1` cohorts, W&B groups, run names, and aggregate results.

| Dataset | Anchor | SLURM array | W&B group |
|---------|--------|-------------|-----------|
| CIFAR10 | [ulij45a2](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ulij45a2) | `25310487` | `paper_bestmodel_v1_cifar10_ulij45a2` |
| MNIST | [lcvbyyss](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/lcvbyyss) | `25313521` | `paper_bestmodel_v1_mnist_lcvbyyss` |
| MalNet-Tiny | [9h3jqzkm](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/9h3jqzkm) (v1, a0g2) | `25313522` | `paper_bestmodel_v1_malnet_9h3jqzkm` |
| MalNet-Tiny | [4j21kp8d](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/4j21kp8d) (v2, a1g2) | `25324182` | `paper_bestmodel_v2_malnet_4j21kp8d` |
| VOC-SP | [j7ukyzdm](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/j7ukyzdm) | `25322496` | `paper_bestmodel_v1_voc_j7ukyzdm` |
| COCO-SP | [o5hr3tma](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o5hr3tma) | *(pending)* | `paper_bestmodel_v1_coco_o5hr3tma` |

Quick start (CIFAR10 ulij45a2):

```bash
bash bash_interface/cluster/submit_cifar10_hybrid_ulij45a2_paper_repro.sh
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_cifar10_ulij45a2
```

MNIST lcvbyyss:

```bash
bash bash_interface/cluster/submit_mnist_hybrid_lcvbyyss_paper_repro.sh
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_mnist_lcvbyyss
```

MalNet-Tiny 9h3jqzkm (v1, a0g2):

```bash
bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_paper_repro.sh
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_malnet_9h3jqzkm
```

MalNet-Tiny 4j21kp8d (v2, a1g2):

```bash
bash bash_interface/cluster/submit_malnet_hybrid_4j21kp8d_paper_repro.sh
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v2_malnet_4j21kp8d
```

VOC-SP j7ukyzdm:

```bash
bash bash_interface/cluster/submit_voc_hybrid_j7ukyzdm_paper_repro.sh
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_voc_j7ukyzdm
```

COCO-SP o5hr3tma:

```bash
bash bash_interface/cluster/submit_coco_hybrid_o5hr3tma_paper_repro.sh
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_bestmodel_v1_coco_o5hr3tma
```

## COCO hybrid anchored on 5b4z9l3u (GatedGCN+ baseline)

Best MP-only baseline: [5b4z9l3u](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5b4z9l3u)  
(`configs/gatedgcn/coco.yaml`, seed 1, `layer_type: gatedgcn`, metric `test/f1`)

| Variant | Config | Architecture |
|---------|--------|----------------|
| baseline (done) | `configs/gatedgcn/coco.yaml` | 20×GatedGCN+ MP-only |
| hybrid a1g1 | `configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1.yaml` | 1×attn + 1×GATEDGCN MP |
| hybrid a2g1 | `configs/gated_hybrid/coco-hybrid-5b4z9l3u-a2g1.yaml` | 2×attn + 1×GATEDGCN MP |
| sweep base | `configs/gated_hybrid/coco-gatedgcn-best-hybrid.yaml` | Bayes over attn/LR/d_h |

**Submit options**

```bash
# Single anchor runs (128GB, 192h)
bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u.sh both

# Fair grid: a1g1/a2g1 × lr {0.001, 0.002} (skip baseline if done)
COCO_FAIR_TASKS=2-5 bash bash_interface/cluster/submit_coco_gatedgcn_fair_comparison.sh

# Bayes sweep: attn {1,2}, LR log-uniform, d_h, batch {8,16}
bash bash_interface/cluster/submit_coco_gatedgcn_best_hybrid_sweep.sh
```

| Job ID | Task | Variant | Sweep / W&B | Log |
|--------|------|---------|-------------|-----|
| `25299176` | — | a1g1 anchor | `coco_hybrid_5b4z9l3u_a1g1_seed1_job25299176` | `logs_gnnplus/coco_hybrid_5b4z9l3u_a1g1_25299176.log` |
| `25299177` | — | a2g1 anchor | `coco_hybrid_5b4z9l3u_a2g1_seed1_job25299177` | `logs_gnnplus/coco_hybrid_5b4z9l3u_a2g1_25299177.log` |
| `25299326` | 1–16 | Bayes sweep | [xrks1f52](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/sweeps/xrks1f52) | `logs_gnnplus/sweep_agent_25299326_<TASK>.log` |

Sweep `xrks1f52` (`GNNplus_best_hybrid-coco-gatedgcn`): attn {1,2}, LR log-uniform, d_h {32,48,52}, batch {8,16}, 16 agents × 4 runs, 128GB, seed 1.

```bash
squeue -u $USER | grep -E 'coco_hybrid|25299326'
tail -f logs_gnnplus/coco_hybrid_5b4z9l3u_a1g1_25299176.log
grep -m1 SWEEP_ID logs_gnnplus/sweep_agent_25299326_1.log
```

## Quick lookup

```bash
# Find dataset for a job ID
JOB=25086299
grep "^${JOB}_" bash_interface/cluster/CLUSTER_JOB_MAP.md   # after refresh
grep -h "SWEEP_ID=\|Loaded dataset" logs_gnnplus/sweep_agent_${JOB}_*.log | head -4

# List running sweep agents
squeue -u $USER -n sweep_agent -o "%.10i %.12j %.8T %.10M"

# W&B: filter by SLURM tag (auto-added since custom_train.py update)
#   tag: job_25086299
```

## Changelog

| Date | Notes |
|------|-------|
| 2026-06-07 | Initial map from `sweep_agent_*.log` grep on holylogin05 |
| 2026-06-07 | GRIT batch: standalone `25296336`/`25296338`/`25296339` (PATTERN/CLUSTER/ZINC); hybrid `25287393` (PATTERN) |
| 2026-06-07 | PATTERN hybrid qcz7umtl anchor configs + `submit_pattern_hybrid_qcz7umtl.sh` |
| 2026-06-07 | Submitted `25297819` (a1g1 anchor) + array `25297831` tasks 2–5 (fair hybrid LR sweep) |
| 2026-06-07 | COCO hybrid 5b4z9l3u anchor configs, fair comparison array, Bayes sweep |
| 2026-06-07 | Submitted COCO `25299176`/`25299177` (a1g1/a2g1) + sweep array `25299326` (`xrks1f52`) |
| 2026-06-07 | Early W&B init before RRWP loader; GRIT troubleshooting notes in job map |
| 2026-06-07 | Fix standalone GRIT `KeyError: gnn.dim_edge` — register key in `grit_config.py` |
| 2026-06-07 | Relaunch standalone PATTERN GRIT `25302952` after `acebf66` pull |
| 2026-06-07 | Fix RRWP routed to `compute_posenc_stats` — skip RRWP in generic PE loop |
| 2026-06-07 | RRWP dense fallback (no torch_sparse); encoder uses PyG `coalesce` |
| 2026-06-07 | Relaunch standalone PATTERN GRIT `25309402` after `614213b` pull |
| 2026-06-07 | CIFAR10 paper repro ulij45a2: anchor yaml, 5-seed array, W&B group + aggregate script |
| 2026-06-07 | Submitted CIFAR10 paper repro array `25310487` (seeds 0–4, group `paper_bestmodel_v1_cifar10_ulij45a2`) |
| 2026-06-07 | Added `PAPER_REPRO_TRACKER.md` for bestmodel_v1 cohorts |
