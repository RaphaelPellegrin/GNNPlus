#!/usr/bin/env bash
# Launch SiGMA paper Table 6 ablations for 1-MP-head LRGB best models.
#
# 3 datasets × 5 variants × 5 seeds = 75 jobs.
# Max concurrent GPUs: PAPER_T6_1MP_PARALLEL (default 10).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table6_lrgb_1mp_hetero.sh
#
# COCO-only on gpu_h200 (keep mweber jobs; see submit_coco_h200_relaunch.sh):
#   PAPER_T6_1MP_ARRAY=51-75 PAPER_T6_1MP_PARALLEL=13 \
#     PAPER_T6_1MP_PARTITION=gpu_h200 PAPER_T6_1MP_NAME_SUFFIX=_h200 \
#     PAPER_T6_1MP_TIME=72:00:00 \
#     bash bash_interface/cluster/submit_paper_table6_lrgb_1mp_hetero.sh
#
# Then paste ARRAY JOBID into Paper_table6_lrgb_1mp.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T6_1MP_NUM_SEEDS:-5}"
NUM_VARIANTS="${PAPER_T6_1MP_NUM_VARIANTS:-5}"
NUM_DATASETS="${PAPER_T6_1MP_NUM_DATASETS:-3}"
NUM_TASKS="${PAPER_T6_1MP_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T6_1MP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T6_1MP_PARALLEL:-10}"
PARTITION="${PAPER_T6_1MP_PARTITION:-mweber_gpu}"
NICE="${PAPER_T6_1MP_NICE:-10000}"
MEM="${PAPER_T6_1MP_MEM:-128GB}"
TIME="${PAPER_T6_1MP_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T6_1MP_WANDB_PREFIX:-paper_T6}"
NAME_SUFFIX="${PAPER_T6_1MP_NAME_SUFFIX:-}"

sbatch_args=(
    --parsable
    --job-name=sigma_T6_1mp
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T6_1mp_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T6_1MP_NUM_SEEDS="${NUM_SEEDS}",PAPER_T6_1MP_NUM_VARIANTS="${NUM_VARIANTS}",PAPER_T6_1MP_NUM_DATASETS="${NUM_DATASETS}",PAPER_T6_1MP_NUM_TASKS="${NUM_TASKS}",PAPER_T6_1MP_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T6_1MP_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T6_1MP_MAX_EPOCH="${PAPER_T6_1MP_MAX_EPOCH:-}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table6_lrgb_1mp_hetero.sh
)"

cat <<EOF

=== SiGMA Table 6 (1-MP LRGB) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Name suffix:   ${NAME_SUFFIX:-<none>}
  Max epoch:     ${PAPER_T6_1MP_MAX_EPOCH:-<cfg default>}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/sigma_T6_1mp_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B group pattern:  ${WANDB_PREFIX}_<dataset>_<Variant>
  W&B tags:           paper_table6, <Variant>, <dataset>, seed<k>

  Variants:
    SiGMA              = best model as-is (ng=1, gated)
    Homog_MP           = +1 same-type MP head (ng=2, gated)
    Hetero_MP          = +1 different-type MP head (ng=2, gated)
    Homog_MP_ungated   = +1 same-type MP head, gate=none
    Hetero_MP_ungated  = +1 different-type MP head, gate=none

  Datasets / base → homog / hetero:
    peptides_func    GCN       → GCN,GCN       / GCN,GINE
    peptides_struct  GINE      → GINE,GINE     / GINE,GGNN
    coco             GATEDGCN  → GATEDGCN×2    / GATEDGCN,GCN

  Anchors / source runs:
    peptides_func    l31u4b3k   peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml
    peptides_struct  bqkect9l   peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml
    coco             xgjakrz0   coco-hybrid-5b4z9l3u-a1g1-anchor.yaml

  Paste JOBID into Paper_table6_lrgb_1mp.md

  Aggregate when done:
    for ds in peptides_func peptides_struct coco; do
      for v in SiGMA Homog_MP Hetero_MP Homog_MP_ungated Hetero_MP_ungated; do
        python scripts/api_wanndb_query/aggregate_paper_repro.py \\
          --group ${WANDB_PREFIX}_\${ds}_\${v} --metric best_test_perf --state finished
      done
    done

EOF
