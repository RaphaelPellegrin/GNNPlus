#!/usr/bin/env bash
# Launch SiGMA paper Table 5 ablations on mweber_gpu.
#
# 4 LRGB datasets × {SiGMA, SiGMA_ungated, Attn_only, MP_only} × 5 seeds = 80 jobs.
# Max concurrent GPUs: PAPER_T5_PARALLEL (default 18).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   # recommended: avoid holylabs quota on results/*/stats.json
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_ablations.sh
#
# COCO gap relaunch (MP_only seeds 2–4 + SiGMA_ungated seed1):
#   PAPER_T5_ARRAY=67,78-80%3 PAPER_T5_PARALLEL=3 PAPER_T5_TIME=192:00:00 \
#     bash bash_interface/cluster/submit_paper_table5_ablations.sh
#
# Full COCO on gpu_h200 (keep mweber jobs; see submit_coco_h200_relaunch.sh):
#   PAPER_T5_ARRAY=61-80 PAPER_T5_PARALLEL=12 PAPER_T5_PARTITION=gpu_h200 \
#     PAPER_T5_NAME_SUFFIX=_h200 PAPER_T5_TIME=72:00:00 \
#     bash bash_interface/cluster/submit_paper_table5_ablations.sh
#
# Then paste ARRAY JOBID into Paper_ablations.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_NUM_SEEDS:-5}"
NUM_DATASETS="${PAPER_T5_NUM_DATASETS:-4}"
NUM_VARIANTS="${PAPER_T5_NUM_VARIANTS:-4}"
NUM_TASKS="${PAPER_T5_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_PARALLEL:-18}"
PARTITION="${PAPER_T5_PARTITION:-mweber_gpu}"
NICE="${PAPER_T5_NICE:-10000}"
MEM="${PAPER_T5_MEM:-128GB}"
TIME="${PAPER_T5_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T5_WANDB_PREFIX:-paper_T5}"
NAME_SUFFIX="${PAPER_T5_NAME_SUFFIX:-}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_abl
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_abl_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_NUM_DATASETS="${NUM_DATASETS}",PAPER_T5_NUM_VARIANTS="${NUM_VARIANTS}",PAPER_T5_NUM_TASKS="${NUM_TASKS}",PAPER_T5_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T5_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T5_MAX_EPOCH="${PAPER_T5_MAX_EPOCH:-}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_ablations.sh
)"

cat <<EOF

=== SiGMA Table 5 ablations submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Name suffix:   ${NAME_SUFFIX:-<none>}
  Max epoch:     ${PAPER_T5_MAX_EPOCH:-<cfg default>}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/sigma_T5_abl_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B group pattern:  ${WANDB_PREFIX}_<dataset>_<Variant>
  W&B tags:           paper_table5, <Variant>, <dataset>, seed<k>

  Variants (exact names used in groups/tags):
    SiGMA           = best gated hybrid from Paper_final_runs.md
    SiGMA_ungated   = same heads, gate=none
    Attn_only       = MP heads → attention
    MP_only         = attention → same MP type

  Datasets / source runs / configs:
    peptides_func    l31u4b3k   peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml
    peptides_struct  bqkect9l   peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml
    voc              vyt7hjj5   voc-hybrid-j7ukyzdm-a2g2-anchor.yaml
    coco             xgjakrz0   coco-hybrid-5b4z9l3u-a1g1-anchor.yaml

  Paste JOBID into Paper_ablations.md

  Aggregate when done:
    for ds in peptides_func peptides_struct voc coco; do
      for v in SiGMA SiGMA_ungated Attn_only MP_only; do
        python scripts/api_wanndb_query/aggregate_paper_repro.py \\
          --group ${WANDB_PREFIX}_\${ds}_\${v} --metric best_test_perf --state finished
      done
    done

EOF
