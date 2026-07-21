#!/usr/bin/env bash
# Launch SiGMA paper Table 6 PascalVOC-SP homogeneous-MP ablations (fresh runs).
#
# 2 variants × 5 seeds = 10 jobs.
#   Homog_MP          = GATEDGCN,GATEDGCN + headwise (= SiGMA arch, new W&B group)
#   Homog_MP_ungated  = GATEDGCN,GATEDGCN + gate=none
#
# Max concurrent GPUs: PAPER_T6_VOC_HOMOG_PARALLEL (default 5).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table6_voc_homog.sh
#
# Then paste ARRAY JOBID into Paper_table6_voc.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T6_VOC_HOMOG_NUM_SEEDS:-5}"
NUM_VARIANTS="${PAPER_T6_VOC_HOMOG_NUM_VARIANTS:-2}"
NUM_TASKS="${PAPER_T6_VOC_HOMOG_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T6_VOC_HOMOG_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T6_VOC_HOMOG_PARALLEL:-5}"
NICE="${PAPER_T6_VOC_HOMOG_NICE:-10000}"
MEM="${PAPER_T6_VOC_HOMOG_MEM:-128GB}"
TIME="${PAPER_T6_VOC_HOMOG_TIME:-120:00:00}"
WANDB_PREFIX="${PAPER_T6_VOC_HOMOG_WANDB_PREFIX:-paper_T6}"

sbatch_args=(
    --parsable
    --job-name=sigma_T6_voc_homog
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T6_voc_homog_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T6_VOC_HOMOG_NUM_SEEDS="${NUM_SEEDS}",PAPER_T6_VOC_HOMOG_NUM_VARIANTS="${NUM_VARIANTS}",PAPER_T6_VOC_HOMOG_NUM_TASKS="${NUM_TASKS}",PAPER_T6_VOC_HOMOG_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table6_voc_homog.sh
)"

cat <<EOF

=== SiGMA Table 6 VOC Homog_MP submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T6_voc_homog_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B groups:
    ${WANDB_PREFIX}_voc_Homog_MP
    ${WANDB_PREFIX}_voc_Homog_MP_ungated

  Variants:
    Homog_MP          = GATEDGCN,GATEDGCN + headwise (SiGMA arch, fresh seeds)
    Homog_MP_ungated  = GATEDGCN,GATEDGCN + gate=none

  Anchor / source:
    voc  vyt7hjj5  voc-hybrid-j7ukyzdm-a2g2-anchor.yaml

  Paste JOBID into Paper_table6_voc.md + CLUSTER_LAUNCHES.md

  Aggregate when done:
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6

EOF
