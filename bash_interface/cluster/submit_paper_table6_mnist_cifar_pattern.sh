#!/usr/bin/env bash
# Launch SiGMA Table 7 (code paper_T6) for MNIST + CIFAR10 + PATTERN.
#
# Baselines already have multiple homogeneous MP heads. Launch only:
#   Homog_MP_ungated / Hetero_MP / Hetero_MP_ungated
# SiGMA / Homog_MP gated → reuse paper_bestmodel (do not relaunch).
#
# Hetero = swap ONE (last) MP head to a different type.
# 3 datasets × 3 variants × 5 seeds = 45 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Cancel broken 75-job array if needed, then:
#   scancel 35720920
#   bash bash_interface/cluster/submit_paper_table6_mnist_cifar_pattern.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T6_MC_NUM_SEEDS:-5}"
NUM_DATASETS="${PAPER_T6_MC_NUM_DATASETS:-3}"
NUM_VARIANTS="${PAPER_T6_MC_NUM_VARIANTS:-3}"
NUM_TASKS="${PAPER_T6_MC_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T6_MC_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T6_MC_PARALLEL:-10}"
PARTITION="${PAPER_T6_MC_PARTITION:-mweber_gpu}"
NICE="${PAPER_T6_MC_NICE:-10000}"
MEM="${PAPER_T6_MC_MEM:-96GB}"
TIME="${PAPER_T6_MC_TIME:-96:00:00}"
WANDB_PREFIX="${PAPER_T6_MC_WANDB_PREFIX:-paper_T6}"
NAME_SUFFIX="${PAPER_T6_MC_NAME_SUFFIX:-}"

sbatch_args=(
    --parsable
    --job-name=sigma_T6_mc
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T6_mc_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T6_MC_NUM_SEEDS="${NUM_SEEDS}",PAPER_T6_MC_NUM_DATASETS="${NUM_DATASETS}",PAPER_T6_MC_NUM_VARIANTS="${NUM_VARIANTS}",PAPER_T6_MC_NUM_TASKS="${NUM_TASKS}",PAPER_T6_MC_WANDB_PREFIX="${WANDB_PREFIX}",PAPER_T6_MC_NAME_SUFFIX="${NAME_SUFFIX}",PAPER_T6_MC_MAX_EPOCH="${PAPER_T6_MC_MAX_EPOCH:-}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table6_mnist_cifar_pattern.sh
)"

cat <<EOF

=== SiGMA Table 7 MNIST+CIFAR10+PATTERN (paper_T6_*) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Name suffix:   ${NAME_SUFFIX:-<none>}
  Logs:          logs_gnnplus/sigma_T6_mc_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B group pattern:  ${WANDB_PREFIX}_<dataset>_<Variant>

  Launching only (SiGMA/Homog_MP gated = reuse paper_bestmodel):
    Homog_MP_ungated     = same homog types, gate=none
    Hetero_MP            = swap ONE last MP head, gated
    Hetero_MP_ungated    = same one-head swap, gate=none

  Hetero one-head swap:
    mnist     GATEDGCN,GATEDGCN                 → GATEDGCN,GCN
    cifar10   GATEDGCN×4                        → GATEDGCN,GATEDGCN,GATEDGCN,GCN
    pattern   GCNE,GCNE                         → GCNE,GINE

  Paste JOBID into Paper_table6_mnist_cifar_pattern.md + CLUSTER_LAUNCHES.md

  Aggregate when done:
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6mc

EOF
