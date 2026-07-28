#!/usr/bin/env bash
# Launch SiGMA Table 7 (code paper_T6) for MNIST + CIFAR10 + PATTERN.
#
# 3 datasets × {SiGMA, Homog_MP, Hetero_MP, Homog_MP_ungated, Hetero_MP_ungated}
#   × 5 seeds = 75 jobs.
# Multi-MP VOC-style: keep head counts; ablate homog vs hetero MP ± gates.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   # once if needed:
#   # bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table6_mnist_cifar_pattern.sh
#
# Then paste ARRAY JOBID into Paper_table6_mnist_cifar_pattern.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T6_MC_NUM_SEEDS:-5}"
NUM_DATASETS="${PAPER_T6_MC_NUM_DATASETS:-3}"
NUM_VARIANTS="${PAPER_T6_MC_NUM_VARIANTS:-5}"
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
  W&B tags:           paper_table6, paper_table7, <Variant>, <dataset>, seed<k>

  Variants:
    SiGMA / Homog_MP     = homogeneous MP, gated (paper best arch)
    Hetero_MP            = heterogeneous MP types, gated
    Homog_MP_ungated     = homogeneous MP, gate=none
    Hetero_MP_ungated    = heterogeneous MP, gate=none

  Datasets / anchors / hetero mix:
    mnist     uh7nxm4e   a2g2 GATEDGCN×2   → hetero GATEDGCN,GCN
    cifar10   3tx560wq   a8g4 GATEDGCN×4   → hetero GATEDGCN,GCN×alt
    pattern   ta9qtxb9   a2g2 GCNE×2       → hetero GCNE,GINE

  Paste JOBID into Paper_table6_mnist_cifar_pattern.md + CLUSTER_LAUNCHES.md

  Aggregate when done:
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6mc

EOF
