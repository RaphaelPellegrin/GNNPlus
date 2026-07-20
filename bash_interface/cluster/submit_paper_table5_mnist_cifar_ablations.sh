#!/usr/bin/env bash
# Launch SiGMA Table 5 ablations for MNIST + CIFAR10 on mweber_gpu.
#
# 2 datasets × {SiGMA, SiGMA_ungated, Attn_only, MP_only} × 5 seeds = 40 jobs.
# Max concurrent GPUs: PAPER_T5_MC_PARALLEL (default 10).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   # once, if needed:
#   # bash bash_interface/cluster/prep_gnnplus_datasets.sh mnist cifar10
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_mnist_cifar_ablations.sh
#
# Then paste ARRAY JOBID into Paper_ablations_mnist_cifar.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_MC_NUM_SEEDS:-5}"
NUM_DATASETS="${PAPER_T5_MC_NUM_DATASETS:-2}"
NUM_VARIANTS="${PAPER_T5_MC_NUM_VARIANTS:-4}"
NUM_TASKS="${PAPER_T5_MC_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_MC_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_MC_PARALLEL:-10}"
NICE="${PAPER_T5_MC_NICE:-10000}"
MEM="${PAPER_T5_MC_MEM:-64GB}"
TIME="${PAPER_T5_MC_TIME:-72:00:00}"
WANDB_PREFIX="${PAPER_T5_MC_WANDB_PREFIX:-paper_T5}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_mc
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_mc_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_MC_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_MC_NUM_DATASETS="${NUM_DATASETS}",PAPER_T5_MC_NUM_VARIANTS="${NUM_VARIANTS}",PAPER_T5_MC_NUM_TASKS="${NUM_TASKS}",PAPER_T5_MC_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_mnist_cifar_ablations.sh
)"

cat <<EOF

=== SiGMA Table 5 MNIST+CIFAR ablations submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_mc_${job_id}_<TASK>.log

  W&B entity/project: weber-geoml-harvard-university/GNNPlus
  W&B group pattern:  ${WANDB_PREFIX}_<dataset>_<Variant>
  W&B tags:           paper_table5, <Variant>, <dataset>, seed<k>

  Variants:
    SiGMA           = best gated hybrid (paper Table 3)
    SiGMA_ungated   = same heads, gate=none
    Attn_only       = MP heads → attention
    MP_only         = attention → same MP type

  Datasets / source runs / configs:
    mnist     uh7nxm4e   mnist-hybrid-lcvbyyss-a2g2-anchor.yaml   (a2g2 GATEDGCN)
    cifar10   3tx560wq   cifar10-hybrid-ulij45a2-anchor.yaml      (a8g4 GATEDGCN)

  Paste JOBID into Paper_ablations_mnist_cifar.md + CLUSTER_LAUNCHES.md

  Aggregate when done:
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc

EOF
