#!/usr/bin/env bash
# Launch peptides func+struct a0g2 UniGCN MP mixes (no attention).
#
# 2 ds × {UNIGCN+GINE, UNIGCN+GATEDGCN} × 5 seeds = 20 jobs.
# Max concurrent GPUs: PEP_UNIGCN_PARALLEL (default 8).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_peptides_unigcn_a0g2_mp_mixes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PEP_UNIGCN_NUM_SEEDS:-5}"
NUM_DATASETS="${PEP_UNIGCN_NUM_DATASETS:-2}"
NUM_VARIANTS="${PEP_UNIGCN_NUM_VARIANTS:-2}"
NUM_TASKS="${PEP_UNIGCN_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PEP_UNIGCN_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PEP_UNIGCN_PARALLEL:-8}"
NICE="${PEP_UNIGCN_NICE:-10000}"
MEM="${PEP_UNIGCN_MEM:-128GB}"
TIME="${PEP_UNIGCN_TIME:-120:00:00}"
WANDB_PREFIX="${PEP_UNIGCN_WANDB_PREFIX:-paper_peptides}"

sbatch_args=(
    --parsable
    --job-name=pep_unigcn_a0g2
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/pep_unigcn_a0g2_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PEP_UNIGCN_NUM_SEEDS="${NUM_SEEDS}",PEP_UNIGCN_NUM_DATASETS="${NUM_DATASETS}",PEP_UNIGCN_NUM_VARIANTS="${NUM_VARIANTS}",PEP_UNIGCN_NUM_TASKS="${NUM_TASKS}",PEP_UNIGCN_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_unigcn_a0g2_mp_mixes.sh
)"

cat <<EOF

=== Peptides UniGCN a0g2 MP mixes submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_DATASETS} ds × ${NUM_VARIANTS} mixes × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/pep_unigcn_a0g2_${job_id}_<TASK>.log

  HPs from best SiGMA lineages; attention removed (a0g2):
    peptides_func    homog a1g2 / o5cdk766  (AP 0.7080)
    peptides_struct  g3bsaq32 a1g1 GINE

  Mixes (gated):
    UNIGCN,GINE
    UNIGCN,GATEDGCN

  W&B groups: ${WANDB_PREFIX}_<ds>_a0g2_{UNIGCN_GINE,UNIGCN_GATEDGCN}

  Paste JOBID into Paper_peptides_unigcn_a0g2_mp_mixes.md + CLUSTER_LAUNCHES.md

  Aggregate:
    for ds in peptides_func peptides_struct; do
      for v in UNIGCN_GINE UNIGCN_GATEDGCN; do
        python scripts/api_wanndb_query/aggregate_paper_repro.py \\
          --group ${WANDB_PREFIX}_\${ds}_a0g2_\${v} --metric best_test_perf --state finished
      done
    done

EOF
