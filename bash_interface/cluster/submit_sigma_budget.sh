#!/usr/bin/env bash
# Launch SiGMA baby/tiny budget fills (14 families × 5 seeds = 70, %20 GPUs).
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_sigma_budget.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_BUDGET_NUM_SEEDS:-5}"
NUM_FAMILIES="${SIGMA_BUDGET_NUM_FAMILIES:-14}"
NUM_TASKS="${SIGMA_BUDGET_NUM_TASKS:-$((NUM_FAMILIES * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_BUDGET_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_BUDGET_PARALLEL:-20}"
PARTITION="${SIGMA_BUDGET_PARTITION:-mweber_gpu}"
NICE="${SIGMA_BUDGET_NICE:-10000}"
MEM="${SIGMA_BUDGET_MEM:-128GB}"
TIME="${SIGMA_BUDGET_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
  echo "[submit_sigma_budget] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

# Sanity: configs exist
missing=0
for cfg in \
  configs/gated_hybrid/budget/mnist-b500k-a1g1.yaml \
  configs/gated_hybrid/budget/cifar10-b500k-a1g1.yaml \
  configs/gated_hybrid/budget/cifar10-b1m-a1g1.yaml \
  configs/gated_hybrid/budget/cifar10-b2m-a1g2.yaml \
  configs/gated_hybrid/budget/pattern-b500k-a1g1-grit.yaml \
  configs/gated_hybrid/budget/pattern-b1m-a1g1-grit.yaml \
  configs/gated_hybrid/budget/cluster-b500k-a1g1.yaml \
  configs/gated_hybrid/budget/cluster-b1m-a1g1.yaml \
  configs/gated_hybrid/budget/peptides-struct-b500k-a1g1.yaml \
  configs/gated_hybrid/budget/voc-b500k-a1g1.yaml \
  configs/gated_hybrid/budget/voc-b1m-a1g1.yaml \
  configs/gated_hybrid/budget/voc-b2m-a1g1.yaml \
  configs/gated_hybrid/budget/coco-b500k-a1g1.yaml \
  configs/gated_hybrid/budget/malnet-b500k-a1g1.yaml
do
  if [ ! -f "${cfg}" ]; then
    echo "MISSING ${cfg}"
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "Generate configs: python scripts/generate_sigma_budget_configs.py"
  exit 1
fi

sbatch_args=(
  --parsable
  --job-name=sigma_budget
  --array="${ARRAY_SPEC}%${PARALLEL}"
  --partition="${PARTITION}"
  --mem="${MEM}"
  --time="${TIME}"
  --gpus=1
  --output="logs_gnnplus/sigma_budget_%A_%a.log"
  --export=ALL,ENV_NAME=gnnplus,SIGMA_BUDGET_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
  sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
  sbatch "${sbatch_args[@]}" \
    bash_interface/cluster/run_sigma_budget.sh
)"

cat <<EOF

=== SiGMA budget (baby/tiny) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_FAMILIES} families × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_budget_${job_id}_<TASK>.log
  Out:           \${GNNPLUS_OUT_DIR}/sigma_budget/<fam>_seed<s>/
  Docs:          Paper_sigma_budget.md

  Families (task blocks of ${NUM_SEEDS}):
    1–5    mnist_b500k
    6–10   cifar10_b500k
    11–15  cifar10_b1m
    16–20  cifar10_b2m
    21–25  pattern_b500k
    26–30  pattern_b1m
    31–35  cluster_b500k
    36–40  cluster_b1m
    41–45  peptides_struct_b500k
    46–50  voc_b500k
    51–55  voc_b1m
    56–60  voc_b2m
    61–65  coco_b500k
    66–70  malnet_b500k

  Skipped (already under budget / existing n≥5 alt):
    ZINC; MNIST/COCO/MalNet @1M+2M; PATTERN/CLUSTER @2M;
    Pep-func (zc371e1n a0g1); Pep-struct @1M+2M (rholn782)

  Paste JOBID into Paper_sigma_budget.md

EOF
