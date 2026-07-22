#!/usr/bin/env bash
# Launch peptides-func SiGMA (o5cdk766) VN×LR grid on mweber_gpu.
#
# 10 configs × 5 seeds = 50 jobs, ≤5 GPUs concurrent, 192h (ep=900).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_peptides_func_o5cdk766_vn_lr_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PEP_FUNC_VN_LR_NUM_SEEDS:-5}"
NUM_CFGS="${PEP_FUNC_VN_LR_NUM_CFGS:-10}"
SEED_OFFSET="${PEP_FUNC_VN_LR_SEED_OFFSET:-0}"
NUM_TASKS="${PEP_FUNC_VN_LR_NUM_TASKS:-$((NUM_CFGS * NUM_SEEDS))}"
ARRAY_SPEC="${PEP_FUNC_VN_LR_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PEP_FUNC_VN_LR_PARALLEL:-5}"
PARTITION="${PEP_FUNC_VN_LR_PARTITION:-mweber_gpu}"
NICE="${PEP_FUNC_VN_LR_NICE:-10000}"
MEM="${PEP_FUNC_VN_LR_MEM:-128GB}"
TIME="${PEP_FUNC_VN_LR_TIME:-192:00:00}"

sbatch_args=(
    --parsable
    --job-name=pep_func_vn_lr
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/pep_func_vn_lr_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PEP_FUNC_VN_LR_NUM_SEEDS="${NUM_SEEDS}",PEP_FUNC_VN_LR_NUM_CFGS="${NUM_CFGS}",PEP_FUNC_VN_LR_SEED_OFFSET="${SEED_OFFSET}",PEP_FUNC_VN_LR_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_o5cdk766_vn_lr_grid.sh
)"

cat <<EOF

=== Peptides-func o5cdk766 VN×LR grid submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_CFGS} configs × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/pep_func_vn_lr_${job_id}_<TASK>.log

  Anchor: peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml (a1g1 GCN, ep=900)
  Base lr: ≈2.083e-4

  Configs (cfg_idx → vn, lr, readout):
    0  novn  2.083e-4  nopyr   (paper control)
    1  vn1   2.083e-4  pyramid
    2  vn2   2.083e-4  pyramid
    3  vn4   2.083e-4  pyramid
    4  vn8   2.083e-4  pyramid
    5  vn4   1e-4      pyramid
    6  vn4   4e-4      pyramid
    7  vn4   2.083e-4  nopyr
    8  vn2   4e-4      pyramid
    9  vn8   1e-4      pyramid

  W&B groups: paper_sigma_peptides_func_<novn|vnK>_lr<tag>_<pyr|nopyr>

  Aggregate one cell:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group paper_sigma_peptides_func_vn4_lr2p083e-4_pyr \\
      --metric best_test_perf --state finished

  Paste JOBID into CLUSTER_LAUNCHES.md

EOF
