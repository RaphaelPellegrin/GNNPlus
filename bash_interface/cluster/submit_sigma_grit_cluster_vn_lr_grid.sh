#!/usr/bin/env bash
# Launch SiGMA+GRIT CLUSTER VN×LR grid on mweber_gpu.
#
# 10 configs × 5 seeds = 50 jobs, ≤5 GPUs concurrent.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_sigma_grit_cluster_vn_lr_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_GRIT_VN_LR_NUM_SEEDS:-5}"
NUM_CFGS="${SIGMA_GRIT_VN_LR_NUM_CFGS:-10}"
SEED_OFFSET="${SIGMA_GRIT_VN_LR_SEED_OFFSET:-0}"
NUM_TASKS="${SIGMA_GRIT_VN_LR_NUM_TASKS:-$((NUM_CFGS * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_GRIT_VN_LR_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_GRIT_VN_LR_PARALLEL:-5}"
PARTITION="${SIGMA_GRIT_VN_LR_PARTITION:-mweber_gpu}"
NICE="${SIGMA_GRIT_VN_LR_NICE:-10000}"
MEM="${SIGMA_GRIT_VN_LR_MEM:-128GB}"
TIME="${SIGMA_GRIT_VN_LR_TIME:-120:00:00}"

sbatch_args=(
    --parsable
    --job-name=sigma_grit_vn_lr
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_grit_vn_lr_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_GRIT_VN_LR_NUM_SEEDS="${NUM_SEEDS}",SIGMA_GRIT_VN_LR_NUM_CFGS="${NUM_CFGS}",SIGMA_GRIT_VN_LR_SEED_OFFSET="${SEED_OFFSET}",SIGMA_GRIT_VN_LR_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_grit_cluster_vn_lr_grid.sh
)"

cat <<EOF

=== SiGMA+GRIT CLUSTER VN×LR grid submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_CFGS} configs × ${NUM_SEEDS} seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Out dir:       ${GNNPLUS_OUT_DIR:-<cfg default>}
  Logs:          logs_gnnplus/sigma_grit_vn_lr_${job_id}_<TASK>.log

  Configs (cfg_idx → vn, lr):
    0  novn   1.492e-3
    1  vn1    1.492e-3
    2  vn2    1.492e-3
    3  vn4    1.492e-3
    4  vn8    1.492e-3
    5  vn4    5e-4
    6  vn4    1e-3
    7  vn4    3e-3
    8  vn8    1e-3
    9  vn2    3e-3

  W&B groups: paper_sigma_grit_cluster_<novn|vnK>_lr<tag>
  Anchor:     cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml
  Force:      gnn.hybrid.attn_type grit

  Aggregate one cell:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group paper_sigma_grit_cluster_vn4_lr1p492e-3 \\
      --metric best_test_perf --state finished

  Paste JOBID into Paper_sigma_grit_attn.md + CLUSTER_LAUNCHES.md

EOF
