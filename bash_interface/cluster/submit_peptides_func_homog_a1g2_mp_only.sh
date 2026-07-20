#!/usr/bin/env bash
# Launch Peptides-func MP-only (a0g3 GCN×3, gated) on NEW best Homog_MP a1g2.
#
# 5 seeds. Max concurrent GPUs: FUNC_HOMOG_MPONLY_PARALLEL (default 5).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_peptides_func_homog_a1g2_mp_only.sh
#
# Then paste ARRAY JOBID into Paper_peptides_func_homog_a1g2_mp_only.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${FUNC_HOMOG_MPONLY_NUM_SEEDS:-5}"
ARRAY_SPEC="${FUNC_HOMOG_MPONLY_ARRAY:-1-${NUM_SEEDS}}"
PARALLEL="${FUNC_HOMOG_MPONLY_PARALLEL:-5}"
NICE="${FUNC_HOMOG_MPONLY_NICE:-10000}"
MEM="${FUNC_HOMOG_MPONLY_MEM:-128GB}"
TIME="${FUNC_HOMOG_MPONLY_TIME:-120:00:00}"
WANDB_PREFIX="${FUNC_HOMOG_MPONLY_WANDB_PREFIX:-paper_T5}"

sbatch_args=(
    --parsable
    --job-name=sigma_func_a0g3
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_func_a0g3_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,FUNC_HOMOG_MPONLY_NUM_SEEDS="${NUM_SEEDS}",FUNC_HOMOG_MPONLY_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_peptides_func_homog_a1g2_mp_only.sh
)"

cat <<EOF

=== Peptides-func Homog_MP → MP_only (a0g3 GCN×3) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (5 seeds)
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_func_a0g3_${job_id}_<TASK>.log

  Model:  a0g3  gnn_types=GCN,GCN,GCN  gate=elementwise (from Homog a1g2)
  Anchor: configs/gated_hybrid/peptides-func-hybrid-homog-a1g2-gcn-anchor.yaml
  W&B group: ${WANDB_PREFIX}_peptides_func_HomogMP_MPonly

  Paste JOBID into Paper_peptides_func_homog_a1g2_mp_only.md + CLUSTER_LAUNCHES.md

  Aggregate when done:
    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
      --group ${WANDB_PREFIX}_peptides_func_HomogMP_MPonly \\
      --metric best_test_perf --state finished

EOF
