#!/usr/bin/env bash
# Submit Xu-recipe SiGMA GCN+SAGE only (a0g2/a1g2 × gated/ungated × 2 ds × 5 seeds).
#
# 4 variants × 2 datasets × 5 seeds = 40 jobs. Prefer GPU routing without GIN.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch (H200, nice=0 recommended):
#   XU_GCS_PARTITION=gpu_h200 XU_GCS_PARALLEL=10 XU_GCS_NICE=0 \
#     bash bash_interface/cluster/submit_heterogeneity_xu_sigma_gcn_sage.sh
#
# Smoke (MUTAG a0g2_gated seed 0 = task 1):
#   XU_GCS_ARRAY=1 XU_GCS_NUM_TASKS=1 \
#     bash bash_interface/cluster/submit_heterogeneity_xu_sigma_gcn_sage.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${XU_GCS_NUM_SEEDS:-5}"
NUM_DATASETS="${XU_GCS_NUM_DATASETS:-2}"
NUM_VARIANTS="${XU_GCS_NUM_VARIANTS:-4}"
NUM_TASKS="${XU_GCS_NUM_TASKS:-$((NUM_VARIANTS * NUM_DATASETS * NUM_SEEDS))}"
ARRAY_SPEC="${XU_GCS_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${XU_GCS_PARALLEL:-10}"
PARTITION="${XU_GCS_PARTITION:-mweber_gpu}"
NICE="${XU_GCS_NICE:-10000}"
MEM="${XU_GCS_MEM:-32GB}"
if [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="${XU_GCS_TIME:-24:00:00}"
else
    TIME="${XU_GCS_TIME:-24:00:00}"
fi

export ENV_NAME=gnnplus
export XU_GCS_NUM_SEEDS="${NUM_SEEDS}"
export XU_GCS_NUM_TASKS="${NUM_TASKS}"
export XU_GCS_GATE_DUMP="${XU_GCS_GATE_DUMP:-1}"
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    export GNNPLUS_DATASET_DIR
fi
if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_xu_sigma_gcn_sage] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
export GNNPLUS_OUT_DIR

sbatch_args=(
    --parsable
    --job-name=xu_sigma_gcs
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/xu_sigma_gcs_%A_%a.log"
    --export=ALL
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_heterogeneity_xu_sigma_gcn_sage.sh
)"

cat <<EOF

=== Xu SiGMA GCN+SAGE (a0g2/a1g2 × gated/ungated) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}  nice=${NICE}
  Tasks:         ${ARRAY_SPEC}  (4 variants × 2 ds × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/xu_sigma_gcs_${job_id}_<TASK>.log
  Outs:          \$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage/
                 <ds>_SiGMA_hetero_<variant>_seed<s>/
                 ├── ckpt/
                 ├── config_used.yaml
                 └── gate_values_per_graph.pt
  Configs:       configs/heterogeneity/powerful_gnns/sigma-gcn-sage-{a0g2,a1g2}-{gated,ungated}-ckpt.yaml
  W&B groups:    xu_sigma_gcs_<variant>_{mutag,enzymes}

  Task map (seed fastest within each variant block of 10):
    1–10   a0g2_gated   (1–5 mutag, 6–10 enzymes)
    11–20  a0g2_ungated
    21–30  a1g2_gated
    31–40  a1g2_ungated

  After finish, join per variant (example a0g2_gated):
    python scripts/heterogeneity/join_tu_gate_operator_preference.py \\
      --datasets mutag,enzymes \\
      --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \\
      --gate-root results/heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage \\
      --lr-tag a0g2_gated --seeds 0,1,2,3,4 --operators GCN,SAGE \\
      --splits val,test --out-dir results/heterogeneity/tu_gate_bridge_analysis_gcs_a0g2_gated

  Tracker: Paper_tu_gate_hetero_bridge.md
  Paste JOBID into CLUSTER_LAUNCHES.md

EOF
