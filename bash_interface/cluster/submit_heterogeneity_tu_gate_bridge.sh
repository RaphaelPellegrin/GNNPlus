#!/usr/bin/env bash
# Launch MUTAG/ENZYMES heterogeneity profiles for gate–operator bridge.
#
# Goal: per-graph operator preference (GCN / GIN / SAGE / GatedGCN) to pair with
# SiGMA hetero gate dumps (Appendix F, results/gate_viz/tu_hh_hetero).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Full launch (8 jobs, ≥100 appearances — slow):
#   bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
#
# Smoke (2 appearances, cap trials):
#   HETERO_REQUIRED_TEST_APPEARANCES=2 HETERO_MAX_TRIALS=20 \
#     bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
#
# Pilot (10 appearances — still useful for join-script debugging):
#   HETERO_REQUIRED_TEST_APPEARANCES=10 HETERO_MAX_TRIALS=200 \
#     bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh
#
# Single task smoke (MUTAG GCN = task 1):
#   HETERO_ARRAY=1 HETERO_NUM_TASKS=1 HETERO_REQUIRED_TEST_APPEARANCES=2 \
#     HETERO_MAX_TRIALS=10 bash bash_interface/cluster/submit_heterogeneity_tu_gate_bridge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${HETERO_NUM_TASKS:-8}"
ARRAY_SPEC="${HETERO_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${HETERO_PARALLEL:-4}"
PARTITION="${HETERO_PARTITION:-mweber_gpu}"
NICE="${HETERO_NICE:-10000}"
MEM="${HETERO_MEM:-64GB}"
if [ -n "${HETERO_TIME:-}" ]; then
    TIME="${HETERO_TIME}"
elif [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="72:00:00"
else
    TIME="192:00:00"
fi
REQUIRED="${HETERO_REQUIRED_TEST_APPEARANCES:-100}"
MAX_TRIALS="${HETERO_MAX_TRIALS:-2000}"
SEED0="${HETERO_SEED0:-0}"
WANDB_USE="${HETERO_WANDB:-1}"

# Comma-separated lists MUST NOT appear inside ``--export=A=x,B=y`` — SLURM
# splits on commas, so ``HETERO_DATASETS=mutag,enzymes`` becomes only ``mutag``
# (and ``HETERO_MODELS=gcn``). That made array tasks 2–8 die with
# ``task_id out of range (1..1)``. Export them via the shell + ``--export=ALL``.
export ENV_NAME=gnnplus
export HETERO_REQUIRED_TEST_APPEARANCES="${REQUIRED}"
export HETERO_MAX_TRIALS="${MAX_TRIALS}"
export HETERO_SEED0="${SEED0}"
export HETERO_WANDB="${WANDB_USE}"
export HETERO_DATASETS="${HETERO_DATASETS:-mutag,enzymes}"
export HETERO_MODELS="${HETERO_MODELS:-gcn,gin,sage,gatedgcn}"
export HETERO_OUT_SUBDIR="${HETERO_OUT_SUBDIR:-heterogeneity/powerful_gnns/tu_gate_bridge}"
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    export GNNPLUS_DATASET_DIR
fi

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_gate_bridge] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
export GNNPLUS_OUT_DIR

sbatch_args=(
    --parsable
    --job-name=hetero_gate_bridge
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/hetero_gate_bridge_%A_%a.log"
    --export=ALL
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_heterogeneity_tu_gate_bridge.sh
)"

cat <<EOF

=== TU gate–operator bridge heterogeneity submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (datasets×models → see lists below)
  Datasets:      ${HETERO_DATASETS}
  Models:        ${HETERO_MODELS}
  Parallel:      ${PARALLEL} GPUs max
  Appearances:   ≥${REQUIRED} per graph
  Max trials:    ${MAX_TRIALS}
  W&B groups:    building_hetero_profile_<ds>_tu_gate_bridge
  Configs:       configs/heterogeneity/powerful_gnns/
  Outs:          \$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/<ds>_<model>/
  Logs:          logs_gnnplus/hetero_gate_bridge_${job_id}_<TASK>.log
  Mem / time:    ${MEM} / ${TIME}

  Task map (model cycles fastest):
    1 mutag_gcn        2 mutag_gin        3 mutag_sage       4 mutag_gatedgcn
    5 enzymes_gcn      6 enzymes_gin      7 enzymes_sage     8 enzymes_gatedgcn

  Gate dumps (already trained): \$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/
  Appendix F plots: results/gate_viz/tu_hh_hetero/

  After jobs finish, pull + join:
    bash bash_interface/local/pull_tu_gate_bridge_hetero.sh
    python scripts/heterogeneity/join_tu_gate_operator_preference.py \\
      --dataset mutag --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \\
      --gate-pt results/tu_sigma_homo_hetero/mutag_SiGMA_hetero_lr001_seed2/gate_values_per_graph.pt \\
      --out-dir results/heterogeneity/tu_gate_bridge_analysis/mutag

  Tracker: Paper_tu_gate_hetero_bridge.md
  Paste JOBID into Paper_tu_gate_hetero_bridge.md

EOF
