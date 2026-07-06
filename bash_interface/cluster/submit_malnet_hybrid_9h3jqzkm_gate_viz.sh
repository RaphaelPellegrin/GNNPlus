#!/usr/bin/env bash
# Submit MalNet-Tiny hybrid gate-viz training (single GPU, seed 2 by default).
#
# Trains with ckpt every 50 epochs into a dedicated out_dir for gate visualization.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_gate_viz.sh
#
# Optional overrides:
#   GATE_VIZ_SEED=0 GATE_VIZ_OUT_DIR=results/gate_viz_malnet_seed0 \
#     bash bash_interface/cluster/submit_malnet_hybrid_9h3jqzkm_gate_viz.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

seed="${GATE_VIZ_SEED:-2}"
max_epoch="${GATE_VIZ_MAX_EPOCH:-250}"
min_lr="${GATE_VIZ_MIN_LR:-1e-6}"
ckpt_period="${GATE_VIZ_CKPT_PERIOD:-50}"
out_dir="${GATE_VIZ_OUT_DIR:-results/gate_viz_malnet_9h3jqzkm_seed${seed}}"
wandb_name="${GATE_VIZ_WANDB_NAME:-malnet_gate_viz_seed${seed}}"
mem="${GATE_VIZ_MEM:-64GB}"
time_limit="${GATE_VIZ_TIME:-48:00:00}"

job_id="$(
    sbatch --parsable \
        --job-name=malnet_gate_viz \
        --partition=mweber_gpu \
        --mem="${mem}" \
        --time="${time_limit}" \
        --gpus=1 \
        --output="logs_gnnplus/malnet_gate_viz_%j.log" \
        --export=ALL,ENV_NAME=gnnplus,GATE_VIZ_SEED="${seed}",GATE_VIZ_MAX_EPOCH="${max_epoch}",GATE_VIZ_MIN_LR="${min_lr}",GATE_VIZ_CKPT_PERIOD="${ckpt_period}",GATE_VIZ_OUT_DIR="${out_dir}",GATE_VIZ_WANDB_NAME="${wandb_name}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}" \
        bash_interface/cluster/run_malnet_hybrid_9h3jqzkm_gate_viz.sh
)"

echo ""
echo "=== MalNet gate-viz job submitted ==="
echo "  JOBID:        ${job_id}"
echo "  Seed:         ${seed}"
echo "  out_dir:      ${out_dir}"
echo "  ckpt_period:  ${ckpt_period}  max_epoch=${max_epoch}  min_lr=${min_lr}"
echo "  Config:       configs/gated_hybrid/malnet-hybrid-9h3jqzkm-anchor.yaml"
echo "  W&B name:     ${wandb_name}"
echo "  Log:          logs_gnnplus/malnet_gate_viz_${job_id}.log"
echo ""
echo "Checkpoints when done:"
echo "  ls -lh ${out_dir}/ckpt/"
echo ""
