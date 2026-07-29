#!/usr/bin/env bash
# Submit ENZYMES SiGMA (ogpkubk9) gate-viz training (single GPU).
#
# Trains with ckpt every 50 epochs into a dedicated out_dir so you can reload
# the model and dump per-graph / per-head gate values offline.
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh
#
# Optional overrides:
#   GATE_VIZ_SEED=0 GATE_VIZ_SCHEDULER=cosine \
#     bash bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

seed="${GATE_VIZ_SEED:-2}"
scheduler="${GATE_VIZ_SCHEDULER:-plateau}"
ckpt_period="${GATE_VIZ_CKPT_PERIOD:-50}"
max_epoch="${GATE_VIZ_MAX_EPOCH:-}"
mem="${GATE_VIZ_MEM:-64GB}"
time_limit="${GATE_VIZ_TIME:-48:00:00}"
partition="${GATE_VIZ_PARTITION:-mweber_gpu}"

if [ -n "${GATE_VIZ_OUT_DIR:-}" ]; then
    out_dir="${GATE_VIZ_OUT_DIR}"
elif [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
else
    out_dir="results/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
fi
wandb_name="${GATE_VIZ_WANDB_NAME:-enzymes_gate_viz_${scheduler}_seed${seed}}"

export_vars="ALL,ENV_NAME=gnnplus"
export_vars+=",GATE_VIZ_SEED=${seed}"
export_vars+=",GATE_VIZ_SCHEDULER=${scheduler}"
export_vars+=",GATE_VIZ_CKPT_PERIOD=${ckpt_period}"
export_vars+=",GATE_VIZ_OUT_DIR=${out_dir}"
export_vars+=",GATE_VIZ_WANDB_NAME=${wandb_name}"
export_vars+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_vars+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR:-}"
if [ -n "${max_epoch}" ]; then
    export_vars+=",GATE_VIZ_MAX_EPOCH=${max_epoch}"
fi

job_id="$(
    sbatch --parsable \
        --job-name=enz_gate_viz \
        --partition="${partition}" \
        --mem="${mem}" \
        --time="${time_limit}" \
        --gpus=1 \
        --output="logs_gnnplus/enz_gate_viz_%j.log" \
        --export="${export_vars}" \
        bash_interface/cluster/run_enzymes_ogpkubk9_gate_viz.sh
)"

echo ""
echo "=== ENZYMES gate-viz job submitted ==="
echo "  JOBID:         ${job_id}"
echo "  Seed:          ${seed}"
echo "  Scheduler:     ${scheduler}"
echo "  out_dir:       ${out_dir}"
echo "  ckpt_period:   ${ckpt_period}"
echo "  Config:        enzymes-hybrid-ogpkubk9-a4g4-${scheduler}-anchor.yaml"
echo "  W&B name:      ${wandb_name}"
echo "  Log:           logs_gnnplus/enz_gate_viz_${job_id}.log"
echo ""
echo "After training, dump per-graph gates from the checkpoint:"
echo "  ls -lh ${out_dir}/ckpt/"
echo "  python scripts/gate_viz/dump_per_graph_gates.py \\"
echo "    --run_dir ${out_dir} \\"
echo "    --cfg configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-${scheduler}-anchor.yaml"
echo ""
