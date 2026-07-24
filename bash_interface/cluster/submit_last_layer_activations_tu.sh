#!/usr/bin/env bash
# Launch TU last-layer activation plots (MUTAG / ENZYMES / PROTEINS × SiGMA).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Activation figure (seed 0 only, 3 jobs):
#   bash bash_interface/cluster/submit_last_layer_activations_tu.sh
#
# Appendix Acc mean±std (seeds 0–4, 15 jobs):
#   ACT_ARRAY=1-15 ACT_PARALLEL=5 \
#     bash bash_interface/cluster/submit_last_layer_activations_tu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

ARRAY_SPEC="${ACT_ARRAY:-1-3}"
PARALLEL="${ACT_PARALLEL:-3}"
PARTITION="${ACT_PARTITION:-mweber_gpu}"
NICE="${ACT_NICE:-10000}"
MEM="${ACT_MEM:-64GB}"
# ENZYMES ogpkubk9 is L12 × up to 1000 epochs — keep walltime generous.
TIME="${ACT_TIME:-96:00:00}"

sbatch_args=(
    --parsable
    --job-name=tu_last_act
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_last_act_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_last_layer_activations_tu.sh
)"

cat <<EOF

=== TU last-layer activations submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (slot: 1=MUTAG, 2=ENZYMES ogpkubk9, 3=PROTEINS; seed=(task-1)//3)
  Parallel:      ${PARALLEL}
  Time:          ${TIME}
  Out:           \${GNNPLUS_OUT_DIR}/activations/<ds>_<tag>_seed<S>/
  Logs:          logs_gnnplus/tu_last_act_${job_id}_<TASK>.log
  W&B groups:    layer_act_{mutag,enzymes,proteins}

  Plots (per snapshot mid/last/best):
    *_all_layers_by_index.png   (x=graph index, y=mean node L2, one curve/layer)
    *_all_layers_heatmap.png
    *_last_layer_by_index.png
  Also:  summary.json (best_test_acc) + mid.pt / last.pt / best.pt

  Paste JOBID into Paper_last_layer_activations.md + CLUSTER_LAUNCHES.md

  Appendix Acc (5 seeds):
    ACT_ARRAY=1-15 ACT_PARALLEL=5 bash bash_interface/cluster/submit_last_layer_activations_tu.sh

EOF
