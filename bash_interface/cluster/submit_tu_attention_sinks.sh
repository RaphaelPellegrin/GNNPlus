#!/usr/bin/env bash
# Submit TU attention-sink training (MUTAG + ENZYMES × gated/ungated × SiGMA/GPS).
#
# 2 datasets × 4 variants × seed 2 = 8 jobs.
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus && git pull
#
# Launch:
#   bash bash_interface/cluster/submit_tu_attention_sinks.sh
#
# MUTAG-only smoke (tasks 1-4):
#   AS_ARRAY=1-4 AS_PARALLEL=4 bash bash_interface/cluster/submit_tu_attention_sinks.sh
#
# Also dump attention on the GPU node after train:
#   AS_DUMP_ATTN=1 bash bash_interface/cluster/submit_tu_attention_sinks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_VARIANTS="${AS_NUM_VARIANTS:-4}"
NUM_DATASETS="${AS_NUM_DATASETS:-2}"
NUM_TASKS="${AS_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS))}"
ARRAY_SPEC="${AS_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${AS_PARALLEL:-8}"
PARTITION="${AS_PARTITION:-mweber_gpu}"
NICE="${AS_NICE:-10000}"
MEM="${AS_MEM:-64GB}"
TIME="${AS_TIME:-48:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_attention_sinks] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_attention_sinks.sh

sbatch_args=(
    --parsable
    --job-name=tu_attn_sinks
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_attn_sinks_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,AS_NUM_VARIANTS="${NUM_VARIANTS}",AS_NUM_TASKS="${NUM_TASKS}",AS_DUMP_ATTN="${AS_DUMP_ATTN:-0}",AS_BASE_LR="${AS_BASE_LR:-0.001}",AS_SEED="${AS_SEED:-2}",AS_SINK_EVERY="${AS_SINK_EVERY:-50}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_attention_sinks.sh
)"

cat <<EOF

=== TU attention-sink campaign submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (MUTAG+ENZYMES × gated/ungated × SiGMA/GPS = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Outs:          \$GNNPLUS_OUT_DIR/tu_attention_sinks/
  W&B sinks:     log_attention_sinks=True · every AS_SINK_EVERY=${AS_SINK_EVERY:-50} epochs
                 (panels: by_layer_head, mean_over_heads, sink_rate, max_alpha, vnorm_ratio)
  Dump attn:     AS_DUMP_ATTN=${AS_DUMP_ATTN:-0}
  Logs:          logs_gnnplus/tu_attn_sinks_${job_id}_<TASK>.log
  Tracker:       Paper_attention_sinks.md

Local after rsync ckpt:
  python scripts/attention_sinks/dump_attention_maps.py \\
    --run_dir results/tu_attention_sinks/mutag_SiGMA_hetero_ungated_attn_lr001_seed2 \\
    --cfg configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-matched-anchor.yaml \\
    gnn.hybrid.gate none gnn.hybrid.mp_gate headwise

  # then in Heterogeneity_Profile:
  python scripts/plots/plot_attention_sinks_aggregate.py \\
    --input-dir ../GNNPlus/results/tu_attention_sinks/mutag_SiGMA_hetero_ungated_attn_lr001_seed2/attention_matrices \\
    --output-dir visualizations/attention_sinks/mutag_ungated_attn

EOF
