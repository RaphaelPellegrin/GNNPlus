#!/usr/bin/env bash
# Submit node+graph gate dumps for paper SiGMA hetero (best LR, seed 2).
#
#   GATE_DUMP_LEVEL=both bash bash_interface/cluster/submit_dump_tu_hh_paper_node_gates.sh
#   TU_HH_NODE_ARRAY=2-6 bash bash_interface/cluster/submit_dump_tu_hh_paper_node_gates.sh  # skip mutag
#
# After jobs finish, rsync gate_values_per_node.pt and:
#   python scripts/gate_viz/plot_tu_hh_gates_batch.py \
#     --root results/tu_sigma_homo_hetero \
#     --out_dir results/gate_viz/tu_hh_hetero \
#     --datasets paper --variants SiGMA_hetero \
#     --seeds 2 --prefer-lr best_from_table --color-by-class \
#     --level node

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus
chmod +x bash_interface/cluster/run_dump_tu_hh_paper_node_gates.sh

ARRAY_SPEC="${TU_HH_NODE_ARRAY:-2-6}"
PARALLEL="${TU_HH_NODE_PARALLEL:-4}"
PARTITION="${TU_HH_NODE_PARTITION:-mweber_gpu}"
MEM="${TU_HH_NODE_MEM:-64GB}"
TIME="${TU_HH_NODE_TIME:-04:00:00}"
EPOCH="${GATE_DUMP_EPOCH:--1}"
LEVEL="${GATE_DUMP_LEVEL:-both}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_dump_tu_hh_paper_node_gates] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

job_id="$(
    sbatch --parsable \
        --job-name=tu_hh_ndump \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/tu_hh_ndump_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,GATE_DUMP_EPOCH="${EPOCH}",GATE_DUMP_LEVEL="${LEVEL}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}" \
        bash_interface/cluster/run_dump_tu_hh_paper_node_gates.sh
)"

cat <<EOF

=== Paper TU HH node gate dump submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (1=mutag … 6=reddit; default 2-6 skips mutag)
  Level:         ${LEVEL}
  Out:           \$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_SiGMA_hetero_<lr>_seed2/gate_values_per_node.pt
  Logs:          logs_gnnplus/tu_hh_ndump_${job_id}_<TASK>.log

EOF
