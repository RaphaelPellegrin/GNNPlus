#!/usr/bin/env bash
# Submit TU gated SiGMA inference gate-clamp eval (Tab.17 + Tab.18 reported LRs).
#
# 2 tables × 6 datasets × 5 seeds = 60 jobs.
# Needs existing gated ckpts under $GNNPLUS_OUT_DIR (not local empty ckpt/).
#
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_tu_gate_clamp.sh
#
# Smoke (MUTAG Tab.17 seed0 = task 1):
#   TU_GATE_CLAMP_ARRAY=1 bash bash_interface/cluster/submit_tu_gate_clamp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus results/gate_clamp

NUM_TASKS="${TU_GATE_CLAMP_NUM_TASKS:-60}"
ARRAY_SPEC="${TU_GATE_CLAMP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_GATE_CLAMP_PARALLEL:-20}"
PARTITION="${TU_GATE_CLAMP_PARTITION:-mweber_gpu}"
NICE="${TU_GATE_CLAMP_NICE:-10000}"
MEM="${TU_GATE_CLAMP_MEM:-64GB}"
TIME="${TU_GATE_CLAMP_TIME:-4:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_tu_gate_clamp] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_tu_gate_clamp.sh

sbatch_args=(
    --parsable
    --job-name=tu_gate_clamp
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_gate_clamp_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_tu_gate_clamp.sh
)"

cat <<EOF

=== TU gate-clamp eval submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (Tab.17 = 1–30, Tab.18 = 31–60)
  Parallel:      ${PARALLEL}
  Logs:          logs_gnnplus/tu_gate_clamp_${job_id}_<TASK>.log
  Per-run CSV:   results/gate_clamp/t{17,18}_<ds>_<lr>_seed<s>.csv
  Ckpts:         \$GNNPLUS_OUT_DIR/tu_sigma_{homo_hetero,1x_gcn}/

  After all finish, merge:
    cat results/gate_clamp/t17_*.csv results/gate_clamp/t18_*.csv \\
      | awk 'NR==1 || !/^dataset/' > results/gate_clamp/tu_gate_clamp_all.csv
    python scripts/gate_viz/eval_gate_clamp.py --help  # see --paired-ttest on merge via notebook

  Paste JOBID into Paper_tu_sigma_homo_hetero.md + CLUSTER_LAUNCHES.md

EOF
