#!/usr/bin/env bash
# Submit GCN/GIN routing post-training analysis (SLURM GPU worker — no python on login).
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull   # after you push analyze script + these bash files
#   bash bash_interface/cluster/submit_analyze_gcn_gin_routing_results.sh
#
# Optional: wait for training arrays to finish:
#   GCN_GIN_ANALYZE_DEPENDENCY=afterok:42432154:42432155 \
#     bash bash_interface/cluster/submit_analyze_gcn_gin_routing_results.sh
#
# Optional overrides:
#   GCN_GIN_ANALYZE_RESULTS_ROOT  (default: $GNNPLUS_OUT_DIR/gcn_gin_routing)
#   GCN_GIN_ANALYZE_OUT_DIR       (default: $REPO/results/gcn_gin_routing/analysis)
#   GCN_GIN_ANALYZE_TRACKS        (default: toy,sigma — use toy;sigma in sbatch export)
#   GCN_GIN_ANALYZE_LR_TAG        (e.g. lr001 — filter runs)
#   GCN_GIN_ANALYZE_DEVICE        (auto|cpu|cuda)
#   GCN_GIN_ANALYZE_PARTITION / _MEM / _TIME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
  echo "[submit_analyze_gcn_gin_routing] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
  echo "[submit_analyze_gcn_gin_routing] GNNPLUS_DATASET_DIR unset → ${GNNPLUS_DATASET_DIR}"
fi

results_root="${GCN_GIN_ANALYZE_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
out_dir="${GCN_GIN_ANALYZE_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing/analysis}"

if [ ! -d "${results_root}" ]; then
  echo "ERROR: results root missing: ${results_root}"
  echo "  Train first: bash bash_interface/cluster/submit_gcn_gin_routing.sh both"
  exit 1
fi
if [ ! -f "${GNNPLUS_DATASET_DIR}/GcnGinRouting/processed/train.pt" ]; then
  echo "ERROR: dataset missing at ${GNNPLUS_DATASET_DIR}/GcnGinRouting"
  exit 1
fi
if [ ! -f "scripts/synthetic/analyze_gcn_gin_routing_results.py" ]; then
  echo "ERROR: analyze script missing — git pull after push"
  exit 1
fi
if ! grep -q "run_dir is not a YACS key" scripts/synthetic/analyze_gcn_gin_routing_results.py; then
  echo "ERROR: stale analyze script (missing run_dir fix) — need commit a5ef4ad+"
  echo "  git fetch origin && git pull"
  echo "  grep 'run_dir is not a YACS' scripts/synthetic/analyze_gcn_gin_routing_results.py"
  exit 1
fi

chmod +x bash_interface/cluster/run_analyze_gcn_gin_routing_results.sh

PARTITION="${GCN_GIN_ANALYZE_PARTITION:-mweber_gpu}"
MEM="${GCN_GIN_ANALYZE_MEM:-32GB}"
TIME="${GCN_GIN_ANALYZE_TIME:-02:00:00}"

# SLURM --export splits on commas; encode multi-track lists with semicolons.
_analyze_tracks_display="${GCN_GIN_ANALYZE_TRACKS:-toy,sigma}"
_analyze_tracks_export="${_analyze_tracks_display//,/\;}"

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR}"
export_list+=",GCN_GIN_ANALYZE_RESULTS_ROOT=${results_root}"
export_list+=",GCN_GIN_ANALYZE_OUT_DIR=${out_dir}"
export_list+=",GCN_GIN_ANALYZE_TRACKS=${_analyze_tracks_export}"
export_list+=",GCN_GIN_ANALYZE_DEVICE=${GCN_GIN_ANALYZE_DEVICE:-auto}"
if [ -n "${GCN_GIN_ANALYZE_LR_TAG:-}" ]; then
  export_list+=",GCN_GIN_ANALYZE_LR_TAG=${GCN_GIN_ANALYZE_LR_TAG}"
fi

job_id="$(
  sbatch --parsable \
    --job-name=gcn_gin_analyze \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --gpus=1 \
    --output="logs_gnnplus/gcn_gin_analyze_%j.log" \
    --export="${export_list}" \
    ${GCN_GIN_ANALYZE_DEPENDENCY:+--dependency="${GCN_GIN_ANALYZE_DEPENDENCY}"} \
    bash_interface/cluster/run_analyze_gcn_gin_routing_results.sh
)"

cat <<EOF

=== GCN/GIN routing analysis submitted ===
  JOBID:         ${job_id}
  Partition:     ${PARTITION} (1 GPU)
  Results:       ${results_root}
  Dataset:       ${GNNPLUS_DATASET_DIR}
  Output:        ${out_dir}
  Tracks:        ${_analyze_tracks_display} (sbatch export: ${_analyze_tracks_export})
  Logs:          logs_gnnplus/gcn_gin_analyze_${job_id}.log

Monitor:
  squeue -j ${job_id}
  tail -f logs_gnnplus/gcn_gin_analyze_${job_id}.log

EOF
