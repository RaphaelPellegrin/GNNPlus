#!/usr/bin/env bash
# Submit the full Appendix H analysis package for paper + d_h-fill tracks,
# writing under results/gcn_gin_routing_2/ (does not overwrite gcn_gin_routing/).
#
# Pipeline (cluster):
#   1) analyze   — per-type acc + root gate figs
#   2) mask      — head-masking ablation (afterok:analyze)
#   3) pairwise  — GCN-only vs GIN-only per graph (afterok:analyze)
#   4) opposite  — opposite-sign pair outcomes + gated (afterok:pairwise)
#   5) dump      — per-node gates for gated runs (parallel with mask)
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_gcn_gin_routing_2_pipeline.sh
#
# Optional:
#   GCN_GIN_R2_TRACKS=toy_dh2,toy_dh3,...   # default: all 8
#   GCN_GIN_R2_SKIP_DUMP=1
#   GCN_GIN_R2_SKIP_MASK=1
#   GCN_GIN_R2_SKIP_PAIRWISE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus results/gcn_gin_routing_2/analysis

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
  export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
fi
if [ -z "${GNNPLUS_DATASET_DIR:-}" ]; then
  export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
fi

RESULTS_ROOT="${GCN_GIN_R2_RESULTS_ROOT:-${GNNPLUS_OUT_DIR}/gcn_gin_routing}"
OUT_DIR="${GCN_GIN_R2_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing_2/analysis}"
TRACKS="${GCN_GIN_R2_TRACKS:-toy,sigma,toy_dh2,toy_dh3,toy_dh4,sigma_dh1,sigma_dh2,sigma_dh3}"
LR_TAG="${GCN_GIN_R2_LR_TAG:-lr001}"

mkdir -p "${OUT_DIR}"

echo "=== GCN/GIN routing_2 Appendix H pipeline ==="
echo "  Results root: ${RESULTS_ROOT}"
echo "  Out dir:      ${OUT_DIR}"
echo "  Tracks:       ${TRACKS}"
echo "  LR:           ${LR_TAG}"
echo ""

# --- 1) Analyze ---
export GCN_GIN_ANALYZE_RESULTS_ROOT="${RESULTS_ROOT}"
export GCN_GIN_ANALYZE_OUT_DIR="${OUT_DIR}"
export GCN_GIN_ANALYZE_TRACKS="${TRACKS}"
export GCN_GIN_ANALYZE_LR_TAG="${LR_TAG}"
export GCN_GIN_ANALYZE_TIME="${GCN_GIN_ANALYZE_TIME:-08:00:00}"

analyze_out="$(bash bash_interface/cluster/submit_analyze_gcn_gin_routing_results.sh)"
echo "${analyze_out}"
analyze_job="$(echo "${analyze_out}" | sed -n 's/.*JOBID:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
if [ -z "${analyze_job}" ]; then
  echo "ERROR: could not parse analyze JOBID" >&2
  exit 1
fi
echo "Parsed ANALYZE_JOB=${analyze_job}"

# --- 2) Mask (after analyze) ---
mask_job=""
if [ "${GCN_GIN_R2_SKIP_MASK:-0}" != "1" ]; then
  export GCN_GIN_MASK_RESULTS_ROOT="${RESULTS_ROOT}"
  export GCN_GIN_MASK_OUT_DIR="${OUT_DIR}"
  export GCN_GIN_MASK_TRACKS="${TRACKS}"
  export GCN_GIN_MASK_LR_TAG="${LR_TAG}"
  export GCN_GIN_MASK_TIME="${GCN_GIN_MASK_TIME:-04:00:00}"
  export GCN_GIN_MASK_DEPENDENCY="afterok:${analyze_job}"
  mask_out="$(bash bash_interface/cluster/submit_eval_gcn_gin_routing_masks.sh)"
  echo "${mask_out}"
  mask_job="$(echo "${mask_out}" | sed -n 's/.*JOBID:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
fi

# --- 3) Pairwise (after analyze) ---
pairwise_job=""
if [ "${GCN_GIN_R2_SKIP_PAIRWISE:-0}" != "1" ]; then
  export GCN_GIN_PAIRWISE_RESULTS_ROOT="${RESULTS_ROOT}"
  export GCN_GIN_PAIRWISE_OUT_DIR="${OUT_DIR}"
  export GCN_GIN_PAIRWISE_TRACKS="${TRACKS}"
  export GCN_GIN_PAIRWISE_LR_TAG="${LR_TAG}"
  export GCN_GIN_PAIRWISE_TIME="${GCN_GIN_PAIRWISE_TIME:-04:00:00}"
  # submit_compare has no dependency hook — add via sbatch override if needed.
  # Re-submit with dependency by wrapping: use env only when script supports it.
  pairwise_out="$(
    # Temporary: inject dependency via sbatch if compare script lacks it —
    # call run via a one-shot dependency sbatch.
    TRACKS_EXPORT="${TRACKS//,/\;}"
    sbatch --parsable \
      --job-name=gcn_gin_pairwise \
      --partition="${GCN_GIN_PAIRWISE_PARTITION:-mweber_gpu}" \
      --mem="${GCN_GIN_PAIRWISE_MEM:-16GB}" \
      --time="${GCN_GIN_PAIRWISE_TIME:-04:00:00}" \
      --gpus=1 \
      --dependency="afterok:${analyze_job}" \
      --output="logs_gnnplus/gcn_gin_pairwise_%j.log" \
      --export="ALL,ENV_NAME=gnnplus,GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR},GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR},GCN_GIN_PAIRWISE_RESULTS_ROOT=${RESULTS_ROOT},GCN_GIN_PAIRWISE_OUT_DIR=${OUT_DIR},GCN_GIN_PAIRWISE_LR_TAG=${LR_TAG},GCN_GIN_PAIRWISE_TRACKS=${TRACKS_EXPORT}" \
      bash_interface/cluster/run_compare_gcn_gin_baselines_per_graph.sh
  )"
  pairwise_job="${pairwise_out}"
  echo "=== Pairwise submitted JOBID=${pairwise_job} (afterok:${analyze_job}) ==="
fi

# --- 4) Opposite-sign pairs (after pairwise; include gated) ---
opp_job=""
if [ "${GCN_GIN_R2_SKIP_PAIRWISE:-0}" != "1" ] && [ -n "${pairwise_job}" ]; then
  export GCN_GIN_OPPOSITE_RESULTS_ROOT="${RESULTS_ROOT}"
  export GCN_GIN_OPPOSITE_OUT_DIR="${OUT_DIR}"
  export GCN_GIN_OPPOSITE_TRACKS="${TRACKS}"
  export GCN_GIN_OPPOSITE_LR_TAG="${LR_TAG}"
  export GCN_GIN_OPPOSITE_INCLUDE_GATED=1
  export GCN_GIN_OPPOSITE_FROM_CSV="${OUT_DIR}/pairwise_baseline_per_graph.csv"
  TRACKS_EXPORT="${TRACKS//,/\;}"
  opp_job="$(
    sbatch --parsable \
      --job-name=gcn_gin_opp_pairs \
      --partition="${GCN_GIN_OPPOSITE_PARTITION:-mweber_gpu}" \
      --mem="${GCN_GIN_OPPOSITE_MEM:-16GB}" \
      --time="${GCN_GIN_OPPOSITE_TIME:-02:00:00}" \
      --gpus=1 \
      --dependency="afterok:${pairwise_job}" \
      --output="logs_gnnplus/gcn_gin_opp_pairs_%j.log" \
      --export="ALL,ENV_NAME=gnnplus,GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR},GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR},GCN_GIN_OPPOSITE_RESULTS_ROOT=${RESULTS_ROOT},GCN_GIN_OPPOSITE_OUT_DIR=${OUT_DIR},GCN_GIN_OPPOSITE_LR_TAG=${LR_TAG},GCN_GIN_OPPOSITE_FROM_CSV=${OUT_DIR}/pairwise_baseline_per_graph.csv,GCN_GIN_OPPOSITE_INCLUDE_GATED=1,GCN_GIN_OPPOSITE_TRACKS=${TRACKS_EXPORT}" \
      bash_interface/cluster/run_analyze_opposite_sign_pairs.sh
  )"
  echo "=== Opposite-sign pairs submitted JOBID=${opp_job} (afterok:${pairwise_job}) ==="
fi

# --- 5) Node gate dump (parallel with mask; after analyze not required but safer) ---
dump_job=""
if [ "${GCN_GIN_R2_SKIP_DUMP:-0}" != "1" ]; then
  export GCN_GIN_GATE_DUMP_RESULTS_ROOT="${RESULTS_ROOT}"
  export GCN_GIN_GATE_DUMP_TRACKS="${TRACKS}"
  export GCN_GIN_GATE_DUMP_DEPENDENCY="afterok:${analyze_job}"
  dump_out="$(bash bash_interface/cluster/submit_dump_gcn_gin_routing_node_gates.sh)"
  echo "${dump_out}"
  dump_job="$(echo "${dump_out}" | sed -n 's/.*JOBID:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
fi

cat <<EOF

=== routing_2 pipeline submitted ===
  ANALYZE:   ${analyze_job}
  MASK:      ${mask_job:-skipped}
  PAIRWISE:  ${pairwise_job:-skipped}
  OPPOSITE:  ${opp_job:-skipped}
  GATE DUMP: ${dump_job:-skipped}

  Out: ${OUT_DIR}

Monitor:
  squeue -u \$USER | grep gcn_gin
  tail -f logs_gnnplus/gcn_gin_analyze_${analyze_job}.log

After jobs finish — inspect node gates + pack:
  for t in ${TRACKS//,/ }; do
    python scripts/synthetic/inspect_gcn_gin_routing_node_gates.py \\
      --results-root ${RESULTS_ROOT}/\$t --split test \\
      --export-csv ${OUT_DIR}/gate_node_summary_\${t}_test.csv
  done
  bash bash_interface/cluster/pack_gcn_gin_routing_2_for_pull.sh

Local polish (Mac, after pull):
  python scripts/synthetic/plot_gcn_gin_routing_paper_figures.py \\
    --analysis-dir results/gcn_gin_routing_2/analysis

Log JOBIDs in CLUSTER_LAUNCHES.md / Paper_gcn_gin_routing_synthetic.md.

EOF
