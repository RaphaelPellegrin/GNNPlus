#!/usr/bin/env bash
# Pack GCN/GIN routing_2 (d_h fill + paper tracks analysis) into ONE tarball.
#
# Usage (on cluster, after pipeline finishes):
#   bash bash_interface/cluster/pack_gcn_gin_routing_2_for_pull.sh
#
# Then on Mac:
#   scp fasrc:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull/gcn_gin_routing_2_bundle.tar.gz /tmp/
#   mkdir -p results/gcn_gin_routing_2
#   tar -xzf /tmp/gcn_gin_routing_2_bundle.tar.gz -C results/gcn_gin_routing_2/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
NET_ROOT="${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gcn_gin_routing"
ANALYSIS_SRC="${GCN_GIN_R2_OUT_DIR:-${REPO_ROOT}/results/gcn_gin_routing_2/analysis}"
OUT_DIR="${GCN_GIN_PACK_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull}"
BUNDLE="${OUT_DIR}/gcn_gin_routing_2_bundle.tar.gz"
STAGING="${OUT_DIR}/gcn_gin_routing_2_staging"
TRACKS="${GCN_GIN_R2_TRACKS:-toy,sigma,toy_dh2,toy_dh3,toy_dh4,sigma_dh1,sigma_dh2,sigma_dh3}"

echo "=== Pack GCN/GIN routing_2 bundle ==="
echo "  Analysis: ${ANALYSIS_SRC}"
echo "  Net:      ${NET_ROOT}"
echo "  Output:   ${BUNDLE}"
echo ""

rm -rf "${STAGING}"
mkdir -p "${STAGING}/analysis" "${STAGING}/gates" "${STAGING}/logs" "${OUT_DIR}"

if [ ! -d "${ANALYSIS_SRC}" ]; then
  echo "ERROR: missing analysis dir ${ANALYSIS_SRC}" >&2
  exit 1
fi

echo "[1/3] Copy analysis artifacts..."
cp -a "${ANALYSIS_SRC}/." "${STAGING}/analysis/"

echo "[2/3] Copy gate_graph_summary.csv (gated runs)..."
IFS=',' read -r -a track_arr <<< "${TRACKS}"
for track in "${track_arr[@]}"; do
  for lr_tag in lr001 lr01; do
    for seed in 0 1 2 3 4; do
      run_name="a0g2_gated_${lr_tag}_seed${seed}"
      src="${NET_ROOT}/${track}/${run_name}/gate_graph_summary.csv"
      if [[ -f "${src}" ]]; then
        dest_dir="${STAGING}/gates/${track}/${run_name}"
        mkdir -p "${dest_dir}"
        cp "${src}" "${dest_dir}/gate_graph_summary.csv"
      fi
    done
  done
done

echo "[3/3] Copy recent SLURM logs (gcn_gin_*)..."
LOG_DIR="${REPO_ROOT}/logs_gnnplus"
shopt -s nullglob
for f in "${LOG_DIR}"/gcn_gin_analyze_*.log \
         "${LOG_DIR}"/gcn_gin_mask_*.log \
         "${LOG_DIR}"/gcn_gin_pairwise_*.log \
         "${LOG_DIR}"/gcn_gin_opp_pairs_*.log; do
  cp "${f}" "${STAGING}/logs/" 2>/dev/null || true
done
# Keep gate-dump logs small: only first few if huge.
count=0
for f in "${LOG_DIR}"/gcn_gin_gdump_*_*.log; do
  cp "${f}" "${STAGING}/logs/" 2>/dev/null || true
  count=$((count + 1))
  if [ "${count}" -ge 20 ]; then
    break
  fi
done
shopt -u nullglob

echo "Creating tarball..."
tar -czf "${BUNDLE}" -C "${STAGING}" .
rm -rf "${STAGING}"

echo ""
echo "Done."
ls -lh "${BUNDLE}"
echo ""
echo "On Mac:"
echo "  scp fasrc:${BUNDLE} /tmp/"
echo "  mkdir -p results/gcn_gin_routing_2"
echo "  tar -xzf /tmp/gcn_gin_routing_2_bundle.tar.gz -C results/gcn_gin_routing_2/"
