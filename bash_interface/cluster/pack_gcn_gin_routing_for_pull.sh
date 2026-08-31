#!/usr/bin/env bash
# Pack GCN/GIN routing artifacts into ONE tarball (run on cluster after one ssh login).
#
# Usage (on cluster):
#   bash bash_interface/cluster/pack_gcn_gin_routing_for_pull.sh
#   ls -lh /n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull/gcn_gin_routing_bundle.tar.gz
#
# Then on Mac (ONE scp):
#   scp fasrc:/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull/gcn_gin_routing_bundle.tar.gz /tmp/
#   tar -xzf /tmp/gcn_gin_routing_bundle.tar.gz -C results/gcn_gin_routing/
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
NET_ROOT="${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gcn_gin_routing"
OUT_DIR="${GCN_GIN_PACK_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull}"
BUNDLE="${OUT_DIR}/gcn_gin_routing_bundle.tar.gz"
STAGING="${OUT_DIR}/gcn_gin_routing_staging"

echo "=== Pack GCN/GIN routing bundle ==="
echo "  Repo:    ${REPO_ROOT}"
echo "  Net:     ${NET_ROOT}"
echo "  Output:  ${BUNDLE}"
echo ""

rm -rf "${STAGING}"
mkdir -p "${STAGING}/analysis" "${STAGING}/gates" "${STAGING}/logs" "${OUT_DIR}"

echo "[1/3] Copy analysis artifacts..."
cp -a "${REPO_ROOT}/results/gcn_gin_routing/analysis/." "${STAGING}/analysis/"

echo "[2/3] Copy gate_graph_summary.csv (20 gated runs)..."
for track in toy sigma; do
  for lr_tag in lr001 lr01; do
    for seed in 0 1 2 3 4; do
      run_name="a0g2_gated_${lr_tag}_seed${seed}"
      src="${NET_ROOT}/${track}/${run_name}/gate_graph_summary.csv"
      if [[ -f "${src}" ]]; then
        dest_dir="${STAGING}/gates/${track}/${run_name}"
        mkdir -p "${dest_dir}"
        cp "${src}" "${dest_dir}/gate_graph_summary.csv"
      else
        echo "  WARN missing: ${src}" >&2
      fi
    done
  done
done

echo "[3/3] Copy SLURM logs..."
LOG_DIR="${REPO_ROOT}/logs_gnnplus"
for f in gcn_gin_analyze_42666091.log; do
  [[ -f "${LOG_DIR}/${f}" ]] && cp "${LOG_DIR}/${f}" "${STAGING}/logs/"
done
for f in "${LOG_DIR}"/gcn_gin_gdump_42666914_*.log; do
  [[ -f "${f}" ]] && cp "${f}" "${STAGING}/logs/"
done

echo "Creating tarball..."
tar -czf "${BUNDLE}" -C "${STAGING}" .
rm -rf "${STAGING}"

echo ""
echo "Done."
ls -lh "${BUNDLE}"
echo ""
echo "On Mac (one scp after ssh fasrc once, or use pull script with multiplexing):"
echo "  scp fasrc:${BUNDLE} /tmp/"
echo "  mkdir -p results/gcn_gin_routing"
echo "  tar -xzf /tmp/gcn_gin_routing_bundle.tar.gz -C results/gcn_gin_routing/"
