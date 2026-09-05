#!/usr/bin/env bash
# Pack GIN depth-routing analysis artifacts into ONE tarball (run on cluster).
#
# Usage (on cluster):
#   bash bash_interface/cluster/pack_gin_depth_routing_for_pull.sh
#
# Then on Mac:
#   scp fasrc:/n/netscratch/.../gnnplus_pull/gin_depth_routing_bundle.tar.gz /tmp/
#   tar -xzf /tmp/gin_depth_routing_bundle.tar.gz -C results/gin_routing_depth/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
NET_ROOT="${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gin_routing_depth"
OUT_DIR="${GIN_DEPTH_PACK_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull}"
BUNDLE="${OUT_DIR}/gin_depth_routing_bundle.tar.gz"
STAGING="${OUT_DIR}/gin_depth_routing_staging"

echo "=== Pack GIN depth-routing bundle ==="
echo "  Repo:    ${REPO_ROOT}"
echo "  Net:     ${NET_ROOT}"
echo "  Output:  ${BUNDLE}"
echo ""

rm -rf "${STAGING}"
mkdir -p "${STAGING}/analysis" "${STAGING}/gates" "${STAGING}/logs" "${OUT_DIR}"

echo "[1/3] Copy analysis artifacts..."
if [[ -d "${REPO_ROOT}/results/gin_routing_depth/analysis" ]]; then
  cp -a "${REPO_ROOT}/results/gin_routing_depth/analysis/." "${STAGING}/analysis/"
else
  echo "  WARN: analysis dir missing" >&2
fi

echo "[2/3] Copy gate_graph_summary.csv (gated runs)..."
for lr_tag in lr001 lr01; do
  for seed in 0 1 2 3 4; do
    run_name="l2_a0g1_gated_${lr_tag}_seed${seed}"
    src="${NET_ROOT}/toy/${run_name}/gate_graph_summary.csv"
    if [[ -f "${src}" ]]; then
      dest_dir="${STAGING}/gates/toy/${run_name}"
      mkdir -p "${dest_dir}"
      cp "${src}" "${dest_dir}/gate_graph_summary.csv"
    else
      echo "  WARN missing: ${src}" >&2
    fi
  done
done

echo "[3/3] Copy recent SLURM logs..."
LOG_DIR="${REPO_ROOT}/logs_gnnplus"
shopt -s nullglob
for f in \
  "${LOG_DIR}"/gin_depth_analyze_*.log \
  "${LOG_DIR}"/gin_depth_opp_*.log \
  "${LOG_DIR}"/gin_depth_lmask_*.log \
  "${LOG_DIR}"/gin_depth_gdump_*.log
do
  cp "${f}" "${STAGING}/logs/" || true
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
echo "  mkdir -p results/gin_routing_depth"
echo "  tar -xzf /tmp/gin_depth_routing_bundle.tar.gz -C results/gin_routing_depth/"
