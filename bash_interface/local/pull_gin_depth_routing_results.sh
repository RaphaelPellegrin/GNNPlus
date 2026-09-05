#!/usr/bin/env bash
# Pull GIN depth-routing cluster artifacts — ONE SSH login (multiplexing).
#
# Usage (from repo root):
#   bash bash_interface/local/pull_gin_depth_routing_results.sh
#   bash bash_interface/local/pull_gin_depth_routing_results.sh --bundle
#
set -euo pipefail

HOST="${GIN_DEPTH_SCP_HOST:-fasrc}"
REPO_REMOTE="${GIN_DEPTH_REPO_REMOTE:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
NET_REMOTE="${GIN_DEPTH_NET_REMOTE:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/gin_routing_depth}"
PACK_REMOTE="${GIN_DEPTH_PACK_REMOTE:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull/gin_depth_routing_bundle.tar.gz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${GIN_DEPTH_LOCAL_DEST:-${REPO_ROOT}/results/gin_routing_depth}"

ANALYSIS_REMOTE="${REPO_REMOTE}/results/gin_routing_depth/analysis"

SOCKET_DIR="${HOME}/.ssh/sockets"
CONTROL_PATH="${SOCKET_DIR}/rpellegrinext@holylogin.rc.fas.harvard.edu-22"

DRY_RUN=0
USE_BUNDLE=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --bundle) USE_BUNDLE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg} (try --help)" >&2
      exit 1
      ;;
  esac
done

ssh_cmd() {
  ssh -o ControlPath="${CONTROL_PATH}" "$@"
}

scp_cmd() {
  scp -o ControlPath="${CONTROL_PATH}" "$@"
}

rsync_cmd() {
  rsync -avz -e "ssh -o ControlPath=${CONTROL_PATH}" "$@"
}

ensure_master() {
  mkdir -p "${SOCKET_DIR}"
  if ssh -O check -o ControlPath="${CONTROL_PATH}" "${HOST}" 2>/dev/null; then
    echo "Reusing existing SSH master to ${HOST}"
    return 0
  fi
  echo ""
  echo ">>> Opening SSH master to ${HOST} — authenticate ONCE (password + 2FA)"
  echo ""
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ssh -fN ${HOST}"
    return 0
  fi
  ssh -o ControlMaster=yes -o ControlPath="${CONTROL_PATH}" -fN "${HOST}"
}

PACK_SCRIPT_LOCAL="${SCRIPT_DIR}/../cluster/pack_gin_depth_routing_for_pull.sh"

remote_pack_bundle() {
  local remote_pack="${REPO_REMOTE}/bash_interface/cluster/pack_gin_depth_routing_for_pull.sh"
  if ssh_cmd "${HOST}" "test -f '${remote_pack}'"; then
    echo "Using cluster pack script: ${remote_pack}"
    ssh_cmd "${HOST}" "REPO_ROOT='${REPO_REMOTE}' bash '${remote_pack}'"
  elif [[ -f "${PACK_SCRIPT_LOCAL}" ]]; then
    echo "Cluster pack script not found — streaming local copy over SSH..."
    ssh_cmd "${HOST}" "REPO_ROOT='${REPO_REMOTE}' bash -s" < "${PACK_SCRIPT_LOCAL}"
  else
    echo "ERROR: pack script missing locally and on cluster." >&2
    exit 1
  fi
}

run_pull_bundle() {
  echo "=== Bundle mode (one file) ==="
  ensure_master
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ssh pack + scp bundle"
    return 0
  fi
  echo "Building tarball on cluster..."
  remote_pack_bundle
  mkdir -p "${DEST}"
  local tmp_bundle
  tmp_bundle="$(mktemp /tmp/gin_depth_routing_bundle.XXXXXX.tar.gz)"
  echo "Downloading bundle..."
  scp_cmd "${HOST}:${PACK_REMOTE}" "${tmp_bundle}"
  tar -xzf "${tmp_bundle}" -C "${DEST}"
  rm -f "${tmp_bundle}"
}

run_pull_rsync() {
  ensure_master
  mkdir -p "${DEST}/analysis" "${DEST}/gates"

  echo "[1/2] Analysis..."
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    rsync_cmd "${HOST}:${ANALYSIS_REMOTE}/" "${DEST}/analysis/"
  fi

  echo "[2/2] gate_graph_summary.csv..."
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    rsync_cmd \
      --include='toy/***' \
      --include='*/l2_a0g1_gated_*/' \
      --include='*/l2_a0g1_gated_*/gate_graph_summary.csv' \
      --exclude='*' \
      "${HOST}:${NET_REMOTE}/" "${DEST}/gates/"
  fi
}

print_status() {
  echo ""
  echo "=== Local status ==="
  if [[ -d "${DEST}/analysis" ]]; then
    ls -lh "${DEST}/analysis/" 2>/dev/null | tail -n +2 || true
  fi
  echo ""
  echo "Next (local plots from dumped gates):"
  echo "  python scripts/synthetic/plot_gin_depth_routing_ranked_gates.py \\"
  echo "    --results-root results/gin_routing_depth \\"
  echo "    --out-dir results/gin_routing_depth/analysis/ranked_gates"
}

echo "=== GIN depth-routing pull (single SSH session) ==="
echo "  Host:  ${HOST}"
echo "  Dest:  ${DEST}"
echo ""

if [[ "${USE_BUNDLE}" -eq 1 ]]; then
  run_pull_bundle
else
  run_pull_rsync
fi

print_status

echo ""
echo "SSH master left open. Close with:"
echo "  ssh -O exit -o ControlPath=${CONTROL_PATH} ${HOST}"
