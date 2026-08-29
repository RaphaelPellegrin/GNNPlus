#!/usr/bin/env bash
# Pull GCN/GIN routing cluster artifacts — ONE SSH login (multiplexing).
#
# The old version ran 30+ separate scp calls → password+2FA hell.
# This version opens a master SSH connection once, then reuses it.
#
# Prereq: SSH alias ``fasrc`` in ~/.ssh/config.
#
# Usage (from repo root):
#   bash bash_interface/local/pull_gcn_gin_routing_results.sh
#   bash bash_interface/local/pull_gcn_gin_routing_results.sh --bundle   # one tarball (best)
#   bash bash_interface/local/pull_gcn_gin_routing_results.sh --full     # + all .pt (~2GB)
#
# Alternative (zero local script): ssh fasrc once, run pack on cluster, one scp:
#   bash bash_interface/cluster/pack_gcn_gin_routing_for_pull.sh
#   scp fasrc:/n/netscratch/.../gcn_gin_routing_bundle.tar.gz /tmp/
#   tar -xzf /tmp/gcn_gin_routing_bundle.tar.gz -C results/gcn_gin_routing/
#
set -euo pipefail

HOST="${GCN_GIN_SCP_HOST:-fasrc}"
REPO_REMOTE="${GCN_GIN_REPO_REMOTE:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}"
NET_REMOTE="${GCN_GIN_NET_REMOTE:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/gcn_gin_routing}"
PACK_REMOTE="${GCN_GIN_PACK_REMOTE:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_pull/gcn_gin_routing_bundle.tar.gz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${GCN_GIN_LOCAL_DEST:-${REPO_ROOT}/results/gcn_gin_routing}"

ANALYSIS_REMOTE="${REPO_REMOTE}/results/gcn_gin_routing/analysis"
LOGS_REMOTE="${REPO_REMOTE}/logs_gnnplus"

SOCKET_DIR="${HOME}/.ssh/sockets"
CONTROL_PATH="${SOCKET_DIR}/rpellegrinext@holylogin.rc.fas.harvard.edu-22"

DRY_RUN=0
PULL_NODE_PT=0
USE_BUNDLE=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --full) PULL_NODE_PT=1 ;;
    --bundle) USE_BUNDLE=1 ;;
    -h|--help)
      sed -n '2,25p' "$0"
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
  echo ">>> All following transfers reuse this connection."
  echo ""
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ssh -fN ${HOST}"
    return 0
  fi
  ssh -o ControlMaster=yes -o ControlPath="${CONTROL_PATH}" -fN "${HOST}"
}

PACK_SCRIPT_LOCAL="${SCRIPT_DIR}/../cluster/pack_gcn_gin_routing_for_pull.sh"

remote_pack_bundle() {
  local remote_pack="${REPO_REMOTE}/bash_interface/cluster/pack_gcn_gin_routing_for_pull.sh"
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
  tmp_bundle="$(mktemp /tmp/gcn_gin_routing_bundle.XXXXXX.tar.gz)"
  echo "Downloading bundle..."
  scp_cmd "${HOST}:${PACK_REMOTE}" "${tmp_bundle}"
  tar -xzf "${tmp_bundle}" -C "${DEST}"
  rm -f "${tmp_bundle}"
}

run_pull_rsync() {
  ensure_master
  mkdir -p "${DEST}/analysis" "${DEST}/gates" "${DEST}/logs"

  echo "[1/3] Analysis (one rsync)..."
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] rsync analysis/"
  else
    rsync_cmd "${HOST}:${ANALYSIS_REMOTE}/" "${DEST}/analysis/"
  fi

  echo "[2/3] gate_graph_summary.csv (one rsync with includes)..."
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] rsync gates/"
  else
    # Pull only gate_graph_summary.csv under gated run dirs; preserve paths.
    rsync_cmd \
      --include='toy/***' \
      --include='sigma/***' \
      --include='*/a0g2_gated_*/' \
      --include='*/a0g2_gated_*/gate_graph_summary.csv' \
      --exclude='*' \
      "${HOST}:${NET_REMOTE}/" "${DEST}/gates/"
  fi

  if [[ "${PULL_NODE_PT}" -eq 1 ]]; then
    echo "[2b] gate_values_per_node.pt (large, one rsync)..."
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      rsync_cmd \
        --include='toy/***' \
        --include='sigma/***' \
        --include='*/a0g2_gated_*/' \
        --include='*/a0g2_gated_*/gate_values_per_node.pt' \
        --exclude='*' \
        "${HOST}:${NET_REMOTE}/" "${DEST}/gates/"
    fi
  fi

  echo "[3/3] SLURM logs..."
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    rsync_cmd \
      --include='gcn_gin_analyze_42666091.log' \
      --include='gcn_gin_gdump_42666914_*.log' \
      --exclude='*' \
      "${HOST}:${REPO_REMOTE}/logs_gnnplus/" "${DEST}/logs/" || true
  fi
}

print_status() {
  echo ""
  echo "=== Local status ==="
  if [[ -d "${DEST}/analysis" ]]; then
    echo "Analysis:"
    ls -lh "${DEST}/analysis/" 2>/dev/null | tail -n +2 || true
    wc -l "${DEST}/analysis/per_run_metrics.csv" 2>/dev/null || true
  fi
  if [[ -d "${DEST}/gates" ]]; then
    find "${DEST}/gates" -name 'gate_graph_summary.csv' 2>/dev/null | wc -l \
      | xargs echo "gate_graph_summary.csv count:"
  fi
  echo ""
  echo "You're DONE for paper figures if analysis/ has CSVs + fig_*.png."
  echo "gate_graph_summary (20 files) is optional deep-dive only."
  echo ""
  echo "  python scripts/synthetic/plot_gcn_gin_routing_paper_figures.py \\"
  echo "    --analysis-dir results/gcn_gin_routing/analysis"
}

echo "=== GCN/GIN routing pull (single SSH session) ==="
echo "  Host:  ${HOST}"
echo "  Dest:  ${DEST}"
echo ""

if [[ "${USE_BUNDLE}" -eq 1 ]]; then
  run_pull_bundle
else
  run_pull_rsync
fi

print_status

# Leave master open for reuse (ControlPersist 8h in ssh config).
echo ""
echo "SSH master left open (~8h). Close with:"
echo "  ssh -O exit -o ControlPath=${CONTROL_PATH} ${HOST}"
