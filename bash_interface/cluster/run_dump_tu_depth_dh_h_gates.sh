#!/usr/bin/env bash
# =============================================================================
# Dump per-graph gates for TU L×d_h×H gated runs (best-LR cells).
#
# Default slice (MUTAG heatmap interest): L=1, H=8, gated only
#   Cells (comma-separated for sbatch --export):
#     1:lr01,2:lr01,4:lr001,16:lr01
#   × 5 seeds = 20 array tasks.
#
# Requires existing ckpts under:
#   $GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/<ds>_L{L}_dh{d}_H{H}_gated_<lr>_seed<s>/ckpt/
#
# Submit:
#   bash bash_interface/cluster/submit_dump_tu_depth_dh_h_gates.sh
# =============================================================================

#SBATCH --job-name=tu_LdhH_gdmp
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${TU_LDHH_GDUMP_NUM_SEEDS:-5}"
epoch="${GATE_DUMP_EPOCH:--1}"
level="${GATE_DUMP_LEVEL:-graph}"

ds_tag="${TU_LDHH_GDUMP_DS:-mutag}"
ds_name="${TU_LDHH_GDUMP_DS_NAME:-MUTAG}"
L="${TU_LDHH_GDUMP_L:-1}"
H="${TU_LDHH_GDUMP_H:-8}"

# Comma-separated "dh:lr_tag" pairs (spaces break sbatch --export).
# Default = MUTAG L1 H8 best gated LRs from heatmap.
IFS=',' read -r -a cells <<< "${TU_LDHH_GDUMP_CELLS:-1:lr01,2:lr01,4:lr001,16:lr01}"
num_cells=${#cells[@]}
num_tasks=$((num_cells * num_seeds))

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks})"
  exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
cell_idx=$((idx / num_seeds))
cell="${cells[$cell_idx]}"
d_h="${cell%%:*}"
lr_tag="${cell##*:}"

cfg="configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml"
run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_depth_dh_h/${ds_tag}_L${L}_dh${d_h}_H${H}_gated_${lr_tag}_seed${seed}"
out_pt="${run_dir}/gate_values_per_graph.pt"

if [ ! -d "${run_dir}/ckpt" ]; then
  log_message "ERROR: missing ckpt/ under ${run_dir}"
  exit 1
fi

if [ -f "${out_pt}" ] && [ "${GATE_DUMP_SKIP_EXISTING:-0}" = "1" ]; then
  log_message "skip existing ${out_pt}"
  exit 0
fi

log_message "gate-dump ${task_id}/${num_tasks}: ${ds_name} L=${L} d_h=${d_h} H=${H} gated ${lr_tag} seed=${seed}"
log_message "run_dir=${run_dir} → ${out_pt}"

extra=(
  seed "${seed}"
  dataset.name "${ds_name}"
  gnn.layers_mp "${L}"
  gnn.dim_inner "${H}"
  gnn.hybrid.d_h "${d_h}"
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
  extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

python -u scripts/gate_viz/dump_per_graph_gates.py \
  --run_dir "${run_dir}" \
  --epoch "${epoch}" \
  --level "${level}" \
  --out "${out_pt}" \
  --cfg "${cfg}" \
  "${extra[@]}"

log_message "wrote $(ls -lh "${out_pt}" | awk '{print $5, $9}')"
