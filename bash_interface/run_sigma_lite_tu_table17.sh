#!/usr/bin/env bash
# SiGMA-lite on a TU dataset with the Table 17 / 18 recipe (a2g4, L=12, H=64,
# random 50/25/25 split): every base_lr x seed, run in parallel, then aggregated
# into mean ± std test accuracy (best LR per family, as in Table 17).
#
# Runs on a laptop CPU (MUTAG: ~7 min per 1000-epoch run). Usage:
#   bash bash_interface/run_sigma_lite_tu_table17.sh                     # MUTAG, Table 17 (d_h=16)
#   D_H=4 bash bash_interface/run_sigma_lite_tu_table17.sh               # Table 18 (d_h=4)
#   GATE_ACT=gelu bash bash_interface/run_sigma_lite_tu_table17.sh       # Gated-GPS gate
#   DATASET=ENZYMES JOBS=3 bash bash_interface/run_sigma_lite_tu_table17.sh
#   DATASET=REDDIT-BINARY BATCH_SIZE=16 bash bash_interface/run_sigma_lite_tu_table17.sh  # paper batch sizes: 64, COLLAB 32, REDDIT-B 16
#   WANDB=True WANDB_ENTITY=<entity> bash bash_interface/run_sigma_lite_tu_table17.sh
#   MAX_EPOCH=20 SEEDS="0" LRS="0.001" bash bash_interface/run_sigma_lite_tu_table17.sh   # smoke
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

export DATASET="${DATASET:-MUTAG}"
export D_H="${D_H:-16}"
export GATE_ACT="${GATE_ACT:-sigmoid}"
export MAX_EPOCH="${MAX_EPOCH:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets}"
export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/sigma_lite_tu}"
export WANDB="${WANDB:-False}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LRS="${LRS:-0.001 0.01}"
JOBS="${JOBS:-5}"
# Threads per run; JOBS x OMP_NUM_THREADS should not exceed the number of cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local lr="$1" seed="$2"
  local tag="${DATASET}_dh${D_H}_${GATE_ACT}_lr${lr}_seed${seed}"
  local wandb_args=()
  if [[ -n "${WANDB_ENTITY}" ]]; then
    wandb_args=(wandb.entity "${WANDB_ENTITY}")
  fi
  echo "[start] ${tag}"
  if "${PYTHON}" main.py --cfg configs/sigma_lite/tu-sigma-lite-a2g4.yaml \
      dataset.name "${DATASET}" \
      dataset.dir "${DATASET_DIR}" \
      gnn.hybrid.d_h "${D_H}" \
      gnn.sigma_lite.gate_act "${GATE_ACT}" \
      optim.base_lr "${lr}" \
      optim.max_epoch "${MAX_EPOCH}" \
      train.batch_size "${BATCH_SIZE}" \
      seed "${seed}" \
      out_dir "${OUT_ROOT}/${tag}" \
      wandb.use "${WANDB}" \
      wandb.name "sigma_lite_${tag}" \
      wandb.group "sigma_lite_tu_${DATASET}_dh${D_H}_${GATE_ACT}" \
      ${wandb_args[@]+"${wandb_args[@]}"} \
      > "${OUT_ROOT}/${tag}.log" 2>&1; then
    echo "[done]  ${tag}"
  else
    echo "[FAIL]  ${tag} (see ${OUT_ROOT}/${tag}.log)"
  fi
}
export -f run_one

for lr in ${LRS}; do
  for seed in ${SEEDS}; do
    echo "${lr} ${seed}"
  done
done | xargs -P "${JOBS}" -n 2 bash -c 'run_one "$@"' _

"${PYTHON}" scripts/aggregate_sigma_lite_tu.py "${OUT_ROOT}"
