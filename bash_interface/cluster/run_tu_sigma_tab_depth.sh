#!/usr/bin/env bash
# =============================================================================
# TU Tables 17/18 — SiGMA hetero gated vs ungated at shallow depths L ∈ {4,2,1}.
#
# Same recipe as Tab.17 (d_h=16) / Tab.18 (d_h=4), H=64, a2g4 hetero, but
# layers_mp ∈ {4, 2, 1} instead of L=12. Report better of {1e-3, 1e-2}.
#
# Depths (outer):
#   0  L=4
#   1  L=2
#   2  L=1
#
# Tables:
#   0  Tab.17  d_h=16  configs/.../sigma-hetero-a2g4-anchor.yaml
#   1  Tab.18  d_h=4   configs/.../sigma-hetero-a2g4-matched-anchor.yaml
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
# Variants (per L × table × dataset):
#   0  SiGMA_hetero gated   lr=0.001
#   1  SiGMA_hetero gated   lr=0.01
#   2  SiGMA_ungated        lr=0.001   (gate=none)
#   3  SiGMA_ungated        lr=0.01
#
# Layout: 3 L × 2 tables × 6 ds × 4 variants × 5 seeds = 720
#   task_id = ((((L_idx * NUM_TABLES + table_idx) * NUM_DATASETS + ds_idx)
#               * NUM_VARIANTS + var_idx) * NUM_SEEDS) + seed + 1
#
# Blocks (240 tasks each):
#   L=4 → 1–240 · L=2 → 241–480 · L=1 → 481–720
# Within each L: Tab.17 (1–120 of block) then Tab.18 (121–240 of block).
#
# W&B:
#   Tab.17 → tu_L{k}_hh_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}
#   Tab.18 → tu_L{k}_1x_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_tab_depth.sh
# =============================================================================

#SBATCH --job-name=tu_tab_L
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
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
num_seeds="${TU_TAB_L_NUM_SEEDS:-5}"
num_variants="${TU_TAB_L_NUM_VARIANTS:-4}"
num_datasets="${TU_TAB_L_NUM_DATASETS:-6}"
num_tables="${TU_TAB_L_NUM_TABLES:-2}"
num_depths="${TU_TAB_L_NUM_DEPTHS:-3}"
num_tasks="${TU_TAB_L_NUM_TASKS:-$((num_depths * num_tables * num_datasets * num_variants * num_seeds))}"
do_gate_dump="${TU_TAB_L_GATE_DUMP:-1}"

# Depth list (override with e.g. TU_TAB_L_LS="4 2 1")
# shellcheck disable=SC2206
layers_list=(${TU_TAB_L_LS:-4 2 1})
if [ "${#layers_list[@]}" -ne "${num_depths}" ]; then
    log_message "TU_TAB_L_LS has ${#layers_list[@]} entries but NUM_DEPTHS=${num_depths}"
    exit 1
fi

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
declare -A batch_for=(
    [mutag]="${TU_TAB_L_BATCH_DEFAULT:-64}"
    [enzymes]="${TU_TAB_L_BATCH_DEFAULT:-64}"
    [proteins]="${TU_TAB_L_BATCH_DEFAULT:-64}"
    [collab]="${TU_TAB_L_BATCH_COLLAB:-32}"
    [imdb_binary]="${TU_TAB_L_BATCH_IMDB:-64}"
    [reddit_binary]="${TU_TAB_L_BATCH_REDDIT:-16}"
)

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
rest=$((rest / num_variants))
dataset_idx=$((rest % num_datasets))
rest=$((rest / num_datasets))
table_idx=$((rest % num_tables))
L_idx=$((rest / num_tables))

if [ "${L_idx}" -lt 0 ] || [ "${L_idx}" -ge "${num_depths}" ]; then
    log_message "bad L_idx=${L_idx}"
    exit 1
fi

L="${layers_list[$L_idx]}"
ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
batch_size="${batch_for[$ds_tag]}"

cfg_dir="configs/tu_sigma_homo_hetero"
case "${table_idx}" in
    0)
        table_tag="t17"
        wandb_prefix="tu_L${L}_hh"
        out_subdir="tu_sigma_tab_depth"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        arch_tag="a2g4_dh16"
        dh_tag="dh16"
        dh=16
        ;;
    1)
        table_tag="t18"
        wandb_prefix="tu_L${L}_1x"
        out_subdir="tu_sigma_tab_depth"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        arch_tag="a2g4_dh4"
        dh_tag="dh4"
        dh=4
        ;;
    *)
        log_message "bad table_idx=${table_idx}"
        exit 1
        ;;
esac

gate_override=()
case "${variant_idx}" in
    0)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        base_lr="0.001"
        lr_tag="lr001"
        ;;
    1)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        base_lr="0.01"
        lr_tag="lr01"
        ;;
    2)
        family="SiGMA_ungated"
        variant="SiGMA_ungated"
        base_lr="0.001"
        lr_tag="lr001"
        gate_override+=(gnn.hybrid.gate none)
        do_gate_dump=0
        ;;
    3)
        family="SiGMA_ungated"
        variant="SiGMA_ungated"
        base_lr="0.01"
        lr_tag="lr01"
        gate_override+=(gnn.hybrid.gate none)
        do_gate_dump=0
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="${wandb_prefix}_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_sigma_tab_depth,L${L},${table_tag},${ds_tag},${variant},${lr_tag},seed${seed},${arch_tag},${dh_tag}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/${out_subdir}/${ds_tag}_L${L}_${variant}_${dh_tag}_${lr_tag}_seed${seed}"
else
    run_dir="results/${out_subdir}/${ds_tag}_L${L}_${variant}_${dh_tag}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU tab-depth ${task_id}/${num_tasks}: L=${L} ${table_tag} ds=${ds_name} ${variant} lr=${base_lr} batch=${batch_size} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=${family}
variant=${variant}
table=${table_tag}
L=${L}
d_h=${dh}
dh_tag=${dh_tag}
lr=${base_lr}
lr_tag=${lr_tag}
batch_size=${batch_size}
seed=${seed}
cfg=${cfg}
arch_tag=${arch_tag}
gate_override=${gate_override[*]:-headwise}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
wandb_name=${wandb_name}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    optim.base_lr "${base_lr}"
    train.batch_size "${batch_size}"
    gnn.layers_mp "${L}"
    out_dir "${run_dir}"
    train.enable_ckpt True
    train.ckpt_best True
    train.ckpt_clean True
    "${gate_override[@]}"
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

export WANDB_EXTRA_TAGS="${wandb_tags}"

python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    "${extra_args[@]}"

log_message "Training finished. ckpt listing:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/ under ${run_dir}"

if [ "${family}" = "SiGMA_hetero" ] && [ "${do_gate_dump}" = "1" ]; then
    if [ ! -d "${run_dir}/ckpt" ]; then
        log_message "ERROR: expected ckpt/ for gate dump but missing"
        exit 1
    fi
    out_pt="${run_dir}/gate_values_per_graph.pt"
    dump_extra=()
    if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
        dump_extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
    fi
    log_message "Dumping per-graph gates → ${out_pt}"
    python scripts/gate_viz/dump_per_graph_gates.py \
        --run_dir "${run_dir}" \
        --epoch -1 \
        --out "${out_pt}" \
        --cfg "${cfg}" \
        seed "${seed}" \
        dataset.name "${ds_name}" \
        gnn.layers_mp "${L}" \
        "${dump_extra[@]}"
else
    log_message "Skipping gate dump (family=${family}, dump=${do_gate_dump})"
fi

log_message "Task ${task_id} complete."
