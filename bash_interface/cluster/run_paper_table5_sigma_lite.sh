#!/usr/bin/env bash
# =============================================================================
# Table 5 — SiGMA-lite ablation (gate after concat / out_proj).
#
# Same hyperparameter / head recipes as the Table 5 SiGMA row, but
# model.type=sigma_lite (ungated MMA heads + one elementwise sigmoid gate on
# the mixed update). No other HPs are tuned.
#
# Datasets × seeds (match Table 5 SiGMA cohorts):
#   0 peptides_func   seeds 0–4   Homog_MP a1g2 (o5cdk766 + GCN×2)
#   1 peptides_struct seeds 0–4   g3bsaq32-b7m0
#   2 voc             seeds 0–4   j7ukyzdm a2g2
#   3 mnist           seeds 0–4   lcvbyyss a2g2
#   4 cifar10         seeds 0–4   ulij45a2 a8g4
#   5 pattern         seeds 5–9   ta9qtxb9 GRIT+VN4
#   6 cluster         seeds 0–4   ht9bntg2 a1g1
#
# Layout: 7 ds × 5 seeds = 35
#   task_id = ds_idx * 5 + seed_i + 1
#   seed    = seed_offset[ds] + seed_i
#
# W&B groups (project GNNPlus):
#   paper_T5_<ds>_SiGMA_lite
#   pattern → paper_T5_pattern_gritvn4_SiGMA_lite
#   peptides_func → paper_T5_peptides_func_SiGMA_lite
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_sigma_lite.sh
# =============================================================================

#SBATCH --job-name=t5_sigma_lite
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
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

# Ensure SiGMA-lite is importable from this checkout.
python -c "import GNNPlus.network.sigma_lite_gnn" || {
    log_message "ERROR: cannot import GNNPlus.network.sigma_lite_gnn from ${REPO_ROOT}"
    exit 1
}

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${PAPER_T5_LITE_NUM_SEEDS:-5}"
num_datasets=7
num_tasks="${PAPER_T5_LITE_NUM_TASKS:-$((num_datasets * num_seeds))}"
gate_act="${PAPER_T5_LITE_GATE_ACT:-sigmoid}"

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed_i=$((idx % num_seeds))
dataset_idx=$((idx / num_seeds))

# Match Table 5 SiGMA seed cohorts (PATTERN uses 5–9).
seed_offsets=(0 0 0 0 0 5 0)
seed_offset="${seed_offsets[$dataset_idx]}"
seed=$((seed_offset + seed_i))

extra_args=()
variant="SiGMA_lite"

case "${dataset_idx}" in
    0)
        ds_tag="peptides_func"
        cfg="configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml"
        source_run="n3beo5px"
        wandb_group="paper_T5_peptides_func_SiGMA_lite"
        extra_args+=(
            gnn.hybrid.num_attn_heads 1
            gnn.hybrid.num_gnn_heads 2
            gnn.hybrid.gnn_types "GCN,GCN"
        )
        ;;
    1)
        ds_tag="peptides_struct"
        cfg="configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
        source_run="joskj6gk"
        wandb_group="paper_T5_peptides_struct_SiGMA_lite"
        ;;
    2)
        ds_tag="voc"
        cfg="configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml"
        source_run="zhfigzdi"
        wandb_group="paper_T5_voc_SiGMA_lite"
        ;;
    3)
        ds_tag="mnist"
        cfg="configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml"
        source_run="uh7nxm4e"
        wandb_group="paper_T5_mnist_SiGMA_lite"
        ;;
    4)
        ds_tag="cifar10"
        cfg="configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml"
        source_run="skdtqk7t"
        wandb_group="paper_T5_cifar10_SiGMA_lite"
        ;;
    5)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
        source_run="w8u40h6j"
        wandb_group="paper_T5_pattern_gritvn4_SiGMA_lite"
        extra_args+=(
            gnn.hybrid.attn_type grit
            dataset.add_virtual_nodes True
            dataset.num_virtual_nodes 4
        )
        ;;
    6)
        ds_tag="cluster"
        cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml"
        source_run="qdkezojh"
        wandb_group="paper_T5_cluster_SiGMA_lite"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,SiGMA_lite,${ds_tag},seed${seed},gate_${gate_act},source_${source_run}"

log_message "T5 SiGMA-lite ${task_id}/${num_tasks}: ds=${ds_tag} seed=${seed} gate=${gate_act} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
fi

export WANDB_EXTRA_TAGS="${wandb_tags}"

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    model.type sigma_lite \
    gnn.sigma_lite.gate_act "${gate_act}" \
    gnn.hybrid.log_gate_stats True \
    gnn.hybrid.identity_proj False \
    "${extra_args[@]}"
