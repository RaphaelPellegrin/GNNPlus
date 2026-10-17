#!/usr/bin/env bash
# =============================================================================
# Table 5 — SiGMA vs MMA (ungated), 15 *new* seeds per dataset.
#
# MMA = same hybrid recipe as SiGMA with gating removed:
#   gnn.hybrid.gate none
# (empty mp_gate inherits gate → MP heads also ungated; see hybrid_gnn.py).
#
# Datasets (paper Table 5 columns) and seed blocks (avoid prior cohorts):
#   0 peptides_func   seeds 5–19   Homog_MP a1g2 (o5cdk766 + GCN×2)
#   1 peptides_struct seeds 5–19   g3bsaq32-b7m0
#   2 voc             seeds 5–19   j7ukyzdm a2g2
#   3 mnist           seeds 5–19   lcvbyyss a2g2
#   4 cifar10         seeds 5–19   ulij45a2 a8g4
#   5 pattern         seeds 25–39  ta9qtxb9 GRIT+VN4 (10–24 already filled)
#   6 cluster         seeds 20–34  ht9bntg2 (5–19 already filled)
#
# Variants:
#   0  SiGMA
#   1  MMA  (= SiGMA_ungated; W&B group keeps SiGMA_ungated for continuity)
#
# Layout: 7 ds × 2 variants × 15 seeds = 210
#   task_id = ((ds_idx * 2 + var_idx) * 15) + seed_i + 1
#   seed    = seed_offset[ds] + seed_i
#
# W&B (project GNNPlus):
#   peptides_func SiGMA → paper_T6_peptides_func_Homog_MP  (matches Table 5 export)
#   other SiGMA        → paper_T5_<ds>_SiGMA
#   pattern SiGMA      → paper_T5_pattern_gritvn4_SiGMA
#   MMA                → paper_T5_<ds>_SiGMA_ungated
#                        (pattern: paper_T5_pattern_gritvn4_SiGMA_ungated)
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_sigma_mma_seed15.sh
# =============================================================================

#SBATCH --job-name=t5_sigma_mma_s15
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

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${PAPER_T5_S15_NUM_SEEDS:-15}"
num_variants=2
num_datasets=7
num_tasks="${PAPER_T5_S15_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed_i=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
dataset_idx=$((rest / num_variants))

# Per-dataset seed offsets (new seeds beyond Table 5 / prior fills).
seed_offsets=(5 5 5 5 5 25 20)
seed_offset="${seed_offsets[$dataset_idx]}"
seed=$((seed_offset + seed_i))

extra_args=()
wandb_group_prefix="paper_T5"

case "${dataset_idx}" in
    0)
        ds_tag="peptides_func"
        # Table 5 SiGMA cell = Homog_MP a1g2 (not bare a1g1).
        cfg="configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml"
        source_run="n3beo5px"
        sigma_group="paper_T6_peptides_func_Homog_MP"
        mma_group="paper_T5_peptides_func_SiGMA_ungated"
        # Match Reproduce args of n3beo5px; MMA must keep the same heads.
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
        sigma_group="paper_T5_peptides_struct_SiGMA"
        mma_group="paper_T5_peptides_struct_SiGMA_ungated"
        # SiGMA baseline group used for the table is bestmodel; new gated seeds
        # go to paper_T5_*_SiGMA so they are distinct but same cfg.
        ;;
    2)
        ds_tag="voc"
        cfg="configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml"
        source_run="zhfigzdi"
        sigma_group="paper_T5_voc_SiGMA"
        mma_group="paper_T5_voc_SiGMA_ungated"
        ;;
    3)
        ds_tag="mnist"
        cfg="configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml"
        source_run="uh7nxm4e"
        sigma_group="paper_T5_mnist_SiGMA"
        mma_group="paper_T5_mnist_SiGMA_ungated"
        ;;
    4)
        ds_tag="cifar10"
        cfg="configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml"
        source_run="skdtqk7t"
        sigma_group="paper_T5_cifar10_SiGMA"
        mma_group="paper_T5_cifar10_SiGMA_ungated"
        ;;
    5)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
        source_run="w8u40h6j"
        sigma_group="paper_T5_pattern_gritvn4_SiGMA"
        mma_group="paper_T5_pattern_gritvn4_SiGMA_ungated"
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
        sigma_group="paper_T5_cluster_SiGMA"
        mma_group="paper_T5_cluster_SiGMA_ungated"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

case "${variant_idx}" in
    0)
        variant="SiGMA"
        wandb_group="${sigma_group}"
        # Gated: no gate override (anchor / Homog_MP args already set).
        ;;
    1)
        variant="MMA"
        wandb_group="${mma_group}"
        # Same architecture as SiGMA; remove gates only.
        extra_args+=(gnn.hybrid.gate none)
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
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,sigma_vs_mma_seed15,${variant},${ds_tag},seed${seed},source_${source_run}"
if [ "${variant}" = "MMA" ]; then
    wandb_tags="${wandb_tags},SiGMA_ungated,MMA"
fi

log_message "T5 SiGMA/MMA s15 ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} seed=${seed} cfg=${cfg}"
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
    model.type hybrid_gnn \
    gnn.hybrid.log_gate_stats True \
    gnn.hybrid.identity_proj False \
    "${extra_args[@]}"
