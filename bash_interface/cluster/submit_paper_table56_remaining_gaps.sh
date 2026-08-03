#!/usr/bin/env bash
# One-shot launcher for remaining Table 5/6 gaps (PATTERN gritvn4 + retries).
#
# Submits five arrays:
#   1) T5 gap-fill: CIFAR MP_only + COCO ungated_attn seeds 1–4
#   2) T5 PATTERN grit+VN4 architecture ablations
#   3) T6 PATTERN grit+VN4 homog/hetero MP
#   4) T5 CLUSTER ht9bntg2 architecture ablations (78.956% SiGMA)
#   5) T6 CLUSTER +1 MP homog/hetero
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus && git pull
#
#   bash bash_interface/cluster/submit_paper_table56_remaining_gaps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== 1/5 Table 5 gap-fill (CIFAR MP_only + COCO ungated_attn) ==="
bash bash_interface/cluster/submit_paper_table5_gap_fill.sh

echo "=== 2/5 Table 5 PATTERN grit+VN4 architecture ablations ==="
bash bash_interface/cluster/submit_paper_table5_pattern_gritvn4_ablations.sh

echo "=== 3/5 Table 6 PATTERN grit+VN4 homog/hetero ==="
bash bash_interface/cluster/submit_paper_table6_pattern_gritvn4.sh

echo "=== 4/5 Table 5 CLUSTER ht9bntg2 architecture ablations ==="
bash bash_interface/cluster/submit_paper_table5_cluster_ablations.sh

echo "=== 5/5 Table 6 CLUSTER +1 MP homog/hetero ==="
bash bash_interface/cluster/submit_paper_table6_cluster_1mp.sh

cat <<'EOF'

=== Already finished (no relaunch) ===
  Table 5 CIFAR SiGMA_ungated_attn : 79.754±0.339%  (paper_T5_cifar10_SiGMA_ungated_attn)
  Table 6 CIFAR Hetero_MP          : 79.262±0.405%  (paper_T6_cifar10_Hetero_MP)
  Table 6 CIFAR Hetero_MP_ungated  : 79.560±0.659%
  PATTERN SiGMA grit+VN4           : 87.395±0.194%  (paper_sigma_grit_attn_pattern_vn4)
  CLUSTER SiGMA (vanilla)          : 78.956±0.112%  (paper_bestmodel_v1_cluster_ht9bntg2)

Paste all JOBIDs into CLUSTER_LAUNCHES.md

EOF
