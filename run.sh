# Local multi-GPU launcher. For Harvard cluster (mweber_gpu + W&B), see:
#   bash_interface/cluster/RUNNING.md
counter=$1

for gnn in gcn gine gatedgcn
do

export CUDA_VISIBLE_DEVICES=$((counter % 8))

python main.py --cfg configs/$gnn/$2.yaml --repeat $3 seed 0 

done
