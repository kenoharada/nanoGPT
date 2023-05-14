#!/bin/bash
#PBS -q SQUID
#PBS -l elapstim_req=1:00:00
#PBS -b 2
#PBS -l gpunum_job=8

# debug用
export NCCL_DEBUG=INFO
cat $PBS_NODEFILE
# wandb用
export http_proxy="http://ibgw1f-ib0:3128"
export https_proxy="https://ibgw1f-ib0:3128"
# singularity用
export PATH_TO_EXP=/sqfs/work/$GROUP_ID/$USER/nanoGPT
newgrp $GROUP_ID

# MASTER_ADDR, NODE_RANK, MASTER_PORTの設定
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export NODE_RANK=$(awk -v hostname="$(hostname)" '$0 ~ hostname { print NR-1; exit }' $PBS_NODEFILE)

export MASTER_PORT=29400 # 任意のポート番号を設定
export NNODES=$(cat $PBS_NODEFILE | wc -l)
(echo >/dev/tcp/$MASTER_ADDR/$MASTER_PORT) &>/dev/null
# If the exit code ($?) is 1, the port is free
if [ $? -eq 1 ]; then
    echo "Port $MASTER_PORT is available"
fi

echo 'NNODES'
echo $NNODES
echo 'MASTER_ADDR'
echo $MASTER_ADDR
echo 'NODE_RANK'
echo $NODE_RANK
echo 'WORLD_SIZE'
echo $WORLD_SIZE
echo 'MASTER_PORT'
echo $MASTER_PORT
nvidia-smi

DISTRIBUTED_ARGS="--nproc_per_node=gpu --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"


# singularity exec --bind $PATH_TO_EXP:$PATH_TO_EXP --pwd $PATH_TO_EXP --nv $PATH_TO_EXP/gpt.sif torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --wandb_log=True
singularity exec --bind $PATH_TO_EXP:$PATH_TO_EXP --pwd $PATH_TO_EXP --nv $PATH_TO_EXP/gpt.sif torchrun $DISTRIBUTED_ARGS train.py config/train_shakespeare_char.py --wandb_log=True --gradient_accumulation_steps=40
