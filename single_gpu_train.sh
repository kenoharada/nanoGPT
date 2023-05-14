#!/bin/bash
#PBS -q SQUID
#PBS -l elapstim_req=1:00:00
#PBS -b 1
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

nvidia-smi
singularity exec --bind $PATH_TO_EXP:$PATH_TO_EXP --pwd $PATH_TO_EXP --nv $PATH_TO_EXP/gpt.sif torchrun --standalone --nproc_per_node=gpu train.py config/train_gpt2.py --wandb_log=True
