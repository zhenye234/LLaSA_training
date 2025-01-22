#!/bin/bash
#SBATCH --job-name=tts            
#SBATCH --nodes=5                
#SBATCH --gpus-per-node=8          
#SBATCH --ntasks=5               
#SBATCH --cpus-per-task=224         
#SBATCH --mem=1000G
#SBATCH --time=168:00:00
#SBATCH --partition=xxx
#SBATCH --exclusive
 
export LOGLEVEL=INFO
 

 
export MASTER_PORT=29503

 
 
export MASTER_ADDR=dgx-069
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

 
export NCCL_DEBUG=INFO 
 
 
srun  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_tts.py config.json  
