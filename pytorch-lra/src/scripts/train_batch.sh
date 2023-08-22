#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -J lra_training
#SBATCH -A eecs
#SBATCH -p dgxs
##SBATCH -p ampere
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=120G

#SBATCH -o logs/softmax_cls_retrieval_train.log

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agostinv@oregonstate.edu

#module loads
module load python3/3.8
module load gcc/11.2
module load cuda/11.4
module load nccl/2.12.10

#activation of environment, moving to working directory, installation of necessary libraries for environment
cd /nfs/hpc/share/agostinv/pytorch_lra_implementation/src
source /nfs/hpc/share/agostinv/pytorch_lra_venv/bin/activate

python main.py \
    --mode train --attn softmax --task lra-retrieval --log-affix fresh-softmax-cls \
    --pooling-mode CLS \
    #--enable-learned-numer --learned-numer-inter-size 1 \
