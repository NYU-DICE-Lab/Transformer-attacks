#!/bin/sh
#SBATCH --job-name=vit_eval
#SBATCH --mem=32GB
#SBATCH --output=$2-out.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

module load python/intel/3.8.6
module load cuda/11.0.194

source ~/Projects/venv/bin/activate
python patch_attack_grad.py -o $1 -mt $2 --gpu 0 -dpath $3  -it $4 -mp $5 -ni $6 -clip -lr $7
