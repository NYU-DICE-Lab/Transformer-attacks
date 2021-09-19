#!/bin/sh
#SBATCH --job-name=vit_eval
#SBATCH --mem=32GB
#SBATCH --output=out_%A_%j.log
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --time=47:59:00

module load python/intel/3.8.6
module load cuda/10.2.89

source /scratch/aaj458/venv/bin/activate;
python patch_attack_grad.py -o /scratch/aaj458/transformer_results_8new/output_mt_$1_it_$3_mp_$4_ni_$5_lr_$6_ps_$7_$9 -mt $1 --gpu 0 -dpath $2  -it $3 -mp $4 -ni $5 -clip -lr $6 -ps $7 -si $8 -eps $9
