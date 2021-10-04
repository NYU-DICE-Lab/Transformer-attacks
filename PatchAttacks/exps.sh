sbatch eval_attacks.sh vit224 /scratch/aaj458/data/ImageNet/val 32 20 300 0.1 16 0
sleep 2
sbatch eval_attacks.sh wide-resnet /scratch/aaj458/data/ImageNet/val 32 20 300 0.1 16 0 
sleep 2
sbatch eval_attacks.sh deit224 /scratch/aaj458/data/ImageNet/val 32 20 300 0.1 16 0
sleep 2
sbatch eval_attacks.sh deit224_distill  /scratch/aaj458/data/ImageNet/val 32 20 300 0.1 16 0
sleep 2
sbatch eval_attacks.sh vit384  /scratch/aaj458/data/ImageNet/val 32 20 300  0.1 16  0
sbatch eval_attacks.sh resnet50  /scratch/aaj458/data/ImageNet/val 32 20 300  0.1 16  0
sbatch eval_attacks.sh resnet101d  /scratch/aaj458/data/ImageNet/val 32 20 300  0.1 16  0

