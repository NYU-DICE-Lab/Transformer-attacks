sbatch eval_attacks.sh vit224 /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 16 0
sleep 10
sbatch eval_attacks.sh bit_152_4 /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 16 38

sleep 10
#sbatch eval_attacks.sh wide-resnet /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 16 557 
#sleep 10
#sbatch eval_attacks.sh deit224 /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 16  
#sleep 10
#sbatch eval_attacks.sh deit224_distill  /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 16 995
sleep 10
sbatch eval_attacks.sh effnet  /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 16  22
#sleep 10
#sbatch eval_attacks.sh vit384  /scratch/aaj458/data/ImageNet/val 100 20 1000  0.01 16  391

