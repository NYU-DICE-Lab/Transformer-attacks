sbatch eval_attacks.sh vit224 /scratch/gbj221/data/ImageNet/val 100 20 300 0.01 8 
sleep 10
sbatch eval_attacks.sh bit_152_4 /scratch/gbj221/data/ImageNet/val 100 20 1000 0.01 8
sleep 10
#sbatch eval_attacks.sh wide-resnet /scratch/gbj221/data/ImageNet/val 100 20 1000 0.01 8 
#sleep 10
#sbatch eval_attacks.sh deit224 /scratch/gbj221/data/ImageNet/val 100 20 1000 0.01 8
#sleep 10
#sbatch eval_attacks.sh deit224_distill  /scratch/gbj221/data/ImageNet/val 100 20 1000 0.01 8
sleep 10
sbatch eval_attacks.sh effnet  /scratch/gbj221/data/ImageNet/val 100 20 1000 0.01 8
#sleep 10
#sbatch eval_attacks.sh vit384  /scratch/gbj221/data/ImageNet/val 100 20 1000  0.01 8

