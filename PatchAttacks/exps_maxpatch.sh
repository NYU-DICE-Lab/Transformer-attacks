PATCHSIZE=32
NUM_PATCHES_384=59 #59
NUM_PATCHES=20 #20
EPSILON=1.0

sbatch eval_attacks.sh vit224 /scratch/aaj458/data/ImageNet/val 8 $NUM_PATCHES 300 0.0005 $PATCHSIZE 0 $EPSILON
sleep 2
#sbatch eval_attacks.sh bit_152_4 /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 8
#sleep 10
sbatch eval_attacks.sh wide-resnet /scratch/aaj458/data/ImageNet/val 8 $NUM_PATCHES 300 0.0005 $PATCHSIZE 0 $EPSILON
sleep 2
sbatch eval_attacks.sh deit224 /scratch/aaj458/data/ImageNet/val 8 $NUM_PATCHES 300 0.0005 $PATCHSIZE 0 $EPSILON
sleep 2
sbatch eval_attacks.sh deit224_distill  /scratch/aaj458/data/ImageNet/val 8 $NUM_PATCHES 300 0.0005 $PATCHSIZE 0 $EPSILON
sleep 2
# #sbatch eval_attacks.sh effnet  /scratch/aaj458/data/ImageNet/val 100 20 1000 0.01 8 0 1.0
# #sleep 10
sbatch eval_attacks.sh vit384  /scratch/aaj458/data/ImageNet/val 8 $NUM_PATCHES_384 300  0.0005 $PATCHSIZE 0 $EPSILON
# #sbatch eval_attacks.sh mlpb16  /scratch/aaj458/data/ImageNet/val 100 40 300  0.01 8 0 1.0
sleep 2
sbatch eval_attacks.sh mlpb16  /scratch/aaj458/data/ImageNet/val 8 $NUM_PATCHES 300  0.0005 $PATCHSIZE 0 $EPSILON
sleep 2
#sbatch eval_attacks.sh mlpb16  /scratch/aaj458/data/ImageNet/val 100 20 300  0.01 32 0 1.0
#sleep 2
