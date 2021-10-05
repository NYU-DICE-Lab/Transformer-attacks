PATCHSIZE=1
NUM_PATCHES=5 #20
NUM_PATCHES_384=15 #59
EPSILON=1.0
DPATH=/scratch/aaj458/data/ImageNet/val
LR=0.1

# sbatch eval_attacks.sh vit224 /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300 0.1 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh wide-resnet /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300 0.1 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh deit224 /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300 0.1 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh deit224_distill  /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300 0.1 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh vit384  /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES_384 300  0.1 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh mlpb16  /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300  0.1 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh resnet50 /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300 0.2 $PATCHSIZE 0 $EPSILON
# sleep 2
# sbatch eval_attacks.sh resnet101d /scratch/aaj458/data/ImageNet/val 100 $NUM_PATCHES 300 0.1 $PATCHSIZE 0 $EPSILON