#!/bin/bash

#SBATCH --job-name=cifar_trades
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


singularity exec --nv \
	    --overlay ~/zc2157/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
python3 train_trades_cifar10.py --batch-size 320 --beta 0 --lam 6 --model-dir model-cifar-alp
"





