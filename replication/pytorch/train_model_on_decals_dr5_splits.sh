#!/bin/bash
#SBATCH --job-name=py-vit                    # Job name
#SBATCH --output=%x_%A.log 
#SBATCH --mem=0                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot-kosio
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

RESULTS_DIR=/share/nas2/walml/repos/gz-decals-classifiers/results


# some other possible configurations, testing other architectures:

ARCHITECTURE='maxvit:tiny_224'
BATCH_SIZE=64
GPUS=1

# ARCHITECTURE='efficientnet'
# BATCH_SIZE=512
# GPUS=2
# # requires --mixed-precision to be set on A100s


# ARCHITECTURE='resnet_detectron'
# BATCH_SIZE=256
# GPUS=2
# mixed precision causes rare nan errors - not recommended!
# TODO need to update to ignore stochastic_depth_prob arg

# ARCHITECTURE='resnet_torchvision'
# BATCH_SIZE=256
# GPUS=2
# # mixed precision causes rare nan errors - not recommended!
# # only supports color (so you must add --color)
# TODO need to update to ignore stochastic_depth_prob arg

# be sure to add _color if appropriate
EXPERIMENT_DIR=$RESULTS_DIR/pytorch/dr5/${ARCHITECTURE}_maxvit_tiny_1xgpu

DATA_DIR=/share/nas2/walml/repos/_data/decals_dr5
DATASET=decals_dr5

# DATA_DIR=/share/nas2/walml/repos/_data/legs
# DATASET=legs

$PYTHON $ZOOBOT_DIR/replication/pytorch/train_model_on_decals_dr5_splits.py \
    --experiment-dir $EXPERIMENT_DIR \
    --data-dir $DATA_DIR \
    --dataset $DATASET \
    --architecture $ARCHITECTURE \
    --resize-size 224 \
    --batch-size $BATCH_SIZE \
    --gpus $GPUS \
    --mixed-precision
    
    #  \
    # --color

    #  \
    # --mixed-precision
